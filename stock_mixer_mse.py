import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val   = np.load("data/X_val.npy")
y_val   = np.load("data/y_val.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test:", y_test.shape)



class NumpyDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 10) -> (N, 10, 1)
        self.X = torch.from_numpy(X).float().unsqueeze(-1)
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = NumpyDataset(X_train, y_train)
val_ds   = NumpyDataset(X_val, y_val)
test_ds  = NumpyDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)


input_dim = X_train.shape[1]

class StockMixerBlock(nn.Module):
    def __init__(self, n_stocks, seq_len, hidden_dim, mlp_ratio=4):
        super().__init__()

        # 1) Indicator mixing: по последней оси D (hidden_dim)
        self.mlp_ind = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

        # 2) Temporal mixing: по оси T (seq_len)
        self.mlp_time = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len * mlp_ratio),
            nn.GELU(),
            nn.Linear(seq_len * mlp_ratio, seq_len),
        )

        # 3) Stock mixing: по оси N (n_stocks)
        self.mlp_stock = nn.Sequential(
            nn.LayerNorm(n_stocks),
            nn.Linear(n_stocks, n_stocks * mlp_ratio),
            nn.GELU(),
            nn.Linear(n_stocks * mlp_ratio, n_stocks),
        )

    def forward(self, x):
        # x: [B, N, T, D]

        # 1) Indicator mixing (по признакам)
        x = x + self.mlp_ind(x)  # [B, N, T, D]

        # 2) Temporal mixing (по времени)
        y = x.permute(0, 1, 3, 2)   # [B, N, D, T]
        y = self.mlp_time(y)        # последняя ось = T
        x = x + y.permute(0, 1, 3, 2)  # обратно [B, N, T, D]

        # 3) Stock mixing (по акциям)
        z = x.permute(0, 2, 3, 1)   # [B, T, D, N]
        z = self.mlp_stock(z)       # последняя ось = N
        x = x + z.permute(0, 3, 1, 2)  # обратно [B, N, T, D]

        return x


class StockMixer(nn.Module):
    """
    Упрощённая StockMixer-архитектура для твоего случая:
    - вход: вектор 10 лагов (B, 10)
    - сначала поднимаем его до (B, 10, d_model)
    - несколько Mixer-block'ов
    - затем глобальный readout и выход 1 скаляр (доходность)
    """
    def __init__(self,
                 seq_len: int = 10,
                 d_model: int = 32,
                 num_layers: int = 4,
                 expansion_factor: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # У тебя фактически 1 "признак" (ret), поэтому:
        # вход (B, T) -> (B, T, 1) -> Linear(1 -> d_model)
        self.input_proj = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            StockMixerBlock(seq_len, d_model,
                            expansion_factor=expansion_factor,
                            dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        # Можно сделать readout как flatten(T * D) -> 1
        self.head = nn.Linear(seq_len * d_model, 1)

    def forward(self, x):
        """
        x: (B, seq_len) = (B, 10)
        """
        # (B, T) -> (B, T, 1)
        x = x.unsqueeze(-1)
        # (B, T, 1) -> (B, T, d_model)
        x = self.input_proj(x)

        # Mixer-блоки
        for block in self.blocks:
            x = block(x)

        # Нормализация + readout
        x = self.norm_out(x)
        x = x.reshape(x.size(0), -1)   # (B, T*d_model)
        out = self.head(x)             # (B, 1)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StockMixer(
    seq_len = X_train.shape[1],  # должно быть 10
    d_model = 32,                # можешь поиграть: 32 / 64
    num_layers = 4,
    expansion_factor = 2,
    dropout = 0.1
).to(device)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_epoch(dataloader, model, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        preds = model(X_batch)
        loss = criterion_mse(preds, y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


n_epochs = 50
best_val_loss = float("inf")
best_state_dict = None

for epoch in range(1, n_epochs + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    val_loss = run_epoch(val_loader, model, optimizer=None)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = model.state_dict()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e}")


model.load_state_dict(best_state_dict)
print("Best val MSE:", best_val_loss)


def evaluate(dataloader, model):
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)

            mse = criterion_mse(preds, y_batch)


            batch_size = X_batch.size(0)
            mse_total += mse.item() * batch_size

            n_samples += batch_size

    return mse_total / n_samples

train_mse = evaluate(train_loader, model)
val_mse   = evaluate(val_loader, model)
test_mse  = evaluate(test_loader, model)

print("\nFinal metrics:")
print(f"Train: MSE={train_mse:.6e}")
print(f"Val:   MSE={val_mse:.6e}")
print(f"Test:  MSE={test_mse:.6e}")


def get_predictions(dataloader, model):
    model.eval()
    preds_list = []
    y_list = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_list.append(preds.cpu().numpy().reshape(-1))
            y_list.append(y_batch.numpy().reshape(-1))

    preds_all = np.concatenate(preds_list)
    y_all = np.concatenate(y_list)
    return preds_all, y_all

y_pred_test, y_true_test = get_predictions(test_loader, model)


signal = (y_pred_test > 0).astype(float)
strategy_ret = signal * y_true_test

mean_ret = strategy_ret.mean()
std_ret = strategy_ret.std(ddof=1)

print("\nStrategy stats (Test):")
print("Mean daily return:", mean_ret)
print("Std of daily return:", std_ret)

import math

if std_ret > 0:
    sharpe_daily = mean_ret / std_ret
    sharpe_annual = sharpe_daily * math.sqrt(252)
    print("Sharpe daily:", sharpe_daily)
    print("Sharpe annual:", sharpe_annual)
else:
    print("Std = 0, Sharpe не определён.")

# Pokuta za minusive dni
downside = strategy_ret[strategy_ret < 0]
if len(downside) > 0:
    downside_std = downside.std(ddof=1)
    sortino_daily = mean_ret / downside_std
    sortino_annual = sortino_daily * math.sqrt(252)
    print("Sortino daily:", sortino_daily)
    print("Sortino annual:", sortino_annual)
else:
    print("No minud days")

bh_ret = y_true_test

bh_mean = bh_ret.mean()
bh_std = bh_ret.std(ddof=1)

print("\nBuy & Hold stats (Test):")
print("Mean daily return:", bh_mean)
print("Std of daily return:", bh_std)

if bh_std > 0:
    bh_sharpe_daily = bh_mean / bh_std
    bh_sharpe_annual = bh_sharpe_daily * math.sqrt(252)
    print("Buy & Hold Sharpe annual:", bh_sharpe_annual)
else:
    print("Buy & Hold: Std = 0, Sharpe is not counted")
