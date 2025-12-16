import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# 1. Загрузка данных из готовых .npy (как в MLP/LSTM)
# =====================================================
X_train = np.load("data/X_train.npy")  # shape: (N_train, 10)
y_train = np.load("data/y_train.npy")  # shape: (N_train,)
X_val   = np.load("data/X_val.npy")
y_val   = np.load("data/y_val.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape,  "y_test:", y_test.shape)

seq_len = X_train.shape[1]   # у тебя 10 лагов
n_features = 1               # один признак на шаге времени: ret_lag_t


# =====================================================
# 2. Dataset: превращаем (N, 10) → (N, 10, 1)
# =====================================================

class SequenceDataset(Dataset):
    """
    Каждый объект = одна временная последовательность длины seq_len
    X: (seq_len, n_features), y: скалярный таргет (доходность следующего дня)
    """
    def __init__(self, X, y):
        # X: (N, seq_len) → (N, seq_len, 1)
        self.X = torch.from_numpy(X).float().unsqueeze(-1)
        # y: (N,) → (N, 1)
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = SequenceDataset(X_train, y_train)
val_ds   = SequenceDataset(X_val,   y_val)
test_ds  = SequenceDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)


# =====================================================
# 3. StockMixer-подобный блок
#    (MLP по времени + MLP по признакам с residual)
# =====================================================

class StockMixerBlock(nn.Module):
    """
    Блок mixer-а:
      1) Time-mixing: MLP вдоль оси времени (T)
      2) Feature-mixing: MLP вдоль оси признаков (D = d_model)

    Вход/выход: x формы (B, T, D)
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        expansion_factor_time: float = 2.0,
        expansion_factor_feature: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        hidden_time = int(seq_len * expansion_factor_time)
        hidden_feat = int(d_model * expansion_factor_feature)

        # Нормализации по последней оси (D)
        self.norm_time = nn.LayerNorm(d_model)
        self.norm_feat = nn.LayerNorm(d_model)

        # MLP по времени: линейный слой вдоль оси T
        # Будем работать с тензором формы (B * D, T)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_time),
            nn.GELU(),
            nn.Linear(hidden_time, seq_len),
            nn.Dropout(dropout),
        )

        # MLP по признакам: линейный слой вдоль оси D
        # Будем работать с тензором формы (B * T, D)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_feat),
            nn.GELU(),
            nn.Linear(hidden_feat, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        assert T == self.seq_len, f"ожидали T={self.seq_len}, получили {T}"
        assert D == self.d_model, f"ожидали D={self.d_model}, получили {D}"

        # ---- 1. Time mixing ----
        # нормализация по признакам
        y = self.norm_time(x)                 # (B, T, D)
        # переносим оси: (B, T, D) → (B, D, T)
        y = y.transpose(1, 2)                 # (B, D, T)
        # схлопываем B и D
        y = y.reshape(B * D, T)               # (B*D, T)
        # MLP по времени
        y = self.time_mlp(y)                  # (B*D, T)
        # разворачиваем обратно
        y = y.reshape(B, D, T).transpose(1, 2)  # (B, T, D)
        # residual
        x = x + y

        # ---- 2. Feature mixing ----
        z = self.norm_feat(x)                 # (B, T, D)
        z = z.reshape(B * T, D)               # (B*T, D)
        z = self.feature_mlp(z)               # (B*T, D)
        z = z.reshape(B, T, D)                # (B, T, D)
        x = x + z

        return x


# =====================================================
# 4. Полная модель StockMixer
# =====================================================

class StockMixer(nn.Module):
    """
    Упрощённая реализация StockMixer:
      - линейная проекция признаков в d_model
      - несколько StockMixerBlock
      - LayerNorm
      - голова, которая берёт последний временной шаг и предсказывает r_t
    """
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 64,
        num_layers: int = 4,
        expansion_factor_time: float = 2.0,
        expansion_factor_feature: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Входная проекция: (features → d_model)
        self.input_proj = nn.Linear(n_features, d_model)

        # Стек mixer-блоков
        self.blocks = nn.ModuleList([
            StockMixerBlock(
                seq_len=seq_len,
                d_model=d_model,
                expansion_factor_time=expansion_factor_time,
                expansion_factor_feature=expansion_factor_feature,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # предсказываем r_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_features) → (B, 1)
        """
        # (B, T, F) → (B, T, D)
        x = self.input_proj(x)

        # пропускаем через mixer-блоки
        for block in self.blocks:
            x = block(x)    # (B, T, D)

        x = self.final_norm(x)  # (B, T, D)

        # берём последнюю точку по времени
        last = x[:, -1, :]      # (B, D)
        out = self.head(last)   # (B, 1)
        return out


# =====================================================
# 5. Обучение и оценка
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = StockMixer(
    seq_len=seq_len,
    n_features=n_features,
    d_model=64,
    num_layers=4,
    expansion_factor_time=2.0,
    expansion_factor_feature=2.0,
    dropout=0.1,
).to(device)

criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_epoch(dataloader, model, optimizer=None):
    """
    Одна эпоха: если optimizer is None → eval, иначе → train
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)  # (B, T, 1)
        y_batch = y_batch.to(device)  # (B, 1)

        if optimizer is not None:
            optimizer.zero_grad()

        preds = model(X_batch)       # (B, 1)
        loss = criterion_mse(preds, y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


# ---- цикл обучения ----

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

# загружаем лучшую по валидации модель
model.load_state_dict(best_state_dict)
print("Best val MSE:", best_val_loss)


# =====================================================
# 6. Оценка: MSE/MAE + Sharpe/Sortino
# =====================================================

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
            mae = criterion_mae(preds, y_batch)

            batch_size = X_batch.size(0)
            mse_total += mse.item() * batch_size
            mae_total += mae.item() * batch_size
            n_samples += batch_size

    return mse_total / n_samples, mae_total / n_samples


train_mse, train_mae = evaluate(train_loader, model)
val_mse,   val_mae   = evaluate(val_loader, model)
test_mse,  test_mae  = evaluate(test_loader, model)

print("\nFinal metrics:")
print(f"Train: MSE={train_mse:.6e}, MAE={train_mae:.6e}")
print(f"Val:   MSE={val_mse:.6e}, MAE={val_mae:.6e}")
print(f"Test:  MSE={test_mse:.6e}, MAE={test_mae:.6e}")


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

# простая стратегия: если прогноз > 0 -> long, иначе 0
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

# Sortino
downside = strategy_ret[strategy_ret < 0]
if len(downside) > 0:
    downside_std = downside.std(ddof=1)
    sortino_daily = mean_ret / downside_std
    sortino_annual = sortino_daily * math.sqrt(252)
    print("Sortino daily:", sortino_daily)
    print("Sortino annual:", sortino_annual)
else:
    print("Нет отрицательных дней => Sortino формально бесконечен.")

# Baseline: Buy & Hold на тех же днях теста
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
    print("Buy & Hold: Std = 0, Sharpe не определён.")
