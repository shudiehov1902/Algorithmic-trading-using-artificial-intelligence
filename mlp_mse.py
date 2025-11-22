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
        self.X = torch.from_numpy(X).float()
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

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # предсказываем одну величину — доходность
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_dim).to(device)

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
