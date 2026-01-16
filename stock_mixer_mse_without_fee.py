import copy
import math
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =====================================================
# 0. Reproducibility
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =====================================================
# 1. Loading prepared dataset
# =====================================================
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

val_dates = np.load("data/val_dates.npy", allow_pickle=True)
val_tickers = np.load("data/val_tickers.npy", allow_pickle=True)
test_dates = np.load("data/test_dates.npy", allow_pickle=True)
test_tickers = np.load("data/test_tickers.npy", allow_pickle=True)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test:", y_test.shape)

seq_len = X_train.shape[1]  # 10
n_features = 1


# =====================================================
# 2. Dataset: (N, 10) -> (N, 10, 1)
# =====================================================
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float().unsqueeze(-1)  # (N, T, 1)
        self.y = torch.from_numpy(y).float().view(-1, 1)    # (N, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


train_ds = SequenceDataset(X_train, y_train)
val_ds = SequenceDataset(X_val, y_val)
test_ds = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# =====================================================
# 2b. Backtest & selection config
# =====================================================
TOPK_GRID: List[int] = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
EVAL_EVERY_EPOCH = 1  # set 5 if too slow
VAL_SPLIT_DATE = pd.Timestamp("2023-07-01")  # H1 vs H2 stability


# =====================================================
# 3. StockMixer blocks
# =====================================================
class StockMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        expansion_factor_time: float = 2.0,
        expansion_factor_feature: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_time = int(seq_len * expansion_factor_time)
        hidden_feat = int(d_model * expansion_factor_feature)

        self.norm_time = nn.LayerNorm(d_model)
        self.norm_feat = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_time),
            nn.GELU(),
            nn.Linear(hidden_time, seq_len),
            nn.Dropout(dropout),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_feat),
            nn.GELU(),
            nn.Linear(hidden_feat, d_model),
            nn.Dropout(dropout),
        )

        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert T == self.seq_len and D == self.d_model

        # time mixing
        y = self.norm_time(x)
        y = y.transpose(1, 2).reshape(B * D, T)
        y = self.time_mlp(y)
        y = y.reshape(B, D, T).transpose(1, 2)
        x = x + y

        # feature mixing
        z = self.norm_feat(x)
        z = z.reshape(B * T, D)
        z = self.feature_mlp(z)
        z = z.reshape(B, T, D)
        x = x + z

        return x


class StockMixer(nn.Module):
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
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [
                StockMixerBlock(
                    seq_len=seq_len,
                    d_model=d_model,
                    expansion_factor_time=expansion_factor_time,
                    expansion_factor_feature=expansion_factor_feature,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        last = x[:, -1, :]
        return self.head(last)


# =====================================================
# 4. Helpers
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


def run_epoch(dataloader, model, optimizer=None) -> float:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total = 0.0
    n = 0
    for Xb, yb in dataloader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        pred = model(Xb)
        loss = criterion_mse(pred, yb)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        bs = Xb.size(0)
        total += loss.item() * bs
        n += bs

    return total / n


def evaluate(dataloader, model) -> Tuple[float, float]:
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    n = 0
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            pred = model(Xb)
            mse_total += criterion_mse(pred, yb).item() * Xb.size(0)
            mae_total += criterion_mae(pred, yb).item() * Xb.size(0)
            n += Xb.size(0)
    return mse_total / n, mae_total / n


def get_predictions_np(dataloader, model) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            pred = model(Xb)
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            ys.append(yb.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def sharpe_annual_series(daily_ret: pd.Series) -> float:
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    return (mu / sd) * math.sqrt(252) if sd > 0 else float("nan")


def sortino_annual_series(daily_ret: pd.Series) -> float:
    mu = float(daily_ret.mean())
    downside = daily_ret[daily_ret < 0]
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    return (mu / dd) * math.sqrt(252) if dd > 0 else float("nan")


def portfolio_daily_returns_from_df(df: pd.DataFrame, k_long: int) -> pd.Series:
    daily = []
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("pred", ascending=False)
        r = float(g.head(k_long)["ret"].mean())
        daily.append((d, r))
    return pd.Series(dict(daily)).sort_index()


def market_equal_weight_daily_returns_from_df(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date", sort=True)["ret"].mean().sort_index()


def info_ratio_annual(port_daily: pd.Series, mkt_daily: pd.Series) -> float:
    port_daily, mkt_daily = port_daily.align(mkt_daily, join="inner")
    alpha = port_daily - mkt_daily
    mu = float(alpha.mean())
    sd = float(alpha.std(ddof=1))
    return (mu / sd) * math.sqrt(252) if sd > 0 else float("nan")


def cumulative_return(daily_ret: pd.Series) -> float:
    return float((1.0 + daily_ret).cumprod().iloc[-1] - 1.0)


# =====================================================
# 5. Training: select by robust Val alpha IR (min of H1/H2)
# =====================================================
n_epochs = 50
best_val_loss = float("inf")
best_val_score = -float("inf")  # robust alpha IR score
best_state_dict = None
best_k_long = None

for epoch in range(1, n_epochs + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    val_loss = run_epoch(val_loader, model, optimizer=None)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    do_eval = (EVAL_EVERY_EPOCH == 1) or (epoch % EVAL_EVERY_EPOCH == 0) or (epoch == 1)

    val_score = float("nan")
    k_star = None

    if do_eval:
        y_pred_val, y_true_val = get_predictions_np(val_loader, model)
        df_val = pd.DataFrame(
            {
                "date": pd.to_datetime(val_dates),
                "ticker": val_tickers,
                "pred": y_pred_val.reshape(-1),
                "ret": y_true_val.reshape(-1),
            }
        )

        # market daily on VAL
        mkt_all = market_equal_weight_daily_returns_from_df(df_val)

        best_score_this = -float("inf")
        best_k_this = None

        for k in TOPK_GRID:
            p_all = portfolio_daily_returns_from_df(df_val, k_long=k)

            # split into halves (H1, H2)
            p1 = p_all[p_all.index < VAL_SPLIT_DATE]
            m1 = mkt_all[mkt_all.index < VAL_SPLIT_DATE]
            p2 = p_all[p_all.index >= VAL_SPLIT_DATE]
            m2 = mkt_all[mkt_all.index >= VAL_SPLIT_DATE]

            ir1 = info_ratio_annual(p1, m1)
            ir2 = info_ratio_annual(p2, m2)

            # robust criterion: maximize worst half
            score = min(ir1, ir2)

            if np.isfinite(score) and score > best_score_this:
                best_score_this = score
                best_k_this = k

        val_score = best_score_this
        k_star = best_k_this

        if np.isfinite(val_score) and val_score > best_val_score:
            best_val_score = val_score
            best_k_long = int(k_star)
            best_state_dict = copy.deepcopy(model.state_dict())

    if epoch % 5 == 0 or epoch == 1:
        if do_eval:
            print(
                f"Epoch {epoch:3d} | train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e} "
                f"| robust Val alpha IR (bestK): {val_score:.4f} (K={k_star}) "
                f"| best robust IR: {best_val_score:.4f} (K={best_k_long})"
            )
        else:
            print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e}")

if best_state_dict is None or best_k_long is None:
    raise RuntimeError("Selection failed: best_state_dict/best_k_long is None. Check val_dates/val_tickers files.")

model.load_state_dict(best_state_dict)
print("Best val MSE:", best_val_loss)
print("Best robust Val alpha IR:", best_val_score)
print("Best selected K on Val:", best_k_long)


# =====================================================
# 6. Final metrics
# =====================================================
train_mse, train_mae = evaluate(train_loader, model)
val_mse, val_mae = evaluate(val_loader, model)
test_mse, test_mae = evaluate(test_loader, model)

print("\nFinal metrics:")
print(f"Train: MSE={train_mse:.6e}, MAE={train_mae:.6e}")
print(f"Val:   MSE={val_mse:.6e}, MAE={val_mae:.6e}")
print(f"Test:  MSE={test_mse:.6e}, MAE={test_mae:.6e}")


# =====================================================
# 7. Test backtest
# =====================================================
y_pred_test, y_true_test = get_predictions_np(test_loader, model)

df_test = pd.DataFrame(
    {
        "date": pd.to_datetime(test_dates),
        "ticker": test_tickers,
        "pred": y_pred_test.reshape(-1),
        "ret": y_true_test.reshape(-1),
    }
)

mkt_test = market_equal_weight_daily_returns_from_df(df_test)

port_best = portfolio_daily_returns_from_df(df_test, k_long=best_k_long)
port_50 = portfolio_daily_returns_from_df(df_test, k_long=50)

# optional long-short (for info only)
def long_short_50(df: pd.DataFrame, k: int = 50) -> pd.Series:
    daily = []
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("pred", ascending=False)
        r = float(g.head(k)["ret"].mean() - g.tail(k)["ret"].mean())
        daily.append((d, r))
    return pd.Series(dict(daily)).sort_index()

port_ls50 = long_short_50(df_test, k=50)

print("\n=== DAILY PORTFOLIO BACKTEST (TEST) ===")
print(f"Long-only top{best_k_long:>3} Sharpe annual:", sharpe_annual_series(port_best))
print(f"Long-only top{best_k_long:>3} Sortino annual:", sortino_annual_series(port_best))
print("Long-only top50  Sharpe annual:", sharpe_annual_series(port_50))
print("Long-only top50  Sortino annual:", sortino_annual_series(port_50))
print("Long-short 50/50 Sharpe annual:", sharpe_annual_series(port_ls50))
print("Long-short 50/50 Sortino annual:", sortino_annual_series(port_ls50))
print("Equal-weight mkt Sharpe annual:", sharpe_annual_series(mkt_test))
print("Equal-weight mkt Sortino annual:", sortino_annual_series(mkt_test))

print("\nCumulative return (Test):")
print(f"Long-only top{best_k_long:>3}:", cumulative_return(port_best))
print("Long-only top50 :", cumulative_return(port_50))
print("Long-short 50/50:", cumulative_return(port_ls50))
print("Market eq-weight:", cumulative_return(mkt_test))

print("\nTEST Alpha IR annual:")
print(f"Alpha IR (top{best_k_long:>3}):", info_ratio_annual(port_best, mkt_test))
print("Alpha IR (top 50):", info_ratio_annual(port_50, mkt_test))

alpha_best = (port_best - mkt_test).dropna()
alpha_50 = (port_50 - mkt_test).dropna()
print("\nAlpha (Long-only - Market) cumulative (Test):")
print(f"Alpha cum (top{best_k_long:>3}):", cumulative_return(alpha_best))
print("Alpha cum (top 50):", cumulative_return(alpha_50))
