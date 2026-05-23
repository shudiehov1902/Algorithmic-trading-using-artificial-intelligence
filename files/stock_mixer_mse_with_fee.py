import copy
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from portfolio_utils import (
    DailyCache,
    align_two,
    alpha_series,
    backtest_longonly_buffer_cost,
    build_daily_cache,
    cumulative_excess_wealth,
    cumulative_return,
    info_ratio_annual,
    robust_metric,
    sharpe_annual_series,
    sortino_annual_series,
)


def parse_int_list(text: str) -> List[int]:
    vals = [s.strip() for s in text.split(",") if s.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("List must not be empty.")
    try:
        return [int(v) for v in vals]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {text}") from e


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--train_batch", type=int, default=256)
ap.add_argument("--eval_batch", type=int, default=512)
ap.add_argument("--grid_k", "--grid_K", dest="grid_k", type=parse_int_list, default=[5, 10, 20, 30, 40, 50, 75, 100])
ap.add_argument("--grid_reb", type=parse_int_list, default=[1, 5, 10])
ap.add_argument("--grid_buf", type=parse_int_list, default=[0, 10, 20, 40])
ap.add_argument("--cost_bps", type=float, default=10.0)
ap.add_argument("--charge_entry_cost", dest="charge_entry_cost", action="store_true")
ap.add_argument("--no_charge_entry_cost", dest="charge_entry_cost", action="store_false")
ap.set_defaults(charge_entry_cost=True)
ap.add_argument("--val_split_date", type=str, default="2023-07-01")
ap.add_argument("--eval_every_epoch", type=int, default=5)
ap.add_argument(
    "--select_metric",
    type=str,
    default="net_sortino",
    choices=["net_sortino", "net_sharpe", "alpha_ir_net", "net_cum"],
)
ap.add_argument("--max_avg_turnover", type=float, default=None)
args = ap.parse_args()

TOPK_GRID: List[int] = args.grid_k
REBALANCE_GRID: List[int] = args.grid_reb
BUFFER_GRID: List[int] = args.grid_buf
VAL_SPLIT_DATE = pd.Timestamp(args.val_split_date)
EVAL_EVERY_EPOCH: int = max(1, int(args.eval_every_epoch))
COST_BPS: float = float(args.cost_bps)
CHARGE_ENTRY_COST: bool = bool(args.charge_entry_cost)

# =====================================================
# 0) Reproducibility
# =====================================================
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =====================================================
# 1) Load prepared dataset
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

seq_len = X_train.shape[1]  # 10 lags
n_features = 1


# =====================================================
# 2) Dataset: (N, 10) -> (N, 10, 1)
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

train_loader = DataLoader(train_ds, batch_size=args.train_batch, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.eval_batch, shuffle=False)


# =====================================================
# 3) Search / trading config (GRID)
# =====================================================
# TOPK/REBALANCE/BUFFER/COST_* and split/eval configs are set by CLI above.


# =====================================================
# 4) StockMixer model
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# =====================================================
# 5) Training / prediction helpers
# =====================================================
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


# =====================================================
# 8) Training + per-epoch grid-search on VAL
#    Objective: robust Val metric (min of H1/H2)
# =====================================================
n_epochs = args.epochs
best_val_loss = float("inf")

best_score_global = -float("inf")
best_state_dict = None
best_params = None  # (k, rebalance, buffer)


for epoch in range(1, n_epochs + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    val_loss = run_epoch(val_loader, model, optimizer=None)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    do_eval = (epoch == 1) or (epoch % EVAL_EVERY_EPOCH == 0) or (epoch == n_epochs)
    if not do_eval:
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e}")
        continue

    y_pred_val, y_true_val = get_predictions_np(val_loader, model)
    df_val = pd.DataFrame(
        {
            "date": pd.to_datetime(val_dates),
            "ticker": val_tickers.astype(str),
            "pred": y_pred_val.reshape(-1),
            "ret": y_true_val.reshape(-1),
        }
    )
    cache_val = build_daily_cache(df_val)

    best_score_epoch = -float("inf")
    best_params_epoch: Optional[Tuple[int, int, int]] = None

    for reb in REBALANCE_GRID:
        for buf in BUFFER_GRID:
            for k in TOPK_GRID:
                _, net_s, to_s = backtest_longonly_buffer_cost(
                    cache=cache_val,
                    k=k,
                    rebalance_every=reb,
                    buffer=buf,
                    cost_bps=COST_BPS,
                    charge_entry_cost=CHARGE_ENTRY_COST,
                )
                avg_to = float(to_s.mean()) if len(to_s) else float('nan')
                if args.max_avg_turnover is not None and np.isfinite(avg_to) and avg_to > args.max_avg_turnover:
                    continue
                score = robust_metric(net_s, cache_val.mkt_daily, VAL_SPLIT_DATE, args.select_metric)
                if np.isfinite(score) and score > best_score_epoch:
                    best_score_epoch = score
                    best_params_epoch = (k, reb, buf)

    if np.isfinite(best_score_epoch) and best_score_epoch > best_score_global:
        best_score_global = best_score_epoch
        best_params = best_params_epoch
        best_state_dict = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch:3d} | train MSE: {train_loss:.6e} | val MSE: {val_loss:.6e} "
        f"| best VAL robust metric (epoch): {best_score_epoch:.4f} params(K,reb,buf)={best_params_epoch} "
        f"| best GLOBAL: {best_score_global:.4f} params={best_params}"
    )

if best_state_dict is None or best_params is None:
    raise RuntimeError("Selection failed: best_state_dict/best_params is None (check val_* files).")

model.load_state_dict(best_state_dict)
best_k, best_reb, best_buf = best_params

print("\n=== SELECTION SUMMARY ===")
print(f"Selection metric: {args.select_metric} (robust=min over two val halves)")
print("Best val MSE (tracking only):", best_val_loss)
print("Best GLOBAL robust Val metric:", best_score_global)
print("Best params (K, rebalance_every, buffer):", best_params)
print("Cost config (selection):", f"COST_BPS={COST_BPS}, CHARGE_ENTRY_COST={CHARGE_ENTRY_COST}")


# =====================================================
# 9) Final prediction metrics
# =====================================================
train_mse, train_mae = evaluate(train_loader, model)
val_mse, val_mae = evaluate(val_loader, model)
test_mse, test_mae = evaluate(test_loader, model)

print("\nFinal prediction metrics:")
print(f"Train: MSE={train_mse:.6e}, MAE={train_mae:.6e}")
print(f"Val:   MSE={val_mse:.6e}, MAE={val_mae:.6e}")
print(f"Test:  MSE={test_mse:.6e}, MAE={test_mae:.6e}")


# =====================================================
# 10) TEST backtest with selected params + comparisons + Sortino + Year split + Fee sensitivity
# =====================================================
y_pred_test, y_true_test = get_predictions_np(test_loader, model)
df_test = pd.DataFrame(
    {
        "date": pd.to_datetime(test_dates),
        "ticker": test_tickers.astype(str),
        "pred": y_pred_test.reshape(-1),
        "ret": y_true_test.reshape(-1),
    }
)
cache_test = build_daily_cache(df_test)
mkt = cache_test.mkt_daily

gross_sel, net_sel, to_sel = backtest_longonly_buffer_cost(
    cache=cache_test,
    k=best_k,
    rebalance_every=best_reb,
    buffer=best_buf,
    cost_bps=COST_BPS,
    charge_entry_cost=CHARGE_ENTRY_COST,
)


def run_fixed(k: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return backtest_longonly_buffer_cost(
        cache=cache_test,
        k=k,
        rebalance_every=best_reb,
        buffer=best_buf,
        cost_bps=COST_BPS,
        charge_entry_cost=CHARGE_ENTRY_COST,
    )


gross_5, net_5, to_5 = run_fixed(5)
gross_40, net_40, to_40 = run_fixed(40)
gross_50, net_50, to_50 = run_fixed(50)


def report_block(title: str, port_gross: pd.Series, port_net: pd.Series, mkt_series: pd.Series, turnover: pd.Series):
    a_g = alpha_series(port_gross, mkt_series)
    a_n = alpha_series(port_net, mkt_series)

    print(f"\n=== {title} ===")
    print("Market   Sharpe:", sharpe_annual_series(mkt_series), "Sortino:", sortino_annual_series(mkt_series), "Cum:", cumulative_return(mkt_series))

    print("GROSS    Sharpe:", sharpe_annual_series(port_gross), "Sortino:", sortino_annual_series(port_gross),
          "Cum:", cumulative_return(port_gross), "Excess wealth:", cumulative_excess_wealth(port_gross, mkt_series))
    print("NET      Sharpe:", sharpe_annual_series(port_net), "Sortino:", sortino_annual_series(port_net),
          "Cum:", cumulative_return(port_net), "Excess wealth:", cumulative_excess_wealth(port_net, mkt_series))

    print("Alpha(G) Sharpe/IR:", sharpe_annual_series(a_g), "Alpha(G) Sortino:", sortino_annual_series(a_g))
    print("Alpha(N) Sharpe/IR:", sharpe_annual_series(a_n), "Alpha(N) Sortino:", sortino_annual_series(a_n))
    print("Avg turnover:", float(turnover.mean()))


report_block(
    title=f"TEST RESULTS (Selected params K,reb,buf={best_params})",
    port_gross=gross_sel,
    port_net=net_sel,
    mkt_series=mkt,
    turnover=to_sel,
)

print("\n=== TEST COMPARISONS (same reb/buf, different K) ===")
print("K=5   NET Sharpe:", sharpe_annual_series(net_5), "NET Sortino:", sortino_annual_series(net_5),
      "NET Cum:", cumulative_return(net_5), "Alpha IR:", info_ratio_annual(net_5, mkt), "avg TO:", float(to_5.mean()))
print("K=40  NET Sharpe:", sharpe_annual_series(net_40), "NET Sortino:", sortino_annual_series(net_40),
      "NET Cum:", cumulative_return(net_40), "Alpha IR:", info_ratio_annual(net_40, mkt), "avg TO:", float(to_40.mean()))
print("K=50  NET Sharpe:", sharpe_annual_series(net_50), "NET Sortino:", sortino_annual_series(net_50),
      "NET Cum:", cumulative_return(net_50), "Alpha IR:", info_ratio_annual(net_50, mkt), "avg TO:", float(to_50.mean()))


# -----------------------
# Year-by-year breakdown
# -----------------------
def report_by_year(title: str, gross: pd.Series, net: pd.Series, mkt_series: pd.Series, turnover: pd.Series):
    years = sorted(set(gross.dropna().index.year))
    print(f"\n=== {title} (by year) ===")
    for y in years:
        g_y = gross[gross.index.year == y]
        n_y = net[net.index.year == y]
        m_y = mkt_series[mkt_series.index.year == y]
        to_y = turnover[turnover.index.year == y]
        if len(g_y) < 30 or len(m_y) < 30:
            # skip too-short year fragments
            continue
        report_block(
            title=f"{y}",
            port_gross=g_y,
            port_net=n_y,
            mkt_series=m_y,
            turnover=to_y,
        )


report_by_year(
    title=f"TEST RESULTS Selected params K,reb,buf={best_params}",
    gross=gross_sel,
    net=net_sel,
    mkt_series=mkt,
    turnover=to_sel,
)


# -----------------------
# Fee sensitivity (keep holdings/turnover fixed)
# -----------------------
def apply_cost_from_turnover(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    gross2, to2 = align_two(gross, turnover)
    return gross2 - to2 * (cost_bps / 10000.0)


print("\n=== FEE SENSITIVITY (Selected params, same turnover) ===")
for bps in [0.0, 5.0, 10.0, 20.0]:
    net_bps = apply_cost_from_turnover(gross_sel, to_sel, bps)
    print(
        f"cost_bps={bps:>4.0f} | NET Sharpe={sharpe_annual_series(net_bps):.4f} "
        f"| NET Sortino={sortino_annual_series(net_bps):.4f} "
        f"| NET Cum={cumulative_return(net_bps):.4f} "
        f"| Alpha IR={info_ratio_annual(net_bps, mkt):.4f}"
    )
