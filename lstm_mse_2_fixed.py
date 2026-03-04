import copy
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="data")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--train_batch", type=int, default=256)
ap.add_argument("--eval_batch", type=int, default=512)
ap.add_argument("--cost_bps", type=float, default=10.0)
ap.add_argument("--charge_entry_cost", action="store_true", help="If set, charge cost on day-0 position entry.")
ap.add_argument("--select_metric", type=str, default="net_sortino",
                choices=["net_sortino","net_sharpe","alpha_ir_net","net_cum"],
                help="Metric to select (K,reb,buf) on validation. Robust = min over H1/H2 split.")
ap.add_argument("--fixed_policies", type=str, default="5,5,10;5,10,0;50,10,40",
                help="Semicolon-separated list of K,reb,buf triplets to report on TEST.")
args = ap.parse_args()

# 0) Reproducibility

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# 1) Load prepared dataset

X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))
X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

val_dates = np.load(os.path.join(args.data_dir, "val_dates.npy"), allow_pickle=True)
val_tickers = np.load(os.path.join(args.data_dir, "val_tickers.npy"), allow_pickle=True)
test_dates = np.load(os.path.join(args.data_dir, "test_dates.npy"), allow_pickle=True)
test_tickers = np.load(os.path.join(args.data_dir, "test_tickers.npy"), allow_pickle=True)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test:", y_test.shape)

seq_len = X_train.shape[1]  # 10 lags
n_features = 1



# 2) Dataset for LSTM: (N, 10) -> (N, 10, 1)

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



# 3) Search / trading config (GRID)

TOPK_GRID: List[int] = [5, 10, 20, 30, 40, 50, 75, 100]
REBALANCE_GRID: List[int] = [1, 5, 10]        # daily / weekly / bi-weekly
BUFFER_GRID: List[int] = [0, 10, 20, 40]      # hysteresis buffer

COST_BPS = args.cost_bps
CHARGE_ENTRY_COST = bool(args.charge_entry_cost)

VAL_SPLIT_DATE = pd.Timestamp("2023-07-01")   # robust H1/H2
EVAL_EVERY_EPOCH = 5                          # grid-search frequency



# 4) LSTM model

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # (B, H)
        return self.head(last)         # (B, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LSTMRegressor(input_size=1, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)

# ---- Train loss: MSE
criterion_train = nn.MSELoss()

# ---- Metrics
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# 5) Training / prediction helpers

def run_train_epoch(dataloader, model, optimizer) -> float:
    model.train()
    total = 0.0
    n = 0
    for Xb, yb in dataloader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion_train(pred, yb)
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
            ys.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)



# 6) Finance metrics

def sharpe_annual_series(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return float("nan")
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    return (mu / sd) * math.sqrt(252) if sd > 0 else float("nan")


def sortino_annual_series(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return float("nan")
    mu = float(daily_ret.mean())
    downside = daily_ret[daily_ret < 0.0]
    if len(downside) < 2:
        return float("nan")
    dd = float(downside.std(ddof=1))
    return (mu / dd) * math.sqrt(252) if dd > 0 else float("nan")


def cumulative_return(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) == 0:
        return float("nan")
    return float((1.0 + daily_ret).cumprod().iloc[-1] - 1.0)


def align_two(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    a2, b2 = a.align(b, join="inner")
    return a2.dropna(), b2.dropna()


def alpha_series(port_daily: pd.Series, mkt_daily: pd.Series) -> pd.Series:
    p, m = align_two(port_daily, mkt_daily)
    return p - m


def info_ratio_annual(port_daily: pd.Series, mkt_daily: pd.Series) -> float:
    a = alpha_series(port_daily, mkt_daily)
    return sharpe_annual_series(a)


def cumulative_excess_wealth(port_daily: pd.Series, mkt_daily: pd.Series) -> float:
    p, m = align_two(port_daily, mkt_daily)
    if len(p) == 0:
        return float("nan")
    Wp = (1.0 + p).cumprod().iloc[-1]
    Wm = (1.0 + m).cumprod().iloc[-1]
    return float(Wp - Wm)



# 7) Daily cache for faster backtest/grid

@dataclass
class DailyCache:
    dates: List[pd.Timestamp]
    tickers_sorted: List[List[str]]
    rank_maps: List[Dict[str, int]]
    ret_maps: List[Dict[str, float]]
    mkt_daily: pd.Series


def build_daily_cache(df: pd.DataFrame) -> DailyCache:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values("date")

    dates: List[pd.Timestamp] = []
    tickers_sorted: List[List[str]] = []
    rank_maps: List[Dict[str, int]] = []
    ret_maps: List[Dict[str, float]] = []

    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("pred", ascending=False)
        t = g["ticker"].astype(str).tolist()
        r = g["ret"].astype(float).to_numpy()
        dates.append(pd.Timestamp(d))
        tickers_sorted.append(t)
        rank_maps.append({tt: i + 1 for i, tt in enumerate(t)})
        ret_maps.append({tt: float(rr) for tt, rr in zip(t, r)})

    mkt = df.groupby("date", sort=True)["ret"].mean().sort_index()
    return DailyCache(dates=dates, tickers_sorted=tickers_sorted, rank_maps=rank_maps, ret_maps=ret_maps, mkt_daily=mkt)


def backtest_longonly_buffer_cost(
    cache: DailyCache,
    k: int,
    rebalance_every: int,
    buffer: int,
    cost_bps: float,
    charge_entry_cost: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    holdings: Set[str] = set()
    gross: List[float] = []
    net: List[float] = []
    turnover: List[float] = []

    for i in range(len(cache.dates)):
        top_list = cache.tickers_sorted[i]
        rank_map = cache.rank_maps[i]
        ret_map = cache.ret_maps[i]

        if i == 0:
            holdings = set(top_list[:k])
            to = 1.0 if charge_entry_cost else 0.0
        else:
            do_reb = (i % rebalance_every == 0)
            if do_reb:
                prev = holdings

                kept = set()
                thr = k + buffer
                for t in prev:
                    if rank_map.get(t, 10**9) <= thr:
                        kept.add(t)

                new_holdings = set(kept)
                for t in top_list:
                    if len(new_holdings) >= k:
                        break
                    if t not in new_holdings:
                        new_holdings.add(t)

                overlap = len(prev.intersection(new_holdings))
                to = 1.0 - overlap / float(k)
                holdings = new_holdings
            else:
                to = 0.0

        rr = [ret_map[t] for t in holdings if t in ret_map]
        r_g = float(np.mean(rr)) if len(rr) > 0 else 0.0

        c = to * (cost_bps / 10000.0)
        r_n = r_g - c

        gross.append(r_g)
        net.append(r_n)
        turnover.append(to)

    idx = pd.to_datetime(cache.dates)
    return pd.Series(gross, index=idx), pd.Series(net, index=idx), pd.Series(turnover, index=idx)



# 8) Training + per-epoch grid-search on VAL

n_epochs = args.epochs
best_val_mse = float("inf")  # tracking only

best_score_global = -float("inf")
best_state_dict = None
best_params = None  # (k, rebalance, buffer)


def robust_net_alpha_ir(net_series: pd.Series, mkt_series: pd.Series) -> float:
    net_series = net_series.dropna()
    mkt_series = mkt_series.dropna()

    net1 = net_series[net_series.index < VAL_SPLIT_DATE]
    m1 = mkt_series[mkt_series.index < VAL_SPLIT_DATE]
    net2 = net_series[net_series.index >= VAL_SPLIT_DATE]
    m2 = mkt_series[mkt_series.index >= VAL_SPLIT_DATE]

    ir1 = info_ratio_annual(net1, m1)
    ir2 = info_ratio_annual(net2, m2)
    return min(ir1, ir2)


for epoch in range(1, n_epochs + 1):
    train_mse = run_train_epoch(train_loader, model, optimizer)

    val_mse, val_mae = evaluate(val_loader, model)
    best_val_mse = min(best_val_mse, val_mse)

    do_eval = (epoch == 1) or (epoch % EVAL_EVERY_EPOCH == 0) or (epoch == n_epochs)
    if not do_eval:
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train MSE: {train_mse:.6e} | val MSE: {val_mse:.6e} | val MAE: {val_mae:.6e}")
        continue

    y_pred_val, y_true_val = get_predictions_np(val_loader, model)
    df_val = pd.DataFrame(
        {"date": pd.to_datetime(val_dates), "ticker": val_tickers.astype(str), "pred": y_pred_val, "ret": y_true_val}
    )
    cache_val = build_daily_cache(df_val)

    best_score_epoch = -float("inf")
    best_params_epoch: Optional[Tuple[int, int, int]] = None

    for reb in REBALANCE_GRID:
        for buf in BUFFER_GRID:
            for k in TOPK_GRID:
                _, net_s, _ = backtest_longonly_buffer_cost(
                    cache=cache_val,
                    k=k,
                    rebalance_every=reb,
                    buffer=buf,
                    cost_bps=COST_BPS,
                    charge_entry_cost=CHARGE_ENTRY_COST,
                )
                score = robust_metric(net_s, cache_val.mkt_daily, args.select_metric)
                if np.isfinite(score) and score > best_score_epoch:
                    best_score_epoch = score
                    best_params_epoch = (k, reb, buf)

    if np.isfinite(best_score_epoch) and best_score_epoch > best_score_global:
        best_score_global = best_score_epoch
        best_params = best_params_epoch
        best_state_dict = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch:3d} | train MSE: {train_mse:.6e} | val MSE: {val_mse:.6e} | val MAE: {val_mae:.6e} "
        f"| best VAL robust metric (epoch): {best_score_epoch:.4f} params(K,reb,buf)={best_params_epoch} "
        f"| best GLOBAL: {best_score_global:.4f} params={best_params}"
    )

if best_state_dict is None or best_params is None:
    raise RuntimeError("Selection failed: best_state_dict/best_params is None.")

model.load_state_dict(best_state_dict)
best_k, best_reb, best_buf = best_params

print("\n=== SELECTION SUMMARY ===")
print(f"Selection metric: {args.select_metric} (robust=min over H1/H2)")
print("Best val MSE (tracking only):", best_val_mse)
print("Best GLOBAL robust Val metric:", best_score_global)
print("Best params (K, rebalance_every, buffer):", best_params)
print("Cost config (selection):", f"COST_BPS={COST_BPS}, CHARGE_ENTRY_COST={CHARGE_ENTRY_COST}")


# 9) Final prediction metrics

train_mse, train_mae = evaluate(train_loader, model)
val_mse, val_mae = evaluate(val_loader, model)
test_mse, test_mae = evaluate(test_loader, model)

print("\nFinal prediction metrics:")
print(f"Train: MSE={train_mse:.6e}, MAE={train_mae:.6e}")
print(f"Val:   MSE={val_mse:.6e}, MAE={val_mae:.6e}")
print(f"Test:  MSE={test_mse:.6e}, MAE={test_mae:.6e}")



# 10) TEST backtest + Sortino + Year split + Fee sensitivity

y_pred_test, y_true_test = get_predictions_np(test_loader, model)
df_test = pd.DataFrame(
    {"date": pd.to_datetime(test_dates), "ticker": test_tickers.astype(str), "pred": y_pred_test, "ret": y_true_test}
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

def _parse_fixed_policies(s: str) -> List[Tuple[int,int,int]]:
    out: List[Tuple[int,int,int]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a,b,c = [int(x) for x in part.split(",")]
        out.append((a,b,c))
    return out

print("\n=== TEST RESULTS (FIXED POLICIES) ===")
for (k_f, reb_f, buf_f) in _parse_fixed_policies(args.fixed_policies):
    g_f, n_f, to_f = backtest_longonly_buffer_cost(
        cache=cache_test,
        k=k_f,
        rebalance_every=reb_f,
        buffer=buf_f,
        cost_bps=COST_BPS,
        charge_entry_cost=CHARGE_ENTRY_COST,
    )
    report_block(f"Policy K,reb,buf=({k_f},{reb_f},{buf_f})", g_f, n_f, mkt, to_f)



def report_by_year(title: str, gross: pd.Series, net: pd.Series, mkt_series: pd.Series, turnover: pd.Series):
    years = sorted(set(gross.dropna().index.year))
    print(f"\n=== {title} (by year) ===")
    for y in years:
        g_y = gross[gross.index.year == y]
        n_y = net[net.index.year == y]
        m_y = mkt_series[mkt_series.index.year == y]
        to_y = turnover[turnover.index.year == y]
        if len(g_y) < 30 or len(m_y) < 30:
            continue
        report_block(str(y), g_y, n_y, m_y, to_y)


report_by_year(
    title=f"TEST RESULTS Selected params K,reb,buf={best_params}",
    gross=gross_sel,
    net=net_sel,
    mkt_series=mkt,
    turnover=to_sel,
)


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
