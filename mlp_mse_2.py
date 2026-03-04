"""MLP (MSE) + portfolio selection with turnover-aware validation.

Fixes vs older mlp_mae_2.py:
- Adds CLI args (seed, batch sizes, grids, cost model).
- Deterministic seeding for torch/numpy/random + DataLoader generator.
- Validation selection supports different selection metrics (net_sortino, net_sharpe, net_alpha_ir).
- Turnover-aware selection: either a hard max_avg_turnover constraint, or a soft penalty.

Typical usage:
  python3 mlp_mae_2_fixed.py --charge_entry_cost --cost_bps 10 --select_metric net_sortino \
    --grid_reb 5,10,20 --grid_buf 20,40 --grid_K 10,20,40,50 --max_avg_turnover 0.12

If you omit --max_avg_turnover, use --turnover_penalty (soft penalty) to bias towards lower turnover.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(x.std(ddof=1))


def sharpe(daily: np.ndarray, ann: int = 252) -> float:
    daily = np.asarray(daily, dtype=float)
    mu = float(daily.mean())
    sd = safe_std(daily)
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(ann)


def sortino(daily: np.ndarray, ann: int = 252) -> float:
    daily = np.asarray(daily, dtype=float)
    mu = float(daily.mean())
    downside = daily[daily < 0.0]
    dd = safe_std(downside) if downside.size > 1 else 0.0
    if dd <= 1e-12:
        return 0.0
    return (mu / dd) * math.sqrt(ann)


def alpha_ir(net: np.ndarray, mkt: np.ndarray, ann: int = 252) -> float:
    net = np.asarray(net, dtype=float)
    mkt = np.asarray(mkt, dtype=float)
    a = net - mkt
    mu = float(a.mean())
    sd = safe_std(a)
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(ann)


# -----------------------------
# Dataset
# -----------------------------

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()               # (N, D)
        self.y = torch.from_numpy(y).float().view(-1, 1)   # (N, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Trading/backtest
# -----------------------------

@dataclass(frozen=True)
class Policy:
    K: int
    rebalance_every: int
    buffer: int


def build_df(pred: np.ndarray, y: np.ndarray, dates: np.ndarray, tickers: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "ticker": tickers.astype(str),
        "pred": pred.astype(float),
        "ret": y.astype(float),
    })
    # In case of duplicates, keep stable ordering
    df = df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    return df


def simulate_policy(
    df: pd.DataFrame,
    policy: Policy,
    cost_bps: float,
    charge_entry_cost: bool,
    charge_exit_cost: bool,
) -> Tuple[pd.Series, pd.Series, float]:
    """Return (net_daily, market_daily, avg_turnover).

    - market_daily = equal-weight return across all tickers per day.
    - net_daily = strategy return after costs.
    - avg_turnover = mean( turnover_t ) over all days, turnover_t in [0,1].

    Turnover is approximated as: 1 - overlap/|portfolio| on rebalance days, else 0.
    """
    cost_rate = float(cost_bps) / 1e4

    net_by_day: List[float] = []
    mkt_by_day: List[float] = []
    to_by_day: List[float] = []

    prev_port: List[str] = []
    port: List[str] = []
    days = df["date"].sort_values().unique()

    for t, day in enumerate(days):
        dday = df[df["date"] == day]

        # market (equal-weight)
        mkt_ret = float(dday["ret"].mean())

        do_reb = (t % policy.rebalance_every == 0)
        if do_reb:
            # rank by prediction
            ranked = dday.sort_values("pred", ascending=False, kind="mergesort")

            # hysteresis buffer: keep previous names if within top K+buffer
            if policy.buffer > 0 and prev_port:
                top_keep = set(ranked.head(policy.K + policy.buffer)["ticker"].tolist())
                kept = [x for x in prev_port if x in top_keep]
            else:
                kept = []

            need = max(0, policy.K - len(kept))
            top_candidates = ranked[~ranked["ticker"].isin(kept)].head(need)["ticker"].tolist()
            port = kept + top_candidates

            # turnover
            prev_set = set(prev_port)
            cur_set = set(port)
            overlap = len(prev_set & cur_set)
            turnover = 1.0 - (overlap / max(1, len(cur_set)))

            # costs
            # entry = names newly bought; exit = names removed
            entered = len(cur_set - prev_set)
            exited = len(prev_set - cur_set)
            cost = 0.0
            if charge_entry_cost:
                cost += entered * cost_rate
            if charge_exit_cost:
                cost += exited * cost_rate

            prev_port = port
        else:
            turnover = 0.0
            cost = 0.0

        if port:
            port_ret = float(dday[dday["ticker"].isin(port)]["ret"].mean())
        else:
            port_ret = 0.0

        net_ret = port_ret - cost

        net_by_day.append(net_ret)
        mkt_by_day.append(mkt_ret)
        to_by_day.append(turnover)

    idx = pd.to_datetime(days)
    return pd.Series(net_by_day, index=idx), pd.Series(mkt_by_day, index=idx), float(np.mean(to_by_day))


def robust_metric_over_halves(
    net: pd.Series,
    mkt: pd.Series,
    split_date: pd.Timestamp,
    metric: str,
) -> float:
    """Robust metric = min(metric on H1, metric on H2)."""
    h1_mask = net.index < split_date
    h2_mask = ~h1_mask

    def metric_value(x_net: np.ndarray, x_mkt: np.ndarray) -> float:
        if metric == "net_sortino":
            return sortino(x_net)
        if metric == "net_sharpe":
            return sharpe(x_net)
        if metric == "net_alpha_ir":
            return alpha_ir(x_net, x_mkt)
        raise ValueError(f"Unknown metric: {metric}")

    if h1_mask.sum() < 20 or h2_mask.sum() < 20:
        # too short -> use full period
        return metric_value(net.values, mkt.values)

    m1 = metric_value(net[h1_mask].values, mkt[h1_mask].values)
    m2 = metric_value(net[h2_mask].values, mkt[h2_mask].values)
    return float(min(m1, m2))


# -----------------------------
# Train / eval loops
# -----------------------------

def run_train_epoch(loader: DataLoader, model: nn.Module, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.MSELoss()  # MSE objective
    tot = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        tot += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return tot / max(1, n)


@torch.no_grad()
def predict(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
        outs.append(pred)
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def eval_pred_metrics(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        mse_sum += float(mse_fn(pred, yb).item())
        mae_sum += float(mae_fn(pred, yb).item())
        n += xb.size(0)
    return mse_sum / max(1, n), mae_sum / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_batch", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--grid_K", type=str, default="5,10,20,30,40,50,75,100")
    # NOTE: removed daily rebalance by default to avoid pathological turnover
    p.add_argument("--grid_reb", type=str, default="5,10,20")
    p.add_argument("--grid_buf", type=str, default="0,10,20,40")

    p.add_argument("--select_metric", type=str, default="net_sortino",
                   choices=["net_sortino", "net_sharpe", "net_alpha_ir"])
    p.add_argument("--val_split_date", type=str, default="2023-07-01")
    p.add_argument("--eval_every_epoch", type=int, default=5)

    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--charge_entry_cost", action="store_true")
    p.add_argument("--charge_exit_cost", action="store_true")

    # Turnover control
    p.add_argument("--max_avg_turnover", type=float, default=0.12,
                   help="Hard constraint on avg turnover during validation policy selection. Set <=0 to disable.")
    p.add_argument("--turnover_penalty", type=float, default=0.0,
                   help="Soft penalty: score = robust_metric - turnover_penalty * avg_turnover.")

    args = p.parse_args()

    set_global_seed(args.seed)

    # Load dataset (same as previous scripts)
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

    input_dim = X_train.shape[1]

    # DataLoaders with deterministic shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP(input_dim=input_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    TOPK_GRID = parse_int_list(args.grid_K)
    REBALANCE_GRID = parse_int_list(args.grid_reb)
    BUFFER_GRID = parse_int_list(args.grid_buf)

    split_date = pd.Timestamp(args.val_split_date)

    best_global = {
        "score": -1e18,
        "robust_metric": -1e18,
        "avg_turnover": 1e18,
        "policy": None,
    }

    for epoch in range(1, args.epochs + 1):
        train_mse = run_train_epoch(train_loader, model, optimizer, device)

        if epoch % args.eval_every_epoch != 0 and epoch != 1 and epoch != args.epochs:
            # light logging only
            if epoch % 5 == 0:
                val_mse, val_mae = eval_pred_metrics(val_loader, model, device)
                print(f"Epoch {epoch:>3} | train MSE: {train_mse:.6e} | val MAE: {val_mae:.6e} | val MSE: {val_mse:.6e}")
            continue

        # Prediction metrics
        val_mse, val_mae = eval_pred_metrics(val_loader, model, device)

        # Portfolio selection on validation
        val_pred = predict(val_loader, model, device)
        df_val = build_df(val_pred, y_val, val_dates, val_tickers)

        best_epoch = {
            "score": -1e18,
            "robust_metric": -1e18,
            "avg_turnover": 1e18,
            "policy": None,
        }

        for K in TOPK_GRID:
            for reb in REBALANCE_GRID:
                for buf in BUFFER_GRID:
                    pol = Policy(K=K, rebalance_every=reb, buffer=buf)
                    net, mkt, avg_to = simulate_policy(
                        df_val, pol,
                        cost_bps=args.cost_bps,
                        charge_entry_cost=args.charge_entry_cost,
                        charge_exit_cost=args.charge_exit_cost,
                    )

                    rm = robust_metric_over_halves(net, mkt, split_date, args.select_metric)

                    # hard constraint
                    if args.max_avg_turnover and args.max_avg_turnover > 0:
                        if avg_to > args.max_avg_turnover:
                            continue

                    score = rm - (args.turnover_penalty * avg_to)

                    if score > best_epoch["score"]:
                        best_epoch.update({
                            "score": float(score),
                            "robust_metric": float(rm),
                            "avg_turnover": float(avg_to),
                            "policy": pol,
                        })

        # if everything was filtered out by turnover constraint, fall back to unconstrained best
        if best_epoch["policy"] is None:
            for K in TOPK_GRID:
                for reb in REBALANCE_GRID:
                    for buf in BUFFER_GRID:
                        pol = Policy(K=K, rebalance_every=reb, buffer=buf)
                        net, mkt, avg_to = simulate_policy(
                            df_val, pol,
                            cost_bps=args.cost_bps,
                            charge_entry_cost=args.charge_entry_cost,
                            charge_exit_cost=args.charge_exit_cost,
                        )
                        rm = robust_metric_over_halves(net, mkt, split_date, args.select_metric)
                        score = rm - (args.turnover_penalty * avg_to)
                        if score > best_epoch["score"]:
                            best_epoch.update({
                                "score": float(score),
                                "robust_metric": float(rm),
                                "avg_turnover": float(avg_to),
                                "policy": pol,
                            })

        pol = best_epoch["policy"]
        assert pol is not None

        if best_epoch["score"] > best_global["score"]:
            best_global = dict(best_epoch)

        print(
            f"Epoch {epoch:>3} | train MSE: {train_mse:.6e} | val MAE: {val_mae:.6e} | val MSE: {val_mse:.6e} | "
            f"best VAL robust {args.select_metric}: {best_epoch['robust_metric']:.4f} "
            f"score={best_epoch['score']:.4f} TO={best_epoch['avg_turnover']:.4f} "
            f"params(K,reb,buf)=({pol.K}, {pol.rebalance_every}, {pol.buffer}) | "
            f"best GLOBAL score: {best_global['score']:.4f}"
        )

    # -----------------------------
    # Final evaluation on TEST using best_global policy
    # -----------------------------
    best_pol: Policy = best_global["policy"]
    assert best_pol is not None

    # Prediction metrics
    train_mse, train_mse = eval_pred_metrics(train_loader, model, device)
    val_mse, val_mae = eval_pred_metrics(val_loader, model, device)
    test_mse, test_mae = eval_pred_metrics(test_loader, model, device)

    print("\n=== SELECTION SUMMARY ===")
    print(f"Selection metric: {args.select_metric} (robust=min over H1/H2)")
    print(f"Best GLOBAL score: {best_global['score']:.6f}")
    print(f"Best GLOBAL robust metric: {best_global['robust_metric']:.6f}")
    print(f"Best params (K, rebalance_every, buffer): ({best_pol.K}, {best_pol.rebalance_every}, {best_pol.buffer})")
    print(f"Cost config: COST_BPS={args.cost_bps}, entry={args.charge_entry_cost}, exit={args.charge_exit_cost}")
    print(f"Turnover control: max_avg_turnover={args.max_avg_turnover}, penalty={args.turnover_penalty}")

    print("\nFinal prediction metrics:")
    print(f"Train: MSE={train_mse:.6e}, MAE={train_mse:.6e}")
    print(f"Val:   MSE={val_mse:.6e}, MAE={val_mae:.6e}")
    print(f"Test:  MSE={test_mse:.6e}, MAE={test_mae:.6e}")

    # Test portfolio results
    test_pred = predict(test_loader, model, device)
    df_test = build_df(test_pred, y_test, test_dates, test_tickers)

    net, mkt, avg_to = simulate_policy(
        df_test, best_pol,
        cost_bps=args.cost_bps,
        charge_entry_cost=args.charge_entry_cost,
        charge_exit_cost=args.charge_exit_cost,
    )

    print(f"\n=== TEST RESULTS (Selected params K,reb,buf=({best_pol.K}, {best_pol.rebalance_every}, {best_pol.buffer})) ===")
    print(f"Market   Sharpe: {sharpe(mkt.values):.6f} Sortino: {sortino(mkt.values):.6f} Cum: {(1+mkt).prod()-1:.6f}")
    print(f"NET      Sharpe: {sharpe(net.values):.6f} Sortino: {sortino(net.values):.6f} Cum: {(1+net).prod()-1:.6f}")
    print(f"Alpha(N) IR:     {alpha_ir(net.values, mkt.values):.6f}")
    print(f"Avg turnover: {avg_to:.6f}")


if __name__ == "__main__":
    main()
