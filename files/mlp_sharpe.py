

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities: loading with fallback
# ----------------------------

def _try_load(path: str, allow_pickle: bool = False) -> Optional[np.ndarray]:
    if os.path.exists(path):
        return np.load(path, allow_pickle=allow_pickle)
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_split_arrays(data_dir: str, split: str, prefer_mlp: bool = True):
    """
    Loads arrays for split in a robust way.
    Expected in data/: X_{split}.npy or X_{split}_mlp.npy, y_{split}.npy,
    ticker_id_{split}.npy, {split}_date_id.npy or date_id_{split}.npy.
    """
    # X
    x = None
    if prefer_mlp:
        x = _try_load(os.path.join(data_dir, f"X_{split}_mlp.npy"))
    if x is None:
        x = _try_load(os.path.join(data_dir, f"X_{split}.npy"))
    if x is None:
        raise FileNotFoundError(f"Missing X for split={split} (X_{split}.npy or X_{split}_mlp.npy) in {data_dir}")

    # y
    y = _try_load(os.path.join(data_dir, f"y_{split}.npy"))
    if y is None:
        raise FileNotFoundError(f"Missing y_{split}.npy in {data_dir}")

    # ticker ids (global 0..U-1)
    ticker_id = _try_load(os.path.join(data_dir, f"ticker_id_{split}.npy"))
    if ticker_id is None:
        # fallback: {split}_ticker_id.npy
        ticker_id = _try_load(os.path.join(data_dir, f"{split}_ticker_id.npy"))
    if ticker_id is None:
        raise FileNotFoundError(f"Missing ticker_id_{split}.npy (or {split}_ticker_id.npy) in {data_dir}")

    # date ids (integer per sample)
    date_id = _try_load(os.path.join(data_dir, f"{split}_date_id.npy"))
    if date_id is None:
        date_id = _try_load(os.path.join(data_dir, f"date_id_{split}.npy"))
    if date_id is None:
        raise FileNotFoundError(f"Missing {split}_date_id.npy (or date_id_{split}.npy) in {data_dir}")

    # optional: date index -> actual date
    dates = _try_load(os.path.join(data_dir, f"{split}_dates.npy"), allow_pickle=True)
    # optional: tickers strings
    tickers = _try_load(os.path.join(data_dir, f"{split}_tickers.npy"), allow_pickle=True)

    return x.astype(np.float32), y.astype(np.float32), ticker_id.astype(np.int64), date_id.astype(np.int64), dates, tickers


# ----------------------------
# Model (simple MLP baseline)
# ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, D) -> (N,)
        return self.net(x).squeeze(-1)


# ----------------------------
# Assortment / portfolio helpers
# ----------------------------

@dataclass
class StrategyParams:
    K: int
    rebalance_every: int
    buffer: int  # hysteresis "rank buffer"


def annualize_factor() -> float:
    return math.sqrt(252.0)

def sharpe_ratio(r: np.ndarray) -> float:
    r = r.astype(np.float64)
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * annualize_factor()

def sortino_ratio(r: np.ndarray) -> float:
    r = r.astype(np.float64)
    mu = r.mean()
    downside = r[r < 0.0]
    if downside.size == 0:
        return (mu / 1e-12) * annualize_factor()
    sd = downside.std(ddof=0)
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * annualize_factor()

def cum_return(r: np.ndarray) -> float:
    # cumulative compounded return
    r = r.astype(np.float64)
    return float(np.prod(1.0 + r) - 1.0)

def alpha_ir(alpha: np.ndarray) -> float:
    # Information ratio on alpha series
    alpha = alpha.astype(np.float64)
    mu = alpha.mean()
    sd = alpha.std(ddof=0)
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * annualize_factor()


# ----------------------------
# Differentiable Sharpe loss on daily softmax-portfolio
# ----------------------------

@torch.no_grad()
def _build_day_index(date_id: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    unique_days = np.unique(date_id)
    groups = {}
    for d in unique_days:
        groups[int(d)] = np.where(date_id == d)[0]
    return unique_days, groups

def differentiable_sharpe_loss(
    scores: torch.Tensor,       # (n_day_samples,)
    y_true: torch.Tensor,       # (n_day_samples,)
    ticker_id: torch.Tensor,    # (n_day_samples,) global ids
    U: int,
    prev_w_full: Optional[torch.Tensor],
    cost_bps: float,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For one day: build softmax weights over tickers present that day,
    embed into full universe U (missing tickers weight 0),
    compute rp_net and turnover.
    Returns: rp_net (scalar), turnover (scalar), w_full (U,)
    """
    # weights over present tickers
    # (n,) -> (n,)
    w_present = F.softmax(scores / max(temperature, 1e-6), dim=0)

    # full universe weights
    w_full = torch.zeros((U,), device=scores.device, dtype=scores.dtype)
    w_full.scatter_(0, ticker_id, w_present)

    # gross portfolio return for the day
    rp_gross = torch.sum(w_present * y_true)

    # turnover vs previous day full weights
    if prev_w_full is None:
        turnover = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    else:
        turnover = 0.5 * torch.sum(torch.abs(w_full - prev_w_full))

    rp_net = rp_gross - turnover * (cost_bps / 10000.0)
    return rp_net, turnover, w_full


def train_one_epoch_sharpe(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    ticker_id: torch.Tensor,
    date_id: torch.Tensor,
    unique_days_sorted: np.ndarray,
    day_groups: Dict[int, np.ndarray],
    U: int,
    cost_bps: float,
    temperature: float,
    mse_reg_lambda: float,
    device: torch.device,
    max_days_per_epoch: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train by Sharpe-loss computed on sequence of daily net returns (soft portfolio).
    We iterate days (shuffled) and build rp_net per day, then Sharpe over those days.
    """
    model.train()

    # shuffle days
    days = unique_days_sorted.copy()
    np.random.shuffle(days)
    if max_days_per_epoch is not None:
        days = days[:max_days_per_epoch]

    # compute daily returns sequence in a differentiable way
    rp_list = []
    prev_w_full = None

    # Track MSE regularization to limit score scale during training.
    mse_reg_accum = torch.tensor(0.0, device=device)

    for d in days:
        idx = day_groups[int(d)]
        idx_t = torch.from_numpy(idx).to(device)

        x_d = X.index_select(0, idx_t)
        y_d = y.index_select(0, idx_t)
        tid_d = ticker_id.index_select(0, idx_t)

        scores_d = model(x_d)

        # mse reg on raw prediction vs target
        mse_reg_accum = mse_reg_accum + F.mse_loss(scores_d, y_d)

        rp_net, turnover, w_full = differentiable_sharpe_loss(
            scores=scores_d,
            y_true=y_d,
            ticker_id=tid_d,
            U=U,
            prev_w_full=prev_w_full,
            cost_bps=cost_bps,
            temperature=temperature,
        )
        rp_list.append(rp_net)
        prev_w_full = w_full

    rp = torch.stack(rp_list)  # (T,)
    mu = rp.mean()
    sd = rp.std(unbiased=False) + 1e-8
    sharpe = (mu / sd) * annualize_factor()

    # Sharpe-loss = -Sharpe
    loss = -sharpe

    mse_reg = mse_reg_accum / max(len(rp_list), 1)
    loss = loss + mse_reg_lambda * mse_reg

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt.step()

    return float(loss.detach().cpu().item()), float(mse_reg.detach().cpu().item())


# ----------------------------
# Non-differentiable backtest for validation/test (Top-K + buffer + rebalance)
# ----------------------------

def select_with_buffer(
    scores: np.ndarray,                 # (n_tickers_present,)
    ticker_ids: np.ndarray,             # (n_tickers_present,)
    prev_hold: Optional[np.ndarray],    # tickers held previous rebalance, shape (K,) or None
    K: int,
    buffer: int,
) -> np.ndarray:
    """
    Buffer hysteresis:
    - compute rank of tickers by score desc (higher=better)
    - keep prev holdings if they are within (K + buffer)
    - fill remaining slots from top-ranked
    """
    order = np.argsort(-scores)  # desc
    ranked_ids = ticker_ids[order]

    if prev_hold is None:
        return ranked_ids[:K].copy()

    prev_set = set(prev_hold.tolist())
    # keep those prev that are still "good enough"
    keep = []
    # map ticker -> rank position
    rank_pos = {int(t): i for i, t in enumerate(ranked_ids.tolist())}

    for t in prev_hold:
        t = int(t)
        pos = rank_pos.get(t, None)
        if pos is not None and pos < (K + buffer):
            keep.append(t)

    keep = list(dict.fromkeys(keep))  # unique preserve order
    if len(keep) >= K:
        return np.array(keep[:K], dtype=np.int64)

    # fill with best-ranked not already kept
    out = keep[:]
    for t in ranked_ids:
        t = int(t)
        if t not in prev_set and t not in out:
            out.append(t)
        if len(out) >= K:
            break
    return np.array(out, dtype=np.int64)

def backtest_topk_net(
    y_true: np.ndarray,          # (N,)
    scores: np.ndarray,          # (N,)
    ticker_id: np.ndarray,       # (N,)
    date_id: np.ndarray,         # (N,)
    U: int,
    params: StrategyParams,
    cost_bps: float,
    charge_entry_cost: bool = True,
) -> Dict[str, float]:
    """
    Long-only equal-weight Top-K, rebalanced every 'rebalance_every' days.
    Buffer reduces churn. Costs via turnover.
    Market = equal-weight across available tickers each day.
    """
    # sort by date, then within date doesn't matter
    order = np.lexsort((ticker_id, date_id))
    y = y_true[order]
    s = scores[order]
    tid = ticker_id[order]
    did = date_id[order]

    unique_days = np.unique(did)
    unique_days.sort()

    # day -> slice indices
    day_slices = {}
    start = 0
    for d in unique_days:
        mask = (did == d)
        idx = np.where(mask)[0]
        day_slices[int(d)] = idx

    K = params.K
    reb = params.rebalance_every
    buf = params.buffer

    # holdings at last rebalance (global tickers)
    hold = None
    prev_w_full = np.zeros((U,), dtype=np.float64)

    rp_gross = []
    rp_net = []
    rm = []
    turnover_list = []

    # to define rebalance schedule: rebalance on day index 0, reb, 2*reb, ...
    for t_i, d in enumerate(unique_days):
        idx = day_slices[int(d)]
        y_d = y[idx]
        s_d = s[idx]
        tid_d = tid[idx]

        # market return (equal-weight across available tickers)
        rm_t = float(np.mean(y_d)) if y_d.size > 0 else 0.0
        rm.append(rm_t)

        # rebalance decision
        do_reb = (t_i % reb == 0)

        if do_reb:
            hold_new = select_with_buffer(
                scores=s_d,
                ticker_ids=tid_d,
                prev_hold=hold,
                K=K,
                buffer=buf
            )
            hold = hold_new

            # build new target weights (equal weight in holdings)
            w_full = np.zeros((U,), dtype=np.float64)
            if hold.size > 0:
                w_full[hold] = 1.0 / float(len(hold))

            # turnover
            to = 0.5 * float(np.sum(np.abs(w_full - prev_w_full)))
            # optionally charge entry cost only on rebalance
            if (not charge_entry_cost) and to > 0:
                to = 0.0

            prev_w_full = w_full
        else:
            # no rebalance => weights unchanged
            w_full = prev_w_full
            to = 0.0

        turnover_list.append(to)

        # portfolio gross return uses current day's realized returns for held tickers (equal-weight)
        if hold is None or hold.size == 0:
            rp_t_gross = 0.0
        else:
            # map returns of available tickers, missing tickers treated as 0
            # (in practice holdings should be subset of available most days)
            ret_map = {}
            for tt, rr in zip(tid_d.tolist(), y_d.tolist()):
                ret_map[int(tt)] = float(rr)
            rp_t_gross = float(np.mean([ret_map.get(int(tt), 0.0) for tt in hold.tolist()]))

        rp_t_net = rp_t_gross - to * (cost_bps / 10000.0)

        rp_gross.append(rp_t_gross)
        rp_net.append(rp_t_net)

    rp_gross = np.asarray(rp_gross, dtype=np.float64)
    rp_net = np.asarray(rp_net, dtype=np.float64)
    rm = np.asarray(rm, dtype=np.float64)
    turnover_list = np.asarray(turnover_list, dtype=np.float64)

    alpha_g = rp_gross - rm
    alpha_n = rp_net - rm

    out = {
        "T": float(rp_net.size),
        "Market_Sharpe": sharpe_ratio(rm),
        "Market_Sortino": sortino_ratio(rm),
        "Market_Cum": cum_return(rm),
        "GROSS_Sharpe": sharpe_ratio(rp_gross),
        "GROSS_Sortino": sortino_ratio(rp_gross),
        "GROSS_Cum": cum_return(rp_gross),
        "NET_Sharpe": sharpe_ratio(rp_net),
        "NET_Sortino": sortino_ratio(rp_net),
        "NET_Cum": cum_return(rp_net),
        "AlphaIR_G": alpha_ir(alpha_g),
        "AlphaIR_N": alpha_ir(alpha_n),
        "AvgTurnover": float(turnover_list.mean()),
        "ExcessWealth_G": float(cum_return(rp_gross) - cum_return(rm)),
        "ExcessWealth_N": float(cum_return(rp_net) - cum_return(rm)),
    }
    return out




def pick_val_metric(res: Dict[str, float], select_metric: str) -> float:
    """
    Map a backtest result dict (from backtest_topk_net) to a scalar metric for selection.
    Keys in res: NET_Sharpe, NET_Sortino, NET_Cum, AlphaIR_N (etc).
    """
    if select_metric == "net_sortino":
        return float(res["NET_Sortino"])
    if select_metric == "net_sharpe":
        return float(res["NET_Sharpe"])
    if select_metric == "alpha_ir_net":
        return float(res["AlphaIR_N"])
    if select_metric == "net_cum":
        return float(res["NET_Cum"])
    raise ValueError(f"Unknown select_metric: {select_metric}")

def robust_val_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    ticker_id: np.ndarray,
    date_id: np.ndarray,
    U: int,
    params: StrategyParams,
    cost_bps: float,
    charge_entry_cost: bool,
    select_metric: str,
) -> float:
    """
    Split validation period into two halves by date order.
    Score = min(metric_H1, metric_H2) on NET performance.
    """
    unique_days = np.unique(date_id)
    unique_days.sort()
    if unique_days.size < 10:
        res = backtest_topk_net(y_true, scores, ticker_id, date_id, U, params, cost_bps, charge_entry_cost)
        return pick_val_metric(res, select_metric)

    mid = unique_days.size // 2
    d1 = set(unique_days[:mid].tolist())
    d2 = set(unique_days[mid:].tolist())

    m1 = np.isin(date_id, list(d1))
    m2 = np.isin(date_id, list(d2))

    res1 = backtest_topk_net(y_true[m1], scores[m1], ticker_id[m1], date_id[m1], U, params, cost_bps, charge_entry_cost)
    res2 = backtest_topk_net(y_true[m2], scores[m2], ticker_id[m2], date_id[m2], U, params, cost_bps, charge_entry_cost)

    return float(min(pick_val_metric(res1, select_metric), pick_val_metric(res2, select_metric)))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mse_reg_lambda", type=float, default=0.10, help="Regularize Sharpe-loss with MSE to keep preds stable.")
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--charge_entry_cost", action="store_true")
    ap.add_argument("--max_days_per_epoch", type=int, default=None, help="Optional: limit #days per epoch for speed.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # grid-search space
    ap.add_argument("--K_list", type=str, default="5,10,20,40,50")
    ap.add_argument("--reb_list", type=str, default="5,10")
    ap.add_argument("--buf_list", type=str, default="0,10,20,40")
    ap.add_argument("--save_path", type=str, default="data/mlp_sharpe.pt")
    ap.add_argument(
        "--select_metric",
        type=str,
        default="net_sortino",
        choices=["net_sortino", "net_sharpe", "alpha_ir_net", "net_cum"],
        help="Metric used to select (K, reb, buf) on validation (robust=min over two halves).",
    )
    ap.add_argument(
        "--fixed_policies",
        type=str,
        default="5,5,10;5,10,0;50,10,40",
        help="Semicolon-separated list of fixed (K,reb,buf) to also report on TEST.",
    )

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # universe
    unique_tickers = _try_load(os.path.join(args.data_dir, "unique_tickers.npy"), allow_pickle=True)
    if unique_tickers is None:
        raise FileNotFoundError("Missing data/unique_tickers.npy (needed for stable universe size)")
    U = int(unique_tickers.shape[0])

    # Load splits (prefer *_mlp.npy for X)
    X_train, y_train, tid_train, did_train, _, _ = load_split_arrays(args.data_dir, "train", prefer_mlp=True)
    X_val, y_val, tid_val, did_val, _, _ = load_split_arrays(args.data_dir, "val", prefer_mlp=True)
    X_test, y_test, tid_test, did_test, _, _ = load_split_arrays(args.data_dir, "test", prefer_mlp=True)

    print("Shapes:")
    print(f"X_train: {X_train.shape} y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape} y_test: {y_test.shape}")

    # Torch tensors
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    ttr = torch.from_numpy(tid_train).to(device)
    dtr = torch.from_numpy(did_train).to(device)

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    tv = torch.from_numpy(tid_val).to(device)
    dv = torch.from_numpy(did_val).to(device)

    Xt = torch.from_numpy(X_test).to(device)
    yt = torch.from_numpy(y_test).to(device)
    tt = torch.from_numpy(tid_test).to(device)
    dt = torch.from_numpy(did_test).to(device)

    # For training: group by day on train
    train_days, train_groups = _build_day_index(did_train)

    # Model
    model = MLP(in_dim=X_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Grid search space
    def parse_int_list(s: str) -> List[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    K_list = parse_int_list(args.K_list)
    reb_list = parse_int_list(args.reb_list)
    buf_list = parse_int_list(args.buf_list)

    best_global_score = -1e18
    best_global_params: Optional[StrategyParams] = None

    # helper: predict scores for a split
    @torch.no_grad()
    def predict_scores(X: torch.Tensor) -> np.ndarray:
        model.eval()
        out = model(X).detach().cpu().numpy().astype(np.float64)
        return out

    # training loop
    for epoch in range(1, args.epochs + 1):
        loss_sh, mse_reg = train_one_epoch_sharpe(
            model=model,
            opt=opt,
            X=Xtr,
            y=ytr,
            ticker_id=ttr,
            date_id=dtr,
            unique_days_sorted=train_days,
            day_groups=train_groups,
            U=U,
            cost_bps=args.cost_bps,
            temperature=args.temperature,
            mse_reg_lambda=args.mse_reg_lambda,
            device=device,
            max_days_per_epoch=args.max_days_per_epoch,
        )

        # validation prediction metrics (report only)
        with torch.no_grad():
            model.eval()
            pred_val = model(Xv)
            val_mse = F.mse_loss(pred_val, yv).item()
            val_mae = F.l1_loss(pred_val, yv).item()

        # robust selection on validation (NET alpha IR)
        scores_val = predict_scores(Xv)
        y_val_np = y_val.astype(np.float64)
        tid_val_np = tid_val.astype(np.int64)
        did_val_np = did_val.astype(np.int64)

        best_epoch_score = -1e18
        best_epoch_params = None

        for K in K_list:
            for reb in reb_list:
                for buf in buf_list:
                    params = StrategyParams(K=K, rebalance_every=reb, buffer=buf)
                    score = robust_val_score(
                        y_true=y_val_np,
                        scores=scores_val,
                        ticker_id=tid_val_np,
                        date_id=did_val_np,
                        U=U,
                        params=params,
                        cost_bps=args.cost_bps,
                        charge_entry_cost=args.charge_entry_cost,
                        select_metric=args.select_metric,
                    )
                    if score > best_epoch_score:
                        best_epoch_score = score
                        best_epoch_params = params

        # update global best
        if best_epoch_score > best_global_score:
            best_global_score = best_epoch_score
            best_global_params = best_epoch_params

        # logging
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | train Sharpe-loss: {loss_sh: .6f} | mse_reg: {mse_reg: .6f} "
                f"| val MSE: {val_mse: .6e} | val MAE: {val_mae: .6e} "
                f"| best VAL robust metric (epoch): {best_epoch_score: .4f} "
                f"params(K,reb,buf)=({best_epoch_params.K},{best_epoch_params.rebalance_every},{best_epoch_params.buffer}) "
                f"| best GLOBAL: {best_global_score: .4f} "
                f"params=({best_global_params.K},{best_global_params.rebalance_every},{best_global_params.buffer})"
            )

    print("\n=== SELECTION SUMMARY ===")
    print(f"Selection metric: {args.select_metric} (robust=min over two val halves)")
    print(f"Best GLOBAL robust Val metric: {best_global_score}")
    print(f"Best params (K, rebalance_every, buffer): ({best_global_params.K}, {best_global_params.rebalance_every}, {best_global_params.buffer})")
    print(f"Cost config: COST_BPS={args.cost_bps}, CHARGE_ENTRY_COST={args.charge_entry_cost}")

    # Final report prediction metrics (val/test)
    with torch.no_grad():
        model.eval()
        pred_val = model(Xv)
        pred_test = model(Xt)
        val_mse = F.mse_loss(pred_val, yv).item()
        val_mae = F.l1_loss(pred_val, yv).item()
        test_mse = F.mse_loss(pred_test, yt).item()
        test_mae = F.l1_loss(pred_test, yt).item()

    print("\nFinal prediction metrics:")
    print(f"Val (report only): MSE={val_mse:.6e}, MAE={val_mae:.6e}")
    print(f"Test:             MSE={test_mse:.6e}, MAE={test_mae:.6e}")

    # Test backtest with selected params
    scores_test = predict_scores(Xt)
    res_test = backtest_topk_net(
        y_true=y_test.astype(np.float64),
        scores=scores_test,
        ticker_id=tid_test.astype(np.int64),
        date_id=did_test.astype(np.int64),
        U=U,
        params=best_global_params,
        cost_bps=args.cost_bps,
        charge_entry_cost=args.charge_entry_cost,
    )

    print("\n=== TEST RESULTS (Selected params) ===")
    print(f"NET Sharpe: {res_test['NET_Sharpe']:.6f}")
    print(f"NET Sortino: {res_test['NET_Sortino']:.6f}")
    print(f"Alpha IR (NET): {res_test['AlphaIR_N']:.6f}")
    print(f"NET Cum: {res_test['NET_Cum']:.6f}")
    print(f"Excess wealth (NET): {res_test['ExcessWealth_N']:.6f}")
    print(f"Avg turnover: {res_test['AvgTurnover']:.6f}")
    print(f"T (days): {int(res_test['T'])}")

    # Additional reporting: fixed portfolio policies for fair comparisons
    def _parse_fixed_policies(s: str):
        out = []
        for part in s.split(';'):
            part = part.strip()
            if not part:
                continue
            K, reb, buf = [int(x) for x in part.split(',')]
            out.append(StrategyParams(K=K, rebalance_every=reb, buffer=buf))
        return out

    fixed_list = _parse_fixed_policies(args.fixed_policies)
    if fixed_list:
        print("\n=== TEST RESULTS (FIXED POLICIES) ===")
        for p in fixed_list:
            bt = backtest_topk_net(
                y_true=y_test.astype(np.float64),
                scores=scores_test,
                ticker_id=tid_test.astype(np.int64),
                date_id=did_test.astype(np.int64),
                U=U,
                params=p,
                cost_bps=args.cost_bps,
                charge_entry_cost=args.charge_entry_cost,
            )
            print(f"\nPolicy (K,reb,buf)=({p.K},{p.rebalance_every},{p.buffer})")
            print(f"NET Sharpe: {bt['NET_Sharpe']:.6f}")
            print(f"NET Sortino: {bt['NET_Sortino']:.6f}")
            print(f"Alpha IR (NET): {bt['AlphaIR_N']:.6f}")
            print(f"NET Cum: {bt['NET_Cum']:.6f}")
            print(f"Excess wealth (NET): {bt['ExcessWealth_N']:.6f}")
            print(f"Avg turnover: {bt['AvgTurnover']:.6f}")

    # Save model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": int(X_train.shape[1]),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "best_params": (best_global_params.K, best_global_params.rebalance_every, best_global_params.buffer),
            "best_val_robust_alpha_ir_net": float(best_global_score),
            "cost_bps": float(args.cost_bps),
            "temperature": float(args.temperature),
            "mse_reg_lambda": float(args.mse_reg_lambda),
        },
        args.save_path,
    )
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
