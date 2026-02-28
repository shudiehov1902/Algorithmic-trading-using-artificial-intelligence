#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utils (metrics)
# ----------------------------

def annualize_factor() -> float:
    return float(np.sqrt(252.0))

def cum_return(r: np.ndarray) -> float:
    # Cum = prod(1+r) - 1
    return float(np.prod(1.0 + r) - 1.0)

def sharpe_ratio(r: np.ndarray) -> float:
    mu = float(np.mean(r))
    sd = float(np.std(r)) + 1e-12
    return (mu / sd) * annualize_factor()

def sortino_ratio(r: np.ndarray) -> float:
    mu = float(np.mean(r))
    neg = np.minimum(r, 0.0)
    downside = float(np.sqrt(np.mean(neg * neg))) + 1e-12
    return (mu / downside) * annualize_factor()

def alpha_ir(alpha: np.ndarray) -> float:
    mu = float(np.mean(alpha))
    sd = float(np.std(alpha)) + 1e-12
    return (mu / sd) * annualize_factor()

def _try_load(path: str, allow_pickle: bool = False):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=allow_pickle)

def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ----------------------------
# Strategy params
# ----------------------------

@dataclass(frozen=True)
class StrategyParams:
    K: int
    rebalance_every: int
    buffer: int


# ----------------------------
# Data loading
# ----------------------------

def load_split_arrays(data_dir: str, split: str, prefer_mlp: bool = True):
    """
    Expected files in data_dir:
      X_train_mlp.npy (or X_train.npy)
      y_train.npy
      ticker_id_train.npy
      date_id_train.npy
    same for val/test
    """
    x_path = os.path.join(data_dir, f"X_{split}_mlp.npy") if prefer_mlp else os.path.join(data_dir, f"X_{split}.npy")
    if not os.path.exists(x_path):
        x_path = os.path.join(data_dir, f"X_{split}.npy")

    y_path = os.path.join(data_dir, f"y_{split}.npy")
    tid_path = os.path.join(data_dir, f"ticker_id_{split}.npy")
    did_path = os.path.join(data_dir, f"date_id_{split}.npy")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    tid = np.load(tid_path).astype(np.int64)
    did = np.load(did_path).astype(np.int64)

    return X, y, tid, did


# ----------------------------
# Model
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
        # output shape (n,) float
        return self.net(x).squeeze(-1)


# ----------------------------
# Differentiable portfolio step (soft weights over daily cross-section)
# ----------------------------

def build_universe_weights(
    scores: torch.Tensor,        # (n_day,)
    ticker_id: torch.Tensor,     # (n_day,) int64
    U: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Convert per-sample scores into universe weight vector of size U.
    We do softmax over the day cross-section, then scatter into universe positions.
    If there are duplicates ticker_id within the day (rare), weights are summed.
    Finally normalize to sum=1.
    """
    # stable softmax
    logits = scores / max(temperature, 1e-6)
    logits = logits - logits.max()
    w_day = torch.softmax(logits, dim=0)  # (n_day,)

    w_full = torch.zeros((U,), device=scores.device, dtype=torch.float32)
    w_full.scatter_add_(0, ticker_id, w_day)

    s = w_full.sum()
    if s > 0:
        w_full = w_full / s
    return w_full

def build_universe_returns(
    y_true: torch.Tensor,        # (n_day,)
    ticker_id: torch.Tensor,     # (n_day,) int64
    U: int,
) -> torch.Tensor:
    """
    Place daily realized returns into universe vector (size U).
    If duplicates appear, we sum (shouldn't happen typically).
    """
    r_full = torch.zeros((U,), device=y_true.device, dtype=torch.float32)
    r_full.scatter_add_(0, ticker_id, y_true)
    return r_full

def differentiable_portfolio_return_net(
    scores: torch.Tensor,
    y_true: torch.Tensor,
    ticker_id: torch.Tensor,
    U: int,
    prev_w_full: Optional[torch.Tensor],
    cost_bps: float,
    temperature: float = 1.0,
    charge_entry_cost: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      rp_net (scalar tensor)
      turnover (scalar tensor)
      w_full (U,)
    """
    w_full = build_universe_weights(scores, ticker_id, U, temperature)
    r_full = build_universe_returns(y_true, ticker_id, U)

    rp_gross = (w_full * r_full).sum()

    if prev_w_full is None:
        turnover = torch.tensor(0.0, device=scores.device)
    else:
        turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()

    if not charge_entry_cost:
        # if you wanted to charge only on entry you could customize,
        # but for now keep same behavior as your previous scripts
        pass

    rp_net = rp_gross - turnover * (cost_bps / 10000.0)
    return rp_net, turnover, w_full


# ----------------------------
# Training step (Sortino-loss)
# ----------------------------

def sortino_loss_from_series(rp: torch.Tensor) -> torch.Tensor:
    """
    Sortino = E[r] / downside_vol(r) * sqrt(252)
    downside_vol = sqrt(mean(min(r,0)^2))
    """
    mu = rp.mean()
    neg = torch.clamp(rp, max=0.0)
    downside = torch.sqrt((neg * neg).mean() + 1e-12)
    sortino = (mu / downside) * float(np.sqrt(252.0))
    return -sortino  # loss

def train_one_epoch_sortino(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    ticker_id: torch.Tensor,
    date_id: torch.Tensor,
    U: int,
    device: torch.device,
    cost_bps: float,
    temperature: float,
    mse_reg_lambda: float,
    charge_entry_cost: bool,
    max_days_per_epoch: Optional[int] = None,
) -> Tuple[float, float]:
    model.train()

    # group indices by day
    did_np = date_id.detach().cpu().numpy()
    unique_days = np.unique(did_np)
    unique_days.sort()

    if max_days_per_epoch is not None and len(unique_days) > max_days_per_epoch:
        unique_days = unique_days[:max_days_per_epoch]

    day_groups: Dict[int, np.ndarray] = {}
    for d in unique_days:
        day_groups[int(d)] = np.where(did_np == d)[0]

    rp_list: List[torch.Tensor] = []
    prev_w_full: Optional[torch.Tensor] = None
    mse_reg_accum = torch.tensor(0.0, device=device)

    for d in unique_days:
        idx = day_groups[int(d)]
        idx_t = torch.from_numpy(idx).to(device)

        x_d = X.index_select(0, idx_t)
        y_d = y.index_select(0, idx_t)
        tid_d = ticker_id.index_select(0, idx_t)

        scores_d = model(x_d)

        mse_reg_accum = mse_reg_accum + F.mse_loss(scores_d, y_d)

        rp_net, turnover, w_full = differentiable_portfolio_return_net(
            scores=scores_d,
            y_true=y_d,
            ticker_id=tid_d,
            U=U,
            prev_w_full=prev_w_full,
            cost_bps=cost_bps,
            temperature=temperature,
            charge_entry_cost=charge_entry_cost,
        )
        rp_list.append(rp_net)
        prev_w_full = w_full

    rp = torch.stack(rp_list)  # (T,)
    loss_sortino = sortino_loss_from_series(rp)

    mse_reg = mse_reg_accum / max(len(rp_list), 1)
    loss = loss_sortino + mse_reg_lambda * mse_reg

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt.step()

    return float(loss_sortino.detach().cpu().item()), float(mse_reg.detach().cpu().item())


# ----------------------------
# Backtest for validation/test (Top-K + buffer + rebalance)
# ----------------------------

def select_with_buffer(
    scores: np.ndarray,
    ticker_ids: np.ndarray,
    prev_hold: Optional[np.ndarray],
    K: int,
    buffer: int,
) -> np.ndarray:
    order = np.argsort(-scores)
    ranked_ids = ticker_ids[order]

    if prev_hold is None:
        return ranked_ids[:K].copy()

    prev_set = set(prev_hold.tolist())
    rank_pos = {int(t): i for i, t in enumerate(ranked_ids.tolist())}

    keep = []
    for t in prev_hold:
        t = int(t)
        pos = rank_pos.get(t, None)
        if pos is not None and pos < (K + buffer):
            keep.append(t)
    keep = list(dict.fromkeys(keep))

    if len(keep) >= K:
        return np.array(keep[:K], dtype=np.int64)

    out = keep[:]
    for t in ranked_ids:
        t = int(t)
        if t not in out:
            out.append(t)
        if len(out) >= K:
            break
    return np.array(out, dtype=np.int64)

def backtest_topk_net(
    y_true: np.ndarray,
    scores: np.ndarray,
    ticker_id: np.ndarray,
    date_id: np.ndarray,
    U: int,
    params: StrategyParams,
    cost_bps: float,
    charge_entry_cost: bool = True,
) -> Dict[str, float]:
    order = np.lexsort((ticker_id, date_id))
    y = y_true[order]
    s = scores[order]
    tid = ticker_id[order]
    did = date_id[order]

    unique_days = np.unique(did)
    unique_days.sort()

    day_slices = {}
    for d in unique_days:
        day_slices[int(d)] = np.where(did == d)[0]

    K = params.K
    reb = params.rebalance_every
    buf = params.buffer

    hold = None
    prev_w_full = np.zeros((U,), dtype=np.float64)

    rp_gross = []
    rp_net = []
    rm = []
    turnover_list = []

    for t_i, d in enumerate(unique_days):
        idx = day_slices[int(d)]
        y_d = y[idx]
        s_d = s[idx]
        tid_d = tid[idx]

        rm_t = float(np.mean(y_d)) if y_d.size > 0 else 0.0
        rm.append(rm_t)

        do_reb = (t_i % reb == 0)
        if do_reb:
            hold_new = select_with_buffer(s_d, tid_d, hold, K, buf)
            hold = hold_new

            w_full = np.zeros((U,), dtype=np.float64)
            if hold.size > 0:
                w_full[hold] = 1.0 / float(len(hold))

            to = 0.5 * float(np.sum(np.abs(w_full - prev_w_full)))
            if (not charge_entry_cost) and to > 0:
                to = 0.0
            prev_w_full = w_full
        else:
            w_full = prev_w_full
            to = 0.0

        turnover_list.append(to)

        if hold is None or hold.size == 0:
            rp_t_gross = 0.0
        else:
            ret_map = {int(tt): float(rr) for tt, rr in zip(tid_d.tolist(), y_d.tolist())}
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

    return {
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

def robust_val_score_alpha_ir(
    y_true: np.ndarray,
    scores: np.ndarray,
    ticker_id: np.ndarray,
    date_id: np.ndarray,
    U: int,
    params: StrategyParams,
    cost_bps: float,
    charge_entry_cost: bool,
) -> float:
    unique_days = np.unique(date_id)
    unique_days.sort()
    if unique_days.size < 10:
        res = backtest_topk_net(y_true, scores, ticker_id, date_id, U, params, cost_bps, charge_entry_cost)
        return float(res["AlphaIR_N"])

    mid = unique_days.size // 2
    d1 = unique_days[:mid]
    d2 = unique_days[mid:]

    m1 = np.isin(date_id, d1)
    m2 = np.isin(date_id, d2)

    res1 = backtest_topk_net(y_true[m1], scores[m1], ticker_id[m1], date_id[m1], U, params, cost_bps, charge_entry_cost)
    res2 = backtest_topk_net(y_true[m2], scores[m2], ticker_id[m2], date_id[m2], U, params, cost_bps, charge_entry_cost)

    return float(min(res1["AlphaIR_N"], res2["AlphaIR_N"]))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mse_reg_lambda", type=float, default=0.05)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--charge_entry_cost", action="store_true", default=True)
    ap.add_argument("--max_days_per_epoch", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--K_list", type=str, default="5,10,20,40,50")
    ap.add_argument("--reb_list", type=str, default="5,10")
    ap.add_argument("--buf_list", type=str, default="0,10,20,40")
    ap.add_argument("--save_path", type=str, default="data/mlp_sortino.pt")

    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Universe (robust to object dtype)
    ut_path = os.path.join(args.data_dir, "unique_tickers.npy")
    unique_tickers = _try_load(ut_path, allow_pickle=True)
    if unique_tickers is None:
        raise FileNotFoundError("Missing data/unique_tickers.npy")
    U = int(unique_tickers.shape[0])

    X_train, y_train, tid_train, did_train = load_split_arrays(args.data_dir, "train", prefer_mlp=True)
    X_val, y_val, tid_val, did_val = load_split_arrays(args.data_dir, "val", prefer_mlp=True)
    X_test, y_test, tid_test, did_test = load_split_arrays(args.data_dir, "test", prefer_mlp=True)

    print("Shapes:")
    print(f"X_train: {X_train.shape} y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape} y_test: {y_test.shape}")

    # torch tensors
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    tidtr = torch.from_numpy(tid_train).to(device)
    didtr = torch.from_numpy(did_train).to(device)

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    tidv = torch.from_numpy(tid_val).to(device)
    didv = torch.from_numpy(did_val).to(device)

    Xt = torch.from_numpy(X_test).to(device)
    yt = torch.from_numpy(y_test).to(device)
    tidt = torch.from_numpy(tid_test).to(device)
    didt = torch.from_numpy(did_test).to(device)

    in_dim = int(X_train.shape[1])
    model = MLP(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # parse grid search
    K_list = [int(x) for x in args.K_list.split(",") if x.strip()]
    reb_list = [int(x) for x in args.reb_list.split(",") if x.strip()]
    buf_list = [int(x) for x in args.buf_list.split(",") if x.strip()]

    best_global = -1e18
    best_params: Optional[StrategyParams] = None
    best_state: Optional[dict] = None

    best_val_mse = float("inf")

    for ep in range(1, args.epochs + 1):
        loss_sortino, mse_reg = train_one_epoch_sortino(
            model=model,
            opt=opt,
            X=Xtr,
            y=ytr,
            ticker_id=tidtr,
            date_id=didtr,
            U=U,
            device=device,
            cost_bps=args.cost_bps,
            temperature=args.temperature,
            mse_reg_lambda=args.mse_reg_lambda,
            charge_entry_cost=args.charge_entry_cost,
            max_days_per_epoch=args.max_days_per_epoch,
        )

        # val predictions
        model.eval()
        with torch.no_grad():
            pred_val = model(Xv).detach().cpu().numpy()
            pred_test = model(Xt).detach().cpu().numpy()

        val_mse = mse_np(pred_val, y_val)
        val_mae = mae_np(pred_val, y_val)

        if val_mse < best_val_mse:
            best_val_mse = val_mse

        # grid-search on validation by robust NET Alpha IR
        best_epoch_score = -1e18
        best_epoch_params = None

        for K in K_list:
            for reb in reb_list:
                for buf in buf_list:
                    params = StrategyParams(K=K, rebalance_every=reb, buffer=buf)
                    score = robust_val_score_alpha_ir(
                        y_true=y_val,
                        scores=pred_val,
                        ticker_id=tid_val,
                        date_id=did_val,
                        U=U,
                        params=params,
                        cost_bps=args.cost_bps,
                        charge_entry_cost=args.charge_entry_cost,
                    )
                    if score > best_epoch_score:
                        best_epoch_score = score
                        best_epoch_params = params

        # keep global best (selection criterion)
        if best_epoch_score > best_global:
            best_global = best_epoch_score
            best_params = best_epoch_params
            best_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}

        print(
            f"Epoch {ep:3d} | train Sortino-loss: {loss_sortino: .6f} | mse_reg: {mse_reg: .6f} | "
            f"val MSE: {val_mse: .6e} | val MAE: {val_mae: .6e} | "
            f"best VAL robust NET Alpha IR (epoch): {best_epoch_score: .4f} params(K,reb,buf)=({best_epoch_params.K},{best_epoch_params.rebalance_every},{best_epoch_params.buffer}) | "
            f"best GLOBAL: {best_global: .4f} params={(best_params.K, best_params.rebalance_every, best_params.buffer) if best_params else None}"
        )

    # restore best state
    if best_state is None or best_params is None:
        raise RuntimeError("No best model selected.")
    model.load_state_dict(best_state)

    # final report with selected params
    model.eval()
    with torch.no_grad():
        pred_val = model(Xv).detach().cpu().numpy()
        pred_test = model(Xt).detach().cpu().numpy()

    print("\n=== SELECTION SUMMARY ===")
    print(f"Best GLOBAL robust Val NET Alpha IR: {best_global}")
    print(f"Best params (K, rebalance_every, buffer): ({best_params.K}, {best_params.rebalance_every}, {best_params.buffer})")
    print(f"Cost config: COST_BPS={args.cost_bps}, CHARGE_ENTRY_COST={args.charge_entry_cost}")

    val_mse = mse_np(pred_val, y_val)
    val_mae = mae_np(pred_val, y_val)
    test_mse = mse_np(pred_test, y_test)
    test_mae = mae_np(pred_test, y_test)

    print("\nFinal prediction metrics:")
    print(f"Val (report only): MSE={val_mse:.6e}, MAE={val_mae:.6e}")
    print(f"Test:             MSE={test_mse:.6e}, MAE={test_mae:.6e}")

    # backtest on test with selected params
    res = backtest_topk_net(
        y_true=y_test,
        scores=pred_test,
        ticker_id=tid_test,
        date_id=did_test,
        U=U,
        params=best_params,
        cost_bps=args.cost_bps,
        charge_entry_cost=args.charge_entry_cost,
    )

    print("\n=== TEST RESULTS (Selected params) ===")
    print(f"NET Sharpe: {res['NET_Sharpe']:.6f}")
    print(f"NET Sortino: {res['NET_Sortino']:.6f}")
    print(f"Alpha IR (NET): {res['AlphaIR_N']:.6f}")
    print(f"NET Cum: {res['NET_Cum']:.6f}")
    print(f"Excess wealth (NET): {res['ExcessWealth_N']:.6f}")
    print(f"Avg turnover: {res['AvgTurnover']:.6f}")
    print(f"T (days): {int(res['T'])}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_params": (best_params.K, best_params.rebalance_every, best_params.buffer),
            "best_global_val_alpha_ir_net": best_global,
            "config": vars(args),
        },
        args.save_path,
    )
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()