# stock_mixer_sharpe.py
import os
import math
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from samplers import MultiDateBatchSampler


# --------------------------
# Utils
# --------------------------
def _try_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # unique_tickers may be object dtype
    if os.path.basename(path) == "unique_tickers.npy":
        return np.load(path, allow_pickle=True)
    return np.load(path, allow_pickle=False)


def linear_ramp(epoch: int, start: int, end: int) -> float:
    if end <= start:
        return 1.0 if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    if epoch > end:
        return 1.0
    return (epoch - start) / float(end - start)


def parse_int_list(s: str):
    # accepts "5,10,20" or "5 10 20"
    if s is None:
        return None
    s = s.replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


# --------------------------
# Dataset
# --------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, date_id: np.ndarray, ticker_id: np.ndarray):
        self.X = torch.from_numpy(X).float().unsqueeze(-1)          # (N, T, 1)
        self.y = torch.from_numpy(y).float()                        # (N,)
        self.d = torch.from_numpy(date_id).long()                   # (N,)
        self.t = torch.from_numpy(ticker_id).long()                 # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.d[idx], self.t[idx]


# --------------------------
# StockMixer
# --------------------------
class StockMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        exp_time: float = 2.0,
        exp_feat: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_time = int(seq_len * exp_time)
        hidden_feat = int(d_model * exp_feat)

        self.norm_time = nn.LayerNorm(d_model)
        self.norm_feat = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_time),
            nn.GELU(),
            nn.Linear(hidden_time, seq_len),
            nn.Dropout(dropout),
        )

        self.feat_mlp = nn.Sequential(
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
        z = self.feat_mlp(z)
        z = z.reshape(B, T, D)
        x = x + z
        return x


class StockMixer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        exp_time: float = 2.0,
        exp_feat: float = 2.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [
                StockMixerBlock(
                    seq_len=seq_len,
                    d_model=d_model,
                    exp_time=exp_time,
                    exp_feat=exp_feat,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        last = x[:, -1, :]
        return self.head(last).squeeze(-1)   # (B,)


# --------------------------
# Soft portfolio + objectives (Sharpe / Sortino)
# --------------------------
def softmax_cs(scores: torch.Tensor, date_ids: torch.Tensor, n_tickers: int, eps: float = 1e-8):
    """
    Create cross-sectional softmax weights per day.
    scores: (N,) aligned with (date_ids, ticker_ids) but we do not need ticker ids here.
    Returns weights w: (N,) such that sum_{i in day d} w_i = 1.
    """
    device = scores.device
    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)

    w = torch.zeros_like(scores)
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        s = scores[mask]
        # numerically stable softmax
        s = s - s.max()
        ex = torch.exp(s)
        w[mask] = ex / (ex.sum() + eps)
    return w


def portfolio_gross_returns(scores: torch.Tensor, rets: torch.Tensor, date_ids: torch.Tensor, n_tickers: int):
    """
    Build daily portfolio return series from soft weights:
      r_p(d) = sum_i w_i(d) * r_i(d)
    """
    w = softmax_cs(scores, date_ids, n_tickers)
    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    rp = []
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        rp.append((w[mask] * rets[mask]).sum())
    return torch.stack(rp), w


def turnover_from_weights(w: torch.Tensor, date_ids: torch.Tensor, n_tickers: int):
    """
    Turnover proxy between consecutive soft weight vectors:
      turnover_t = 0.5 * sum_i |w_t(i) - w_{t-1}(i)|
    We need weights aggregated on a fixed universe per day.
    Here, we approximate using per-day concatenated weights in the same ordering of assets within day.
    Since universe can vary slightly, we instead compute turnover on the *intersection* by padding to max assets
    per day via ticker_ids in the discrete backtest (for main reported metrics).
    For training objective, we use a simple day-to-day L1 diff on weights grouped by day with per-day ordering.
    """
    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)

    turns = []
    prev = None
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        wd = w[mask]
        if prev is None:
            turns.append(torch.tensor(1.0, device=w.device))
        else:
            m = min(prev.numel(), wd.numel())
            turns.append(0.5 * torch.abs(wd[:m] - prev[:m]).sum())
        prev = wd
    return torch.stack(turns)


def net_returns_from_gross(rp_gross: torch.Tensor, turnover: torch.Tensor, cost_bps: float):
    return rp_gross - turnover * (cost_bps / 10000.0)


def sharpe_loss(rp: torch.Tensor, eps: float = 1e-8):
    mu = rp.mean()
    sd = rp.std(unbiased=False).clamp_min(eps)
    return -(mu / sd) * math.sqrt(252.0)


def sortino_loss(rp: torch.Tensor, eps: float = 1e-8):
    # RMS downside (more stable)
    mu = rp.mean()
    downside = torch.clamp(rp, max=0.0)
    dd = torch.sqrt((downside ** 2).mean() + eps)
    return -(mu / dd) * math.sqrt(252.0)


def alpha_ir_net(rp_net: torch.Tensor, rm: torch.Tensor, eps: float = 1e-8):
    alpha = rp_net - rm
    mu = alpha.mean()
    sd = alpha.std(unbiased=False).clamp_min(eps)
    return (mu / sd) * math.sqrt(252.0)


# --------------------------
# RankIC (Spearman) for ranking-quality diagnostics
# --------------------------
@torch.no_grad()
def rankic_spearman_stats(scores: torch.Tensor, rets: torch.Tensor, date_ids: torch.Tensor):
    """
    Spearman Rank IC computed per day (cross-sectional):
      IC_d = SpearmanCorr(rank(scores_d), rank(rets_d))
    Returns (mean, std, pos_fraction) across days.
    Note: ranks are computed with argsort (ties broken arbitrarily).
    """
    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)

    ics = []
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        s = scores[mask]
        y = rets[mask]
        n = s.numel()
        if n < 3:
            continue

        rs = torch.argsort(torch.argsort(s)).float()
        ry = torch.argsort(torch.argsort(y)).float()

        rs = (rs - rs.mean()) / (rs.std(unbiased=False).clamp_min(1e-8))
        ry = (ry - ry.mean()) / (ry.std(unbiased=False).clamp_min(1e-8))

        ics.append((rs * ry).mean())

    if len(ics) == 0:
        return float("nan"), float("nan"), float("nan")

    ics = torch.stack(ics)
    mean = float(ics.mean().item())
    std = float(ics.std(unbiased=False).item())
    pos = float((ics > 0).float().mean().item())
    return mean, std, pos


# --------------------------
# Discrete backtest + robust selection on VAL (grid search K, reb, buf)
# --------------------------
@torch.no_grad()
def eval_grid_robust_net_alpha_ir(
    scores: torch.Tensor,
    rets: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    n_tickers: int,
    cost_bps: float,
    charge_entry_cost: bool,
    grid_K,
    grid_reb,
    grid_buf,
):
    device = scores.device
    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    T = len(uniq_dates)
    if T < 10:
        raise RuntimeError("Not enough validation days for robust split.")

    # Build per-day arrays of (ticker_id, score, ret)
    day_data = []
    mkt = []
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        t = ticker_ids[mask]
        s = scores[mask]
        y = rets[mask]
        day_data.append((t, s, y))
        mkt.append(y.mean())
    rm = torch.stack(mkt)

    # robust split: first half vs second half
    split = T // 2
    idx_first = torch.arange(0, split, device=device)
    idx_second = torch.arange(split, T, device=device)

    def run_discrete_backtest(K: int, reb: int, buf: int):
        holdings = torch.zeros((n_tickers,), dtype=torch.bool, device=device)
        prev_holdings = None

        rp_gross = []
        rp_net = []
        turnover = []

        for i, (t, s, y) in enumerate(day_data):
            # rebalance schedule
            do_reb = (i == 0) or (i % reb == 0)

            if do_reb:
                # ranks: high score = better
                order = torch.argsort(s, descending=True)
                top = t[order]

                if i == 0:
                    new_hold = torch.zeros_like(holdings)
                    new_hold[top[:K]] = True
                else:
                    # keep positions that remain within K+buf
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(order.numel(), device=device)
                    # map ticker -> rank (only for tickers present today)
                    rank_full = torch.full((n_tickers,), fill_value=10**9, device=device, dtype=torch.long)
                    rank_full[t] = ranks

                    keep = holdings & (rank_full <= (K + buf))
                    new_hold = keep.clone()

                    # fill up to K from top list
                    for tt in top.tolist():
                        if new_hold.sum().item() >= K:
                            break
                        if not new_hold[tt]:
                            new_hold[tt] = True

                # turnover
                if prev_holdings is None:
                    to = 1.0 if charge_entry_cost else 0.0
                else:
                    overlap = (prev_holdings & new_hold).sum().item()
                    to = 1.0 - overlap / float(K)
                holdings = new_hold
                prev_holdings = holdings.clone()
            else:
                to = 0.0

            # portfolio return = mean of held tickers present today
            held_today = holdings[t]
            if held_today.any():
                r_g = y[held_today].mean()
            else:
                r_g = torch.tensor(0.0, device=device)

            c = to * (cost_bps / 10000.0)
            r_n = r_g - c

            rp_gross.append(r_g)
            rp_net.append(r_n)
            turnover.append(torch.tensor(float(to), device=device))

        rp_gross = torch.stack(rp_gross)
        rp_net = torch.stack(rp_net)
        turnover = torch.stack(turnover)
        return rp_net, rm, turnover

    best_val = -1e18
    best_params = None

    for reb in grid_reb:
        for buf in grid_buf:
            for K in grid_K:
                rp_net, rm_series, _ = run_discrete_backtest(K, reb, buf)
                # robust score = min(IR first half, IR second half)
                ir1 = alpha_ir_net(rp_net[idx_first], rm_series[idx_first]).item()
                ir2 = alpha_ir_net(rp_net[idx_second], rm_series[idx_second]).item()
                score = min(ir1, ir2)
                if np.isfinite(score) and score > best_val:
                    best_val = score
                    best_params = (K, reb, buf)

    return best_val, best_params


@torch.no_grad()
def eval_selected_params_on_test(
    scores: torch.Tensor,
    rets: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    n_tickers: int,
    cost_bps: float,
    charge_entry_cost: bool,
    params,
):
    device = scores.device
    K, reb, buf = params

    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    day_data = []
    mkt = []
    for d in uniq_dates.tolist():
        mask = (date_ids == int(d))
        t = ticker_ids[mask]
        s = scores[mask]
        y = rets[mask]
        day_data.append((t, s, y))
        mkt.append(y.mean())
    rm = torch.stack(mkt)

    holdings = torch.zeros((n_tickers,), dtype=torch.bool, device=device)
    prev_holdings = None

    rp_net = []
    turnover = []

    for i, (t, s, y) in enumerate(day_data):
        do_reb = (i == 0) or (i % reb == 0)

        if do_reb:
            order = torch.argsort(s, descending=True)
            top = t[order]

            if i == 0:
                new_hold = torch.zeros_like(holdings)
                new_hold[top[:K]] = True
            else:
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(order.numel(), device=device)
                rank_full = torch.full((n_tickers,), fill_value=10**9, device=device, dtype=torch.long)
                rank_full[t] = ranks

                keep = holdings & (rank_full <= (K + buf))
                new_hold = keep.clone()

                for tt in top.tolist():
                    if new_hold.sum().item() >= K:
                        break
                    if not new_hold[tt]:
                        new_hold[tt] = True

            if prev_holdings is None:
                to = 1.0 if charge_entry_cost else 0.0
            else:
                overlap = (prev_holdings & new_hold).sum().item()
                to = 1.0 - overlap / float(K)

            holdings = new_hold
            prev_holdings = holdings.clone()
        else:
            to = 0.0

        held_today = holdings[t]
        if held_today.any():
            r_g = y[held_today].mean()
        else:
            r_g = torch.tensor(0.0, device=device)

        c = to * (cost_bps / 10000.0)
        r_n = r_g - c

        rp_net.append(r_n)
        turnover.append(torch.tensor(float(to), device=device))

    rp_net = torch.stack(rp_net)
    turnover = torch.stack(turnover)

    # metrics
    def sharpe(x):
        mu = x.mean()
        sd = x.std(unbiased=False).clamp_min(1e-8)
        return (mu / sd) * math.sqrt(252.0)

    def sortino(x):
        mu = x.mean()
        down = torch.clamp(x, max=0.0)
        dd = torch.sqrt((down ** 2).mean() + 1e-8)
        return (mu / dd) * math.sqrt(252.0)

    cum = torch.cumprod(1.0 + rp_net, dim=0)[-1] - 1.0
    mkt_cum = torch.cumprod(1.0 + rm, dim=0)[-1] - 1.0
    ex_wealth = cum - mkt_cum

    alpha_ir = alpha_ir_net(rp_net, rm).item()

    return {
        "NET Sharpe": float(sharpe(rp_net).item()),
        "NET Sortino": float(sortino(rp_net).item()),
        "Alpha IR (NET)": float(alpha_ir),
        "NET Cum": float(cum.item()),
        "Excess wealth (NET)": float(ex_wealth.item()),
        "Avg turnover": float(turnover.mean().item()),
        "T (days)": int(len(rp_net)),
    }


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--save_path", type=str, default="data/stock_mixer_sharpe.pt")

    # model
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--exp_time", type=float, default=2.0)
    ap.add_argument("--exp_feat", type=float, default=2.0)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    # objective mixing
    ap.add_argument("--objective", type=str, default="sharpe", choices=["sharpe", "sortino"])
    ap.add_argument("--ramp_start", type=int, default=10)
    ap.add_argument("--ramp_end", type=int, default=30)
    ap.add_argument("--w_obj_max", type=float, default=1.0)
    ap.add_argument("--mse_lambda", type=float, default=0.0)
    ap.add_argument("--clip_obj", type=float, default=0.0)

    # discrete backtest selection metric
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--charge_entry_cost", action="store_true")

    ap.add_argument("--grid_K", type=str, default="5,10,20,30,40,50,75,100")
    ap.add_argument("--grid_reb", type=str, default="1,5,10")
    ap.add_argument("--grid_buf", type=str, default="0,10,20,40")

    ap.add_argument("--days_per_batch", type=int, default=20)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load data
    X_train = _try_load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = _try_load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = _try_load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = _try_load(os.path.join(args.data_dir, "y_val.npy"))
    X_test = _try_load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = _try_load(os.path.join(args.data_dir, "y_test.npy"))

    d_train = _try_load(os.path.join(args.data_dir, "date_id_train.npy")).astype(np.int64)
    d_val = _try_load(os.path.join(args.data_dir, "date_id_val.npy")).astype(np.int64)
    d_test = _try_load(os.path.join(args.data_dir, "date_id_test.npy")).astype(np.int64)

    t_train = _try_load(os.path.join(args.data_dir, "ticker_id_train.npy")).astype(np.int64)
    t_val = _try_load(os.path.join(args.data_dir, "ticker_id_val.npy")).astype(np.int64)
    t_test = _try_load(os.path.join(args.data_dir, "ticker_id_test.npy")).astype(np.int64)

    unique_tickers = _try_load(os.path.join(args.data_dir, "unique_tickers.npy"))
    n_tickers = int(len(unique_tickers))

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("X_test: ", X_test.shape, "y_test:", y_test.shape)

    seq_len = int(X_train.shape[1])

    train_ds = SeqDataset(X_train, y_train, d_train, t_train)
    val_ds = SeqDataset(X_val, y_val, d_val, t_val)
    test_ds = SeqDataset(X_test, y_test, d_test, t_test)

    train_sampler = MultiDateBatchSampler(
        date_ids=d_train,
        days_per_batch=args.days_per_batch,
        shuffle_dates=True,
        drop_last=True
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    # model
    model = StockMixer(
        seq_len=seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        exp_time=args.exp_time,
        exp_feat=args.exp_feat,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # grids
    grid_K = parse_int_list(args.grid_K)
    grid_reb = parse_int_list(args.grid_reb)
    grid_buf = parse_int_list(args.grid_buf)

    obj_name = "Sharpe" if args.objective == "sharpe" else "Sortino"

    best_global = -1e18
    best_global_params = None
    best_state = None

    def collect_scores(loader):
        all_scores, all_y, all_d, all_t = [], [], [], []
        model.eval()
        for xb, yb, db, tb in loader:
            xb = xb.to(device)
            with torch.no_grad():
                s = model(xb)  # (B,)
            all_scores.append(s.detach().cpu())
            all_y.append(yb.detach().cpu())
            all_d.append(db.detach().cpu())
            all_t.append(tb.detach().cpu())
        return (
            torch.cat(all_scores).to(device),
            torch.cat(all_y).to(device),
            torch.cat(all_d).to(device),
            torch.cat(all_t).to(device),
        )

    for epoch in range(1, args.epochs + 1):
        model.train()

        w_obj = linear_ramp(epoch, args.ramp_start, args.ramp_end)
        w_obj = min(w_obj, float(args.w_obj_max))

        obj_losses = []
        mse_regs = []

        for xb, yb, db, tb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)

            opt.zero_grad()
            scores = model(xb)

            # objective over this batch: soft portfolio returns per day in the batch
            rp_gross, w = portfolio_gross_returns(scores, yb, db, n_tickers)
            # turnover proxy (training-time)
            to = turnover_from_weights(w, db, n_tickers)
            rp_net = net_returns_from_gross(rp_gross, to, args.cost_bps)

            if args.objective == "sharpe":
                obj_loss = sharpe_loss(rp_net)
            else:
                obj_loss = sortino_loss(rp_net)

            if args.clip_obj and args.clip_obj > 0:
                obj_loss = torch.clamp(obj_loss, -float(args.clip_obj), float(args.clip_obj))

            mse_reg = torch.mean((scores - yb) ** 2)

            loss = w_obj * obj_loss + float(args.mse_lambda) * mse_reg

            loss.backward()
            opt.step()

            obj_losses.append(obj_loss.item())
            mse_regs.append(mse_reg.item())

        # validation: prediction metrics + grid-search selection metric
        val_scores, val_y, val_d, val_t = collect_scores(val_loader)
        val_mse = torch.mean((val_scores - val_y) ** 2).item()
        val_mae = torch.mean(torch.abs(val_scores - val_y)).item()


        # ranking diagnostic (score vs future return)
        if epoch == 1 or epoch % 5 == 0:
            ic_mean, ic_std, ic_pos = rankic_spearman_stats(val_scores, val_y, val_d)
            print(f"         Val RankIC (Spearman): mean={ic_mean:.4f} std={ic_std:.4f} pos%={ic_pos*100:.1f}%")
        best_epoch_val, best_epoch_params = eval_grid_robust_net_alpha_ir(
            scores=val_scores,
            rets=val_y,
            date_ids=val_d,
            ticker_ids=val_t,
            n_tickers=n_tickers,
            cost_bps=args.cost_bps,
            charge_entry_cost=args.charge_entry_cost,
            grid_K=grid_K,
            grid_reb=grid_reb,
            grid_buf=grid_buf,
        )

        if best_epoch_val > best_global:
            best_global = best_epoch_val
            best_global_params = best_epoch_params
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:>3d} | w_obj={w_obj:0.2f} | train {obj_name}-loss: {np.mean(obj_losses): .6f} "
                f"| mse_reg: {np.mean(mse_regs): .6f} | val MSE: {val_mse: .6e} | val MAE: {val_mae: .6e} "
                f"| best VAL robust NET Alpha IR (epoch): {best_epoch_val: .4f} params(K,reb,buf)={best_epoch_params} "
                f"| best GLOBAL: {best_global: .4f} params={best_global_params}"
            )

    print("\n=== SELECTION SUMMARY ===")
    print("Best GLOBAL robust Val NET Alpha IR:", best_global)
    print("Best params (K, rebalance_every, buffer):", best_global_params)
    print(f"Cost config: COST_BPS={args.cost_bps}, CHARGE_ENTRY_COST={args.charge_entry_cost}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metrics
    test_scores, test_y, test_d, test_t = collect_scores(test_loader)

    # Sanity check: score scale + RankIC (scores are ranking scores under Sharpe-loss)
    print("\nSanity check (TEST):")
    print(
        "y_test   min/max/mean/std:",
        float(test_y.min()), float(test_y.max()), float(test_y.mean()), float(test_y.std()),
    )
    print(
        "pred     min/max/mean/std:",
        float(test_scores.min()), float(test_scores.max()), float(test_scores.mean()), float(test_scores.std()),
    )
    ic_mean_t, ic_std_t, ic_pos_t = rankic_spearman_stats(test_scores, test_y, test_d)
    print(f"TEST RankIC (Spearman): mean={ic_mean_t:.4f} std={ic_std_t:.4f} pos%={ic_pos_t*100:.1f}%")

    test_mse = torch.mean((test_scores - test_y) ** 2).item()
    test_mae = torch.mean(torch.abs(test_scores - test_y)).item()

    print("\nFinal prediction metrics:")
    print(f"Test: MSE={test_mse:.6e}, MAE={test_mae:.6e}")

    if best_global_params is not None:
        res = eval_selected_params_on_test(
            scores=test_scores,
            rets=test_y,
            date_ids=test_d,
            ticker_ids=test_t,
            n_tickers=n_tickers,
            cost_bps=args.cost_bps,
            charge_entry_cost=args.charge_entry_cost,
            params=best_global_params,
        )
        print("\n=== TEST RESULTS (Selected params) ===")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    # save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "params": {
                "objective": args.objective,
                "best_params": best_global_params,
                "best_val_robust_net_alpha_ir": best_global,
                "stockmixer": {
                    "seq_len": seq_len,
                    "d_model": args.d_model,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "exp_time": args.exp_time,
                    "exp_feat": args.exp_feat,
                },
            },
        },
        args.save_path,
    )
    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()