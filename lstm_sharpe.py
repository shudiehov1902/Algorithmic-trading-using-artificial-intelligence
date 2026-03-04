# lstm_sharpe.py
import os
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from samplers import MultiDateBatchSampler


# --------------------------
# Utils: loading
# --------------------------
def _try_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # unique_tickers.npy can be object dtype -> allow_pickle=True only for that case
    if os.path.basename(path) == "unique_tickers.npy":
        arr = np.load(path, allow_pickle=True)
        # if object, convert to normal numpy array
        if arr.dtype == object:
            arr = np.array(list(arr))
        return arr
    return np.load(path, allow_pickle=False)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fixed_policies(s: str):
    """Parse 'K,reb,buf;K,reb,buf;...' into list of tuples."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [p.strip() for p in part.split(",")]
        if len(nums) != 3:
            raise ValueError(f"Invalid fixed policy '{part}', expected 'K,reb,buf'")
        out.append((int(nums[0]), int(nums[1]), int(nums[2])))
    return out


# --------------------------
# Dataset: sequences per ticker
# --------------------------
class SequenceDataset(Dataset):
    """
    Builds sequences inside each ticker, ordered by date_id.
    Each sample corresponds to (seq of X ending at t) -> y_t.
    Also returns (date_id_t, ticker_id).
    """

    def __init__(self, X, y, date_id, ticker_id, seq_len: int = 20, require_consecutive: bool = True):
        self.seq_len = int(seq_len)
        self.require_consecutive = bool(require_consecutive)

        # force numpy
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        date_id = np.asarray(date_id, dtype=np.int64)
        ticker_id = np.asarray(ticker_id, dtype=np.int64)

        # sort by (ticker_id, date_id)
        order = np.lexsort((date_id, ticker_id))
        self.X = X[order]
        self.y = y[order]
        self.date_id = date_id[order]
        self.ticker_id = ticker_id[order]

        # build index mapping: sample -> end_position in sorted arrays
        self.end_positions = []
        self.sample_date = []
        self.sample_ticker = []

        n = len(self.X)
        if n == 0:
            raise ValueError("Empty dataset arrays.")

        # iterate per ticker block
        start = 0
        while start < n:
            tid = self.ticker_id[start]
            end = start
            while end < n and self.ticker_id[end] == tid:
                end += 1

            # now [start, end) is one ticker
            # build sequences; optionally reset on date gaps
            if self.require_consecutive:
                # find segments of consecutive dates
                seg_start = start
                for i in range(start + 1, end):
                    if self.date_id[i] != self.date_id[i - 1] + 1:
                        # segment [seg_start, i)
                        self._add_segment(seg_start, i, tid)
                        seg_start = i
                self._add_segment(seg_start, end, tid)
            else:
                self._add_segment(start, end, tid)

            start = end

        self.end_positions = np.asarray(self.end_positions, dtype=np.int64)
        self.sample_date = np.asarray(self.sample_date, dtype=np.int64)
        self.sample_ticker = np.asarray(self.sample_ticker, dtype=np.int64)

    def _add_segment(self, seg_start, seg_end, tid):
        # need at least seq_len points
        L = seg_end - seg_start
        if L < self.seq_len:
            return
        for end_pos in range(seg_start + self.seq_len - 1, seg_end):
            self.end_positions.append(end_pos)
            self.sample_date.append(self.date_id[end_pos])
            self.sample_ticker.append(tid)

    def __len__(self):
        return len(self.end_positions)

    def __getitem__(self, idx):
        end_pos = self.end_positions[idx]
        start_pos = end_pos - self.seq_len + 1
        x_seq = self.X[start_pos:end_pos + 1]  # (seq_len, feat)
        y_t = self.y[end_pos]
        d_t = self.date_id[end_pos]
        t_id = self.ticker_id[end_pos]
        return x_seq, y_t, d_t, t_id


# --------------------------
# Model: LSTM -> score
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq):
        # x_seq: (B, T, F)
        out, _ = self.lstm(x_seq)     # (B, T, H)
        last = out[:, -1, :]          # (B, H)
        score = self.head(last).squeeze(-1)  # (B,)
        return score


# --------------------------
# Portfolio math: weights, turnover, net rp
# --------------------------
def softmax_weights(scores: torch.Tensor, temperature: float = 1.0):
    # stable softmax
    z = scores / max(temperature, 1e-6)
    z = z - z.max()
    w = torch.softmax(z, dim=0)
    return w


def compute_net_portfolio_returns_by_day(
    scores: torch.Tensor,
    rets: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    n_tickers: int,
    cost_bps: float = 10.0,
    charge_entry_cost: bool = True,
    temperature: float = 1.0,
):
    """
    Build daily weights from model scores, compute rp_gross, turnover, rp_net.
    Uses FULL vector size n_tickers to avoid size mismatch (your previous crash).
    Returns:
      rp_net: (n_days,)
      rp_gross: (n_days,)
      avg_turnover
      uniq_dates_sorted (cpu tensor)
    """
    device = scores.device
    date_ids = date_ids.to(device)
    ticker_ids = ticker_ids.to(device)

    uniq_dates = torch.unique(date_ids)
    uniq_dates_sorted, _ = torch.sort(uniq_dates)

    prev_w_full = torch.zeros(n_tickers, device=device)
    first_step = True
    rp_net_list = []
    rp_gross_list = []
    turnovers = []

    cost = cost_bps / 10000.0

    for d in uniq_dates_sorted:
        mask = (date_ids == d)
        s_d = scores[mask]
        r_d = rets[mask]
        t_d = ticker_ids[mask].long()

        # weights only for tickers present, then place to full vector
        w_d = softmax_weights(s_d, temperature=temperature)

        w_full = torch.zeros(n_tickers, device=device)
        # if duplicate ticker rows exist, we aggregate by summing weights; (shouldn't happen, but safe)
        w_full.index_add_(0, t_d, w_d)

        # normalize again on full vector in case duplicates
        w_sum = w_full.sum().clamp_min(1e-12)
        w_full = w_full / w_sum

        # daily portfolio return (gross) computed only on present tickers:
        # (w_full[t_d] already corresponds to those tickers weights)
        rp_gross = (w_full[t_d] * r_d).sum()

        # turnover vs previous day
        turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
        if (not charge_entry_cost) and first_step:
            turnover = torch.tensor(0.0, device=device)
        first_step = False
        rp_net = rp_gross - turnover * cost

        rp_net_list.append(rp_net)
        rp_gross_list.append(rp_gross)
        turnovers.append(turnover)

        prev_w_full = w_full.detach()

    rp_net = torch.stack(rp_net_list)
    rp_gross = torch.stack(rp_gross_list)
    avg_turnover = torch.stack(turnovers).mean()
    return rp_net, rp_gross, avg_turnover, uniq_dates_sorted


def sharpe_loss(rp: torch.Tensor, eps: float = 1e-8):
    mu = rp.mean()
    sd = rp.std(unbiased=False).clamp_min(eps)
    return -(mu / sd) * math.sqrt(252.0)


def sortino_loss(rp: torch.Tensor, eps: float = 1e-8):
    mu = rp.mean()
    downside = torch.clamp(rp, max=0.0)
    dd = downside.std(unbiased=False).clamp_min(eps)
    return -(mu / dd) * math.sqrt(252.0)


def alpha_ir_net(rp_net: torch.Tensor, rm: torch.Tensor, eps: float = 1e-8):
    alpha = rp_net - rm
    mu = alpha.mean()
    sd = alpha.std(unbiased=False).clamp_min(eps)
    return (mu / sd) * math.sqrt(252.0)


def cum_return(rp: torch.Tensor):
    # Cum = prod(1 + r_t) - 1
    return torch.prod(1.0 + rp) - 1.0

def compute_metric(rp_net: torch.Tensor, rm: torch.Tensor, select_metric: str) -> float:
    """Compute metric on a return series (NET)."""
    if select_metric == "net_sharpe":
        return (rp_net.mean() / rp_net.std(unbiased=False).clamp_min(1e-8) * math.sqrt(252.0)).item()
    if select_metric == "net_sortino":
        downside = torch.clamp(rp_net, max=0.0)
        dd = torch.sqrt(torch.mean(downside * downside)).clamp_min(1e-8)
        return (rp_net.mean() / dd * math.sqrt(252.0)).item()
    if select_metric == "alpha_ir_net":
        return alpha_ir_net(rp_net, rm).item()
    if select_metric == "net_cum":
        return (torch.prod(1.0 + rp_net) - 1.0).item()
    raise ValueError(f"Unknown select_metric: {select_metric}")


# --------------------------
# Evaluation: grid search (K, reb, buf)
# --------------------------
@torch.no_grad()
def eval_grid_robust_metric(
    scores: torch.Tensor,
    rets: torch.Tensor,
    date_ids: torch.Tensor,
    ticker_ids: torch.Tensor,
    n_tickers: int,
    cost_bps: float,
    charge_entry_cost: bool,
    select_metric: str,
    grid_K,
    grid_reb,
    grid_buf,
):
    """
    Non-differentiable evaluation:
      - select Top-K tickers by score each rebalance step
      - buffer: keep part of previous holdings stable
    Robustness (robust selection):
      - split validation dates into 2 halves (validation data = "validačné dáta")
      - score = min(metric_H1, metric_H2)
    Returns best (robust_score, params, per_params_details)
    """
    device = scores.device

    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    T = len(uniq_dates)
    if T < 10:
        return -1e18, None, {}

    # map date -> indices
    date_to_idx = {}
    for d in uniq_dates.tolist():
        d = int(d)
        date_to_idx[d] = torch.where(date_ids == d)[0]

    # market return per date: equal-weight mean over tickers present
    rm = []
    for d in uniq_dates.tolist():
        idx = date_to_idx[int(d)]
        rm.append(rets[idx].mean())
    rm = torch.stack(rm)  # (T,)

    best_rob = -1e18
    best_params = None
    details = {}

    cost = cost_bps / 10000.0

    # helper: simulate for a parameter triple
    def simulate(K, reb, buf):
        K = int(K); reb = int(reb); buf = int(buf)

        prev_hold = None  # tensor of held ticker_ids
        prev_w_full = torch.zeros(n_tickers, device=device)
        first_step = True

        rp_net = []
        turnovers = []

        for t_i, d in enumerate(uniq_dates.tolist()):
            idx = date_to_idx[int(d)]
            s_d = scores[idx]
            r_d = rets[idx]
            t_d = ticker_ids[idx].long()

            # decide if rebalance
            do_reb = (t_i % reb == 0) or (prev_hold is None)

            if do_reb:
                # rank tickers present by score
                # take top (K + buf) as candidate to allow buffer mechanism
                topN = min(len(s_d), K + buf if buf > 0 else K)
                vals, pos = torch.topk(s_d, k=topN, largest=True)

                cand = t_d[pos]  # candidate tickers (ids)
                if prev_hold is None or buf == 0:
                    hold = cand[:min(K, len(cand))]
                else:
                    # buffer idea:
                    # keep up to buf tickers from previous holdings if they are still "good"
                    # (i.e., appear in cand), then fill rest from top scores
                    prev_set = set(prev_hold.tolist())
                    cand_list = cand.tolist()

                    keep = [x for x in cand_list if x in prev_set][:min(buf, K)]
                    fill = [x for x in cand_list if x not in keep][:max(0, K - len(keep))]
                    hold = torch.tensor(keep + fill, device=device, dtype=torch.long)

                prev_hold = hold
            else:
                hold = prev_hold

            # weights: equal weights on hold
            w_full = torch.zeros(n_tickers, device=device)
            if hold.numel() > 0:
                w_full[hold] = 1.0 / float(hold.numel())

            # compute rp_gross only on tickers present:
            # find intersection between hold and t_d
            # easiest: take weights for present tickers and dot with returns
            w_present = w_full[t_d]
            rp_gross = (w_present * r_d).sum()

            turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
            if (not charge_entry_cost) and first_step:
                turnover = torch.tensor(0.0, device=device)
            first_step = False
            rp_net_t = rp_gross - turnover * cost

            rp_net.append(rp_net_t)
            turnovers.append(turnover)

            prev_w_full = w_full

        rp_net = torch.stack(rp_net)  # (T,)
        # robust split
        mid = T // 2
        metric1 = compute_metric(rp_net[:mid], rm[:mid], select_metric)
        metric2 = compute_metric(rp_net[mid:], rm[mid:], select_metric)
        robust = min(metric1, metric2)
        out = {
            "robust": robust,
            "metric1": metric1,
            "metric2": metric2,
            "rm_cum": (torch.prod(1.0 + rm) - 1.0).item(),
            "rp_net_cum": (torch.prod(1.0 + rp_net) - 1.0).item(),
            "avg_turnover": torch.stack(turnovers).mean().item(),
            "T": T,
        }
        return robust, rp_net, rm, out

    for K in grid_K:
        for reb in grid_reb:
            for buf in grid_buf:
                robust, rp_net, rm_series, out = simulate(K, reb, buf)
                details[(int(K), int(reb), int(buf))] = out
                if robust > best_rob:
                    best_rob = robust
                    best_params = (int(K), int(reb), int(buf))

    return best_rob, best_params, details


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
    """
    Evaluate chosen (K, reb, buf) on TEST:
      - report NET Sharpe, NET Sortino, Alpha IR (NET), NET Cum, Excess wealth, avg turnover, T(days)
    """
    device = scores.device
    K, reb, buf = params
    K = int(K); reb = int(reb); buf = int(buf)

    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    T = len(uniq_dates)

    # date -> idx
    date_to_idx = {}
    for d in uniq_dates.tolist():
        date_to_idx[int(d)] = torch.where(date_ids == d)[0]

    # market (equal-weight) per date
    rm = []
    for d in uniq_dates.tolist():
        idx = date_to_idx[int(d)]
        rm.append(rets[idx].mean())
    rm = torch.stack(rm)

    cost = cost_bps / 10000.0
    prev_hold = None
    prev_w_full = torch.zeros(n_tickers, device=device)
    first_step = True

    rp_net = []
    turnovers = []

    for t_i, d in enumerate(uniq_dates.tolist()):
        idx = date_to_idx[int(d)]
        s_d = scores[idx]
        r_d = rets[idx]
        t_d = ticker_ids[idx].long()

        do_reb = (t_i % reb == 0) or (prev_hold is None)
        if do_reb:
            topN = min(len(s_d), K + buf if buf > 0 else K)
            _, pos = torch.topk(s_d, k=topN, largest=True)
            cand = t_d[pos]
            if prev_hold is None or buf == 0:
                hold = cand[:min(K, len(cand))]
            else:
                prev_set = set(prev_hold.tolist())
                cand_list = cand.tolist()
                keep = [x for x in cand_list if x in prev_set][:min(buf, K)]
                fill = [x for x in cand_list if x not in keep][:max(0, K - len(keep))]
                hold = torch.tensor(keep + fill, device=device, dtype=torch.long)
            prev_hold = hold
        else:
            hold = prev_hold

        w_full = torch.zeros(n_tickers, device=device)
        if hold.numel() > 0:
            w_full[hold] = 1.0 / float(hold.numel())

        rp_gross = (w_full[t_d] * r_d).sum()
        turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
        if (not charge_entry_cost) and first_step:
            turnover = torch.tensor(0.0, device=device)
        first_step = False
        rp_net_t = rp_gross - turnover * cost

        rp_net.append(rp_net_t)
        turnovers.append(turnover)
        prev_w_full = w_full

    rp_net = torch.stack(rp_net)
    avg_turnover = torch.stack(turnovers).mean().item()

    net_sh = (rp_net.mean() / rp_net.std(unbiased=False).clamp_min(1e-8) * math.sqrt(252.0)).item()
    # downside volatility
    downside = torch.clamp(rp_net, max=0.0)
    dd = torch.sqrt(torch.mean(downside * downside)).clamp_min(1e-8)
    net_so = (rp_net.mean() / dd * math.sqrt(252.0)).item()

    ir = alpha_ir_net(rp_net, rm).item()
    net_cum = (torch.prod(1.0 + rp_net) - 1.0).item()
    mkt_cum = (torch.prod(1.0 + rm) - 1.0).item()
    excess = net_cum - mkt_cum

    return {
        "NET Sharpe": net_sh,
        "NET Sortino": net_so,
        "Alpha IR (NET)": ir,
        "NET Cum": net_cum,
        "Excess wealth (NET)": excess,
        "Avg turnover": avg_turnover,
        "T (days)": int(T),
    }


# --------------------------
# Training loop
# --------------------------
def linear_ramp(epoch, ramp_start, ramp_end):
    if epoch <= ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    return float(epoch - ramp_start) / float(ramp_end - ramp_start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Sharpe/Sortino objective
    parser.add_argument("--objective", type=str, default="sharpe", choices=["sharpe", "sortino"])
    parser.add_argument("--days_per_batch", type=int, default=20)  # for MultiDateBatchSampler
    parser.add_argument("--temperature", type=float, default=1.0)

    # MSE regularization (stability)
    parser.add_argument("--mse_lambda", type=float, default=0.01)

    # ramp Sharpe/Sortino weight
    parser.add_argument("--ramp_start", type=int, default=5)
    parser.add_argument("--ramp_end", type=int, default=10)

    # costs
    parser.add_argument("--cost_bps", type=float, default=10.0)
    parser.add_argument("--charge_entry_cost", action="store_true")

    # eval grid
    parser.add_argument("--grid_K", type=str, default="5,10,20,40,50")
    parser.add_argument("--grid_reb", type=str, default="5,10")
    parser.add_argument("--grid_buf", type=str, default="0,10,20,40")

    parser.add_argument("--select_metric", type=str, default="net_sortino", choices=["net_sortino","net_sharpe","alpha_ir_net","net_cum"],
                        help="Metric for selecting (K,reb,buf) on validation (robust=min over 2 halves).")
    parser.add_argument("--fixed_policies", type=str, default="5,5,10;5,10,0;20,10,10;50,10,40",
                        help="Semicolon-separated fixed (K,reb,buf) policies to report on TEST.")
    parser.add_argument("--shuffle_dates", action="store_true",
                        help="Shuffle dates in training sampler (NOT recommended with turnover/costs).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="data/lstm_sharpe.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load arrays
    X_train = _try_load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = _try_load(os.path.join(args.data_dir, "y_train.npy"))
    d_train = _try_load(os.path.join(args.data_dir, "date_id_train.npy")).astype(np.int64)
    t_train = _try_load(os.path.join(args.data_dir, "ticker_id_train.npy")).astype(np.int64)

    X_val = _try_load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = _try_load(os.path.join(args.data_dir, "y_val.npy"))
    d_val = _try_load(os.path.join(args.data_dir, "date_id_val.npy")).astype(np.int64)
    t_val = _try_load(os.path.join(args.data_dir, "ticker_id_val.npy")).astype(np.int64)

    X_test = _try_load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = _try_load(os.path.join(args.data_dir, "y_test.npy"))
    d_test = _try_load(os.path.join(args.data_dir, "date_id_test.npy")).astype(np.int64)
    t_test = _try_load(os.path.join(args.data_dir, "ticker_id_test.npy")).astype(np.int64)

    unique_tickers = _try_load(os.path.join(args.data_dir, "unique_tickers.npy"))
    n_tickers = int(len(unique_tickers))

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test:", y_test.shape)

    # build sequence datasets
    train_ds = SequenceDataset(X_train, y_train, d_train, t_train, seq_len=args.seq_len, require_consecutive=True)
    val_ds = SequenceDataset(X_val, y_val, d_val, t_val, seq_len=args.seq_len, require_consecutive=True)
    test_ds = SequenceDataset(X_test, y_test, d_test, t_test, seq_len=args.seq_len, require_consecutive=True)

    # samplers (multi-date batches)
    train_sampler = MultiDateBatchSampler(train_ds.sample_date, days_per_batch=args.days_per_batch, shuffle_dates=args.shuffle_dates, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)

    # for val/test we can just one pass (we need full scores anyway), no sampler needed
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=0)

    input_dim = int(train_ds.X.shape[1])
    model = LSTMModel(input_dim=input_dim, hidden_dim=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # grids
    grid_K = [int(x) for x in args.grid_K.split(",") if x.strip() != ""]
    grid_reb = [int(x) for x in args.grid_reb.split(",") if x.strip() != ""]
    grid_buf = [int(x) for x in args.grid_buf.split(",") if x.strip() != ""]

    best_global = -1e18
    best_global_params = None
    best_state = None

    mse_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()

        w_obj = linear_ramp(epoch, args.ramp_start, args.ramp_end)  # 0 -> 1
        obj_name = "Sharpe" if args.objective == "sharpe" else "Sortino"

        losses = []
        mse_regs = []

        for x_seq, y_t, d_t, t_id in train_loader:
            x_seq = x_seq.to(device)
            y_t = y_t.to(device)
            d_t = d_t.to(device)
            t_id = t_id.to(device)

            scores = model(x_seq)

            # build net rp by day from current batch
            rp_net, rp_gross, avg_to, uniq_dates = compute_net_portfolio_returns_by_day(
                scores=scores,
                rets=y_t,
                date_ids=d_t,
                ticker_ids=t_id,
                n_tickers=n_tickers,
                cost_bps=args.cost_bps,
                charge_entry_cost=args.charge_entry_cost,
                temperature=args.temperature,
            )

            if args.objective == "sharpe":
                obj_loss = sharpe_loss(rp_net)
            else:
                obj_loss = sortino_loss(rp_net)

            # MSE regularization (stability vs exploding predictions)
            mse_reg = mse_fn(scores, y_t)

            loss = w_obj * obj_loss + args.mse_lambda * mse_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(obj_loss.item())
            mse_regs.append(mse_reg.item())

        # ---- Validation: get full scores, then grid-search robust Val NET Alpha IR
        model.eval()

        def collect_scores(loader):
            all_scores = []
            all_y = []
            all_d = []
            all_t = []
            for x_seq, y_t, d_t, t_id in loader:
                x_seq = x_seq.to(device)
                with torch.no_grad():
                    s = model(x_seq)
                all_scores.append(s.detach().cpu())
                all_y.append(y_t.detach().cpu())
                all_d.append(d_t.detach().cpu())
                all_t.append(t_id.detach().cpu())
            return (
                torch.cat(all_scores).to(device),
                torch.cat(all_y).to(device),
                torch.cat(all_d).to(device),
                torch.cat(all_t).to(device),
            )

        val_scores, val_y, val_d, val_t = collect_scores(val_loader)

        # prediction metrics (report only)
        val_mse = torch.mean((val_scores - val_y) ** 2).item()
        val_mae = torch.mean(torch.abs(val_scores - val_y)).item()

        best_epoch_val = -1e18
        best_epoch_params = None

        rob, params, _ = eval_grid_robust_metric(
            scores=val_scores,
            rets=val_y,
            date_ids=val_d,
            ticker_ids=val_t,
            n_tickers=n_tickers,
            cost_bps=args.cost_bps,
            charge_entry_cost=args.charge_entry_cost,
            select_metric=args.select_metric,
            grid_K=grid_K,
            grid_reb=grid_reb,
            grid_buf=grid_buf,
        )
        best_epoch_val = rob
        best_epoch_params = params

        if best_epoch_val > best_global:
            best_global = best_epoch_val
            best_global_params = best_epoch_params
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:>3d} | w_obj={w_obj:0.2f} | train {obj_name}-loss: {np.mean(losses): .6f} "
                f"| mse_reg: {np.mean(mse_regs): .6f} | val MSE: {val_mse: .6e} | val MAE: {val_mae: .6e} "
                f"| best VAL robust metric (epoch): {best_epoch_val: .4f} params(K,reb,buf)={best_epoch_params} "
                f"| best GLOBAL: {best_global: .4f} params={best_global_params}"
            )

    # ---- Load best state and final test evaluation
    print("\n=== SELECTION SUMMARY ===")
    print("Best GLOBAL robust Val metric:", best_global)
    print("Best params (K, rebalance_every, buffer):", best_global_params)
    print(f"Cost config: COST_BPS={args.cost_bps}, CHARGE_ENTRY_COST={args.charge_entry_cost}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # test scores
    test_scores, test_y, test_d, test_t = None, None, None, None

    model.eval()
    all_scores = []
    all_y = []
    all_d = []
    all_t = []
    for x_seq, y_t, d_t, t_id in test_loader:
        x_seq = x_seq.to(device)
        with torch.no_grad():
            s = model(x_seq)
        all_scores.append(s.detach().cpu())
        all_y.append(y_t.detach().cpu())
        all_d.append(d_t.detach().cpu())
        all_t.append(t_id.detach().cpu())

    test_scores = torch.cat(all_scores).to(device)
    test_y = torch.cat(all_y).to(device)
    test_d = torch.cat(all_d).to(device)
    test_t = torch.cat(all_t).to(device)

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

    fixed_pols = parse_fixed_policies(args.fixed_policies) if hasattr(args, "fixed_policies") else []
    if fixed_pols:
        print("\n=== TEST RESULTS (FIXED POLICIES) ===")
        for pol in fixed_pols:
            res_pol = eval_selected_params_on_test(
                scores=test_scores,
                rets=test_y,
                date_ids=test_d,
                ticker_ids=test_t,
                n_tickers=n_tickers,
                cost_bps=args.cost_bps,
                charge_entry_cost=args.charge_entry_cost,
                params=pol,
            )
            print(f"\nPolicy (K,reb,buf)={tuple(pol)}")
            for kk, vv in res_pol.items():
                if isinstance(vv, float):
                    print(f"{kk}: {vv:.6f}")
                else:
                    print(f"{kk}: {vv}")

        # save
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "params": {
                    "seq_len": args.seq_len,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "objective": args.objective,
                    "best_params": best_global_params,
                    "best_val_robust_net_alpha_ir": best_global,
                },
            },
            args.save_path,
        )
        print(f"\nSaved: {args.save_path}")

# fixed policy evaluation (optional)



if __name__ == "__main__":
    main()