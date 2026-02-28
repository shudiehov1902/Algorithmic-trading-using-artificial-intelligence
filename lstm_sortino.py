# lstm_sortino.py
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
        if arr.dtype == object:
            arr = np.array(list(arr))
        return arr
    return np.load(path, allow_pickle=False)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        date_id = np.asarray(date_id, dtype=np.int64)
        ticker_id = np.asarray(ticker_id, dtype=np.int64)

        order = np.lexsort((date_id, ticker_id))
        self.X = X[order]
        self.y = y[order]
        self.date_id = date_id[order]
        self.ticker_id = ticker_id[order]

        self.end_positions = []
        self.sample_date = []
        self.sample_ticker = []

        n = len(self.X)
        if n == 0:
            raise ValueError("Empty dataset arrays.")

        start = 0
        while start < n:
            tid = self.ticker_id[start]
            end = start
            while end < n and self.ticker_id[end] == tid:
                end += 1

            if self.require_consecutive:
                seg_start = start
                for i in range(start + 1, end):
                    if self.date_id[i] != self.date_id[i - 1] + 1:
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
        out, _ = self.lstm(x_seq)     # (B, T, H)
        last = out[:, -1, :]          # (B, H)
        score = self.head(last).squeeze(-1)  # (B,)
        return score


# --------------------------
# Portfolio math: weights, turnover, net rp
# --------------------------
def softmax_weights(scores: torch.Tensor, temperature: float = 1.0):
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
    Uses FULL vector size n_tickers to avoid size mismatch.
    Returns:
      rp_net: (n_days,)
      rp_gross: (n_days,)
      avg_turnover
      uniq_dates_sorted
    """
    device = scores.device
    date_ids = date_ids.to(device)
    ticker_ids = ticker_ids.to(device)

    uniq_dates = torch.unique(date_ids)
    uniq_dates_sorted, _ = torch.sort(uniq_dates)

    prev_w_full = torch.zeros(n_tickers, device=device)
    rp_net_list = []
    rp_gross_list = []
    turnovers = []

    cost = cost_bps / 10000.0

    for d in uniq_dates_sorted:
        mask = (date_ids == d)
        s_d = scores[mask]
        r_d = rets[mask]
        t_d = ticker_ids[mask].long()

        w_d = softmax_weights(s_d, temperature=temperature)

        w_full = torch.zeros(n_tickers, device=device)
        w_full.index_add_(0, t_d, w_d)

        w_sum = w_full.sum().clamp_min(1e-12)
        w_full = w_full / w_sum

        rp_gross = (w_full[t_d] * r_d).sum()

        turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
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
    dd = torch.sqrt((downside ** 2).mean() + eps)
    return -(mu / dd) * math.sqrt(252.0)


def alpha_ir_net(rp_net: torch.Tensor, rm: torch.Tensor, eps: float = 1e-8):
    alpha = rp_net - rm
    mu = alpha.mean()
    sd = alpha.std(unbiased=False).clamp_min(eps)
    return (mu / sd) * math.sqrt(252.0)


def cum_return(rp: torch.Tensor):
    return torch.prod(1.0 + rp) - 1.0


# --------------------------
# Evaluation: grid search (K, reb, buf) on robust Val NET Alpha IR
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
        return -1e18, None, {}

    date_to_idx = {}
    for d in uniq_dates.tolist():
        d = int(d)
        date_to_idx[d] = torch.where(date_ids == d)[0]

    rm = []
    for d in uniq_dates.tolist():
        idx = date_to_idx[int(d)]
        rm.append(rets[idx].mean())
    rm = torch.stack(rm)

    best_rob = -1e18
    best_params = None
    details = {}

    cost = cost_bps / 10000.0

    def simulate(K, reb, buf):
        K = int(K); reb = int(reb); buf = int(buf)

        prev_hold = None
        prev_w_full = torch.zeros(n_tickers, device=device)

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
            if hold is not None and hold.numel() > 0:
                w_full[hold] = 1.0 / float(hold.numel())

            w_present = w_full[t_d]
            rp_gross = (w_present * r_d).sum()

            turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
            if (not charge_entry_cost) and (t_i == 0):
                turnover = torch.tensor(0.0, device=device)

            rp_net_t = rp_gross - turnover * cost

            rp_net.append(rp_net_t)
            turnovers.append(turnover)

            prev_w_full = w_full

        rp_net = torch.stack(rp_net)

        mid = T // 2
        ir1 = alpha_ir_net(rp_net[:mid], rm[:mid]).item()
        ir2 = alpha_ir_net(rp_net[mid:], rm[mid:]).item()
        robust = min(ir1, ir2)

        out = {
            "robust": robust,
            "ir1": ir1,
            "ir2": ir2,
            "rm_cum": (torch.prod(1.0 + rm) - 1.0).item(),
            "rp_net_cum": (torch.prod(1.0 + rp_net) - 1.0).item(),
            "avg_turnover": torch.stack(turnovers).mean().item(),
            "T": T,
        }
        return robust, rp_net, rm, out

    for K in grid_K:
        for reb in grid_reb:
            for buf in grid_buf:
                robust, _, _, out = simulate(K, reb, buf)
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
    device = scores.device
    K, reb, buf = params
    K = int(K); reb = int(reb); buf = int(buf)

    uniq_dates = torch.unique(date_ids)
    uniq_dates, _ = torch.sort(uniq_dates)
    T = len(uniq_dates)

    date_to_idx = {}
    for d in uniq_dates.tolist():
        date_to_idx[int(d)] = torch.where(date_ids == d)[0]

    rm = []
    for d in uniq_dates.tolist():
        idx = date_to_idx[int(d)]
        rm.append(rets[idx].mean())
    rm = torch.stack(rm)

    cost = cost_bps / 10000.0
    prev_hold = None
    prev_w_full = torch.zeros(n_tickers, device=device)

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
        if hold is not None and hold.numel() > 0:
            w_full[hold] = 1.0 / float(hold.numel())

        rp_gross = (w_full[t_d] * r_d).sum()
        turnover = 0.5 * torch.abs(w_full - prev_w_full).sum()
        if (not charge_entry_cost) and (t_i == 0):
            turnover = torch.tensor(0.0, device=device)

        rp_net_t = rp_gross - turnover * cost

        rp_net.append(rp_net_t)
        turnovers.append(turnover)
        prev_w_full = w_full

    rp_net = torch.stack(rp_net)
    avg_turnover = torch.stack(turnovers).mean().item()

    net_sh = (rp_net.mean() / rp_net.std(unbiased=False).clamp_min(1e-8) * math.sqrt(252.0)).item()
    downside = torch.clamp(rp_net, max=0.0)
    net_so = (rp_net.mean() / downside.std(unbiased=False).clamp_min(1e-8) * math.sqrt(252.0)).item()

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

    parser.add_argument("--days_per_batch", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--mse_lambda", type=float, default=0.01)

    parser.add_argument("--ramp_start", type=int, default=5)
    parser.add_argument("--ramp_end", type=int, default=10)

    parser.add_argument("--cost_bps", type=float, default=10.0)
    parser.add_argument("--charge_entry_cost", action="store_true")

    parser.add_argument("--grid_K", type=str, default="5,10,20,40,50")
    parser.add_argument("--grid_reb", type=str, default="5,10")
    parser.add_argument("--grid_buf", type=str, default="0,10,20,40")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="data/lstm_sortino.pt")
    args = parser.parse_args()
    args.objective = "sortino"  # fixed for this script

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print("X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("X_test: ", X_test.shape, "y_test:", y_test.shape)

    train_ds = SequenceDataset(X_train, y_train, d_train, t_train, seq_len=args.seq_len, require_consecutive=True)
    val_ds = SequenceDataset(X_val, y_val, d_val, t_val, seq_len=args.seq_len, require_consecutive=True)
    test_ds = SequenceDataset(X_test, y_test, d_test, t_test, seq_len=args.seq_len, require_consecutive=True)

    train_sampler = MultiDateBatchSampler(train_ds.sample_date, days_per_batch=args.days_per_batch, shuffle_dates=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=0)

    input_dim = int(train_ds.X.shape[1])
    model = LSTMModel(input_dim=input_dim, hidden_dim=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    grid_K = [int(x) for x in args.grid_K.split(",") if x.strip() != ""]
    grid_reb = [int(x) for x in args.grid_reb.split(",") if x.strip() != ""]
    grid_buf = [int(x) for x in args.grid_buf.split(",") if x.strip() != ""]

    best_global = -1e18
    best_global_params = None
    best_state = None

    mse_fn = nn.MSELoss()

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

    for epoch in range(1, args.epochs + 1):
        model.train()

        w_obj = linear_ramp(epoch, args.ramp_start, args.ramp_end)
        obj_name = "Sortino"

        losses = []
        mse_regs = []

        for x_seq, y_t, d_t, t_id in train_loader:
            x_seq = x_seq.to(device)
            y_t = y_t.to(device)
            d_t = d_t.to(device)
            t_id = t_id.to(device)

            scores = model(x_seq)

            rp_net, _, _, _ = compute_net_portfolio_returns_by_day(
                scores=scores,
                rets=y_t,
                date_ids=d_t,
                ticker_ids=t_id,
                n_tickers=n_tickers,
                cost_bps=args.cost_bps,
                charge_entry_cost=args.charge_entry_cost,
                temperature=args.temperature,
            )

            obj_loss = sortino_loss(rp_net)

            mse_reg = mse_fn(scores, y_t)

            loss = w_obj * obj_loss + args.mse_lambda * mse_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(obj_loss.item())
            mse_regs.append(mse_reg.item())

        model.eval()

        val_scores, val_y, val_d, val_t = collect_scores(val_loader)

        val_mse = torch.mean((val_scores - val_y) ** 2).item()
        val_mae = torch.mean(torch.abs(val_scores - val_y)).item()

        best_epoch_val, best_epoch_params, _ = eval_grid_robust_net_alpha_ir(
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
                f"Epoch {epoch:>3d} | w_obj={w_obj:0.2f} | train {obj_name}-loss: {np.mean(losses): .6f} "
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

    test_scores, test_y, test_d, test_t = collect_scores(test_loader)

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


if __name__ == "__main__":
    main()