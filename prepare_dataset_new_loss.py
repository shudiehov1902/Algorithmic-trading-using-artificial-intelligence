import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ======================
# CONFIG
# ======================
DATA_DIR = "/Users/vladshudegov/BP/Algorithmic-trading-using-artificial-intelligence/data"  # zmeň ak treba
OUT_DIR = "data"

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

LAGS = 10
RET_COL = "ret"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# LOAD ALL TICKERS
# ======================
dfs = []
for fn in sorted(os.listdir(DATA_DIR)):
    if not fn.endswith(".csv"):
        continue
    ticker = fn.replace(".csv", "")
    path = os.path.join(DATA_DIR, fn)
    df = pd.read_csv(path)

    # normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    if "open_date" not in df.columns:
        # fallback if dataset uses different naming
        for cand in ["date", "datetime", "timestamp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "open_date"})
                break

    df["open_date"] = pd.to_datetime(df["open_date"])
    df = df.sort_values("open_date").reset_index(drop=True)

    # daily return from close
    if "close" not in df.columns:
        raise ValueError(f"{fn}: missing 'close' column")
    df[RET_COL] = df["close"].pct_change()

    df["ticker"] = ticker
    dfs.append(df[["open_date", "ticker", RET_COL]])

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.dropna(subset=[RET_COL]).reset_index(drop=True)

# ======================
# CREATE LAGS
# ======================
# for each ticker: ret_lag_1..ret_lag_LAGS
df_all = df_all.sort_values(["ticker", "open_date"]).reset_index(drop=True)

for i in range(1, LAGS + 1):
    df_all[f"ret_lag_{i}"] = df_all.groupby("ticker")[RET_COL].shift(i)

# target: next day return (t)
# input: previous LAGS returns
# -> after shift, current row uses ret_lag_1..LAGS, target is ret(t)
df_all = df_all.dropna(subset=[f"ret_lag_{i}" for i in range(1, LAGS + 1)]).reset_index(drop=True)

# ======================
# DATE + TICKER IDS (for grouping / batch sampling / backtest)
# ======================
# date_id for each unique date
unique_dates = np.array(sorted(df_all["open_date"].dt.normalize().unique()), dtype="datetime64[ns]")
date_to_id = {pd.Timestamp(d).normalize(): i for i, d in enumerate(unique_dates)}
df_all["date_id"] = df_all["open_date"].dt.normalize().map(date_to_id).astype(np.int64)

# ticker_id for each ticker
unique_tickers = np.array(sorted(df_all["ticker"].astype(str).unique()), dtype=str)
ticker_to_id = {t: i for i, t in enumerate(unique_tickers)}
df_all["ticker_id"] = df_all["ticker"].astype(str).map(ticker_to_id).astype(np.int64)

# ======================
# SPLIT
# ======================
df_all = df_all.sort_values("open_date").reset_index(drop=True)

train_mask = df_all["open_date"] <= pd.Timestamp(TRAIN_END)
val_mask = (df_all["open_date"] > pd.Timestamp(TRAIN_END)) & (df_all["open_date"] <= pd.Timestamp(VAL_END))
test_mask = df_all["open_date"] > pd.Timestamp(VAL_END)

# ======================
# BUILD X / y
# ======================
feat_cols = [f"ret_lag_{i}" for i in range(1, LAGS + 1)]

X_train = df_all.loc[train_mask, feat_cols].to_numpy(dtype=np.float32)
y_train = df_all.loc[train_mask, RET_COL].to_numpy(dtype=np.float32)

X_val = df_all.loc[val_mask, feat_cols].to_numpy(dtype=np.float32)
y_val = df_all.loc[val_mask, RET_COL].to_numpy(dtype=np.float32)

X_test = df_all.loc[test_mask, feat_cols].to_numpy(dtype=np.float32)
y_test = df_all.loc[test_mask, RET_COL].to_numpy(dtype=np.float32)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test:", y_test.shape)

# ======================
# STANDARDIZE (fit ONLY on train)
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nChecking first 3 std")
print("mean:", X_train_scaled.mean(axis=0)[:3])
print("std: ", X_train_scaled.std(axis=0)[:3])

# ======================
# SAVE MAIN ARRAYS
# ======================
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val_scaled)
np.save(os.path.join(OUT_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test_scaled)
np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

# --- dates (raw) ---
np.save(os.path.join(OUT_DIR, "train_dates.npy"), df_all.loc[train_mask, "open_date"].to_numpy())
np.save(os.path.join(OUT_DIR, "val_dates.npy"),   df_all.loc[val_mask, "open_date"].to_numpy())
np.save(os.path.join(OUT_DIR, "test_dates.npy"),  df_all.loc[test_mask, "open_date"].to_numpy())

# --- tickers (raw) ---
# dôležité: uložiť ako string dtype, nie object
np.save(os.path.join(OUT_DIR, "train_tickers.npy"), np.array(df_all.loc[train_mask, "ticker"].astype(str).to_numpy(), dtype=str))
np.save(os.path.join(OUT_DIR, "val_tickers.npy"),   np.array(df_all.loc[val_mask, "ticker"].astype(str).to_numpy(), dtype=str))
np.save(os.path.join(OUT_DIR, "test_tickers.npy"),  np.array(df_all.loc[test_mask, "ticker"].astype(str).to_numpy(), dtype=str))

# --- ids pre backtest / grouping ---
train_date_id = df_all.loc[train_mask, "date_id"].to_numpy(dtype=np.int64)
val_date_id   = df_all.loc[val_mask, "date_id"].to_numpy(dtype=np.int64)
test_date_id  = df_all.loc[test_mask, "date_id"].to_numpy(dtype=np.int64)

train_ticker_id = df_all.loc[train_mask, "ticker_id"].to_numpy(dtype=np.int64)
val_ticker_id   = df_all.loc[val_mask, "ticker_id"].to_numpy(dtype=np.int64)
test_ticker_id  = df_all.loc[test_mask, "ticker_id"].to_numpy(dtype=np.int64)

# original names (у тебя уже были)
np.save(os.path.join(OUT_DIR, "train_date_id.npy"), train_date_id)
np.save(os.path.join(OUT_DIR, "val_date_id.npy"),   val_date_id)
np.save(os.path.join(OUT_DIR, "test_date_id.npy"),  test_date_id)

np.save(os.path.join(OUT_DIR, "train_ticker_id.npy"), train_ticker_id)
np.save(os.path.join(OUT_DIR, "val_ticker_id.npy"),   val_ticker_id)
np.save(os.path.join(OUT_DIR, "test_ticker_id.npy"),  test_ticker_id)

# NEW: aliases for scripts expecting date_id_train.npy naming
np.save(os.path.join(OUT_DIR, "date_id_train.npy"), train_date_id)
np.save(os.path.join(OUT_DIR, "date_id_val.npy"),   val_date_id)
np.save(os.path.join(OUT_DIR, "date_id_test.npy"),  test_date_id)

np.save(os.path.join(OUT_DIR, "ticker_id_train.npy"), train_ticker_id)
np.save(os.path.join(OUT_DIR, "ticker_id_val.npy"),   val_ticker_id)
np.save(os.path.join(OUT_DIR, "ticker_id_test.npy"),  test_ticker_id)

# also save universes (safe dtypes)
np.save(os.path.join(OUT_DIR, "unique_dates.npy"), unique_dates)
np.save(os.path.join(OUT_DIR, "unique_tickers.npy"), unique_tickers)

# scaler params (optional but useful)
np.save(os.path.join(OUT_DIR, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
np.save(os.path.join(OUT_DIR, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

print("\nDataset is saved into ./data/*.npy (including date_id_train/val/test aliases).")