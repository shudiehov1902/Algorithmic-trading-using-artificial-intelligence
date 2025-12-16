import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/data/alpaca/alpaca_sp500_etf_2025_1day_open_filled"
LAGS = 10


files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found files: {len(files)}")

if len(files) == 0:
    raise RuntimeError("There are no files in the data directory.")

all_dfs = []

for path in files:
    ticker = os.path.basename(path).replace(".csv", "")
    print("Downloading ", ticker)

    df = pd.read_csv(path, parse_dates=["open_date"])
    df = df.sort_values("open_date")


    df["ret"] = df["close"].pct_change()


    for lag in range(1, LAGS + 1):
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)


    df["ticker"] = ticker


    df = df.dropna()

    cols = ["open_date", "ticker", "ret"] + [f"ret_lag_{lag}" for lag in range(1, LAGS + 1)]
    df = df[cols]

    all_dfs.append(df)


df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values("open_date").reset_index(drop=True)

print("Итоговый размер df_all:", df_all.shape)
print(df_all.head())


train_mask = (df_all["open_date"] >= "2016-01-01") & (df_all["open_date"] <= "2022-12-31")
val_mask   = (df_all["open_date"] >= "2023-01-01") & (df_all["open_date"] <= "2023-12-31")
test_mask  = (df_all["open_date"] >= "2024-01-01")

feature_cols = [f"ret_lag_{lag}" for lag in range(1, LAGS + 1)]
target_col = "ret"

X_train = df_all.loc[train_mask, feature_cols].to_numpy()
y_train = df_all.loc[train_mask, target_col].to_numpy()

X_val = df_all.loc[val_mask, feature_cols].to_numpy()
y_val = df_all.loc[val_mask, target_col].to_numpy()

X_test = df_all.loc[test_mask, feature_cols].to_numpy()
y_test = df_all.loc[test_mask, target_col].to_numpy()

print("\nBefore zooming:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test:", y_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nChecking first 3 std")
print("mean:", X_train_scaled.mean(axis=0)[:3])
print("std: ", X_train_scaled.std(axis=0)[:3])


np.save("data/X_train.npy", X_train_scaled)
np.save("data/y_train.npy", y_train)
np.save("data/X_val.npy",   X_val_scaled)
np.save("data/y_val.npy",   y_val)
np.save("data/X_test.npy",  X_test_scaled)
np.save("data/y_test.npy",  y_test)


np.save("data/train_dates.npy", df_all.loc[train_mask, "open_date"].to_numpy())
np.save("data/val_dates.npy",   df_all.loc[val_mask, "open_date"].to_numpy())
np.save("data/test_dates.npy",  df_all.loc[test_mask, "open_date"].to_numpy())

# тикеры
np.save("data/train_tickers.npy", df_all.loc[train_mask, "ticker"].to_numpy())
np.save("data/val_tickers.npy",   df_all.loc[val_mask, "ticker"].to_numpy())
np.save("data/test_tickers.npy",  df_all.loc[test_mask, "ticker"].to_numpy())

print("\nDataset is saved/*.npy")
