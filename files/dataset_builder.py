import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBuildResult:
    shapes: Dict[str, tuple]
    feature_cols: List[str]
    files_written: List[str]


def load_price_panel(data_dir: str, lags: int = 10) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise RuntimeError("There are no files in the data directory.")

    all_dfs = []
    for path in files:
        ticker = os.path.basename(path).replace(".csv", "")
        df = pd.read_csv(path, parse_dates=["open_date"])
        df = df.sort_values("open_date")
        df["ret"] = df["close"].pct_change()

        for lag in range(1, lags + 1):
            df[f"ret_lag_{lag}"] = df["ret"].shift(lag)

        df["ticker"] = ticker
        df = df.dropna()

        cols = ["open_date", "ticker", "ret"] + [f"ret_lag_{lag}" for lag in range(1, lags + 1)]
        all_dfs.append(df[cols])

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all.sort_values("open_date").reset_index(drop=True)


def build_dataset_arrays(
    df_all: pd.DataFrame,
    lags: int = 10,
    train_start: str = "2016-01-01",
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
):
    df_all = df_all.copy()
    df_all["open_date"] = pd.to_datetime(df_all["open_date"])
    df_all["ticker"] = df_all["ticker"].astype(str)

    unique_dates = np.array(sorted(df_all["open_date"].dt.normalize().unique()), dtype="datetime64[ns]")
    date_to_id = {pd.Timestamp(date).normalize(): idx for idx, date in enumerate(unique_dates)}
    df_all["date_id"] = df_all["open_date"].dt.normalize().map(date_to_id).astype(np.int64)

    unique_tickers = np.array(sorted(df_all["ticker"].unique()), dtype=str)
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
    df_all["ticker_id"] = df_all["ticker"].map(ticker_to_id).astype(np.int64)

    feature_cols = [f"ret_lag_{lag}" for lag in range(1, lags + 1)]
    target_col = "ret"

    train_mask = (df_all["open_date"] >= train_start) & (df_all["open_date"] <= train_end)
    val_mask = (df_all["open_date"] >= val_start) & (df_all["open_date"] <= val_end)
    test_mask = df_all["open_date"] >= test_start

    X_train = df_all.loc[train_mask, feature_cols].to_numpy()
    y_train = df_all.loc[train_mask, target_col].to_numpy()

    X_val = df_all.loc[val_mask, feature_cols].to_numpy()
    y_val = df_all.loc[val_mask, target_col].to_numpy()

    X_test = df_all.loc[test_mask, feature_cols].to_numpy()
    y_test = df_all.loc[test_mask, target_col].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_date_id = df_all.loc[train_mask, "date_id"].to_numpy(dtype=np.int64)
    val_date_id = df_all.loc[val_mask, "date_id"].to_numpy(dtype=np.int64)
    test_date_id = df_all.loc[test_mask, "date_id"].to_numpy(dtype=np.int64)

    train_ticker_id = df_all.loc[train_mask, "ticker_id"].to_numpy(dtype=np.int64)
    val_ticker_id = df_all.loc[val_mask, "ticker_id"].to_numpy(dtype=np.int64)
    test_ticker_id = df_all.loc[test_mask, "ticker_id"].to_numpy(dtype=np.int64)

    arrays = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "train_dates": df_all.loc[train_mask, "open_date"].to_numpy(),
        "val_dates": df_all.loc[val_mask, "open_date"].to_numpy(),
        "test_dates": df_all.loc[test_mask, "open_date"].to_numpy(),
        "train_tickers": np.array(df_all.loc[train_mask, "ticker"].to_numpy(), dtype=str),
        "val_tickers": np.array(df_all.loc[val_mask, "ticker"].to_numpy(), dtype=str),
        "test_tickers": np.array(df_all.loc[test_mask, "ticker"].to_numpy(), dtype=str),
        "train_date_id": train_date_id,
        "val_date_id": val_date_id,
        "test_date_id": test_date_id,
        "train_ticker_id": train_ticker_id,
        "val_ticker_id": val_ticker_id,
        "test_ticker_id": test_ticker_id,
        "date_id_train": train_date_id,
        "date_id_val": val_date_id,
        "date_id_test": test_date_id,
        "ticker_id_train": train_ticker_id,
        "ticker_id_val": val_ticker_id,
        "ticker_id_test": test_ticker_id,
        "unique_dates": unique_dates,
        "unique_tickers": unique_tickers,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }
    return arrays, feature_cols


def save_dataset_arrays(arrays: Dict[str, np.ndarray], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, arr in arrays.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        written.append(path)
    return written


def build_and_save_dataset(data_dir: str, out_dir: str, lags: int = 10) -> DatasetBuildResult:
    df_all = load_price_panel(data_dir=data_dir, lags=lags)
    arrays, feature_cols = build_dataset_arrays(df_all=df_all, lags=lags)
    files_written = save_dataset_arrays(arrays=arrays, out_dir=out_dir)

    shapes = {name: tuple(arr.shape) for name, arr in arrays.items()}
    return DatasetBuildResult(
        shapes=shapes,
        feature_cols=feature_cols,
        files_written=files_written,
    )
