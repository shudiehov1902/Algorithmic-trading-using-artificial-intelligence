#!/usr/bin/env python3
"""
prepare_dataset_stockmixer.py

Готовит датасет для:
- MLP / LSTM: X_2d (N, seq_len)
- StockMixer: X_3d (N, seq_len, n_features) = (N, 10, 1)

Фичи:
  - ret_lag_1 ... ret_lag_10
Таргет:
  - ret (доходность следующего дня)

Сплит по датам:
  - train: open_date < 2023-01-01
  - val:   2023-01-01 <= open_date < 2024-01-01
  - test:  open_date >= 2024-01-01
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# --- Настройки ---
SEQ_LEN = 10  # число лагов
# Каталоги с сырыми данными (сервер / локально)
POSSIBLE_DATA_DIRS = [
    Path("/data/alpaca/alpaca_sp500_etf_2025_1day_open_filled"),  # Athena
    Path("./data"),               # локально
]

OUT_DIR = Path("./data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_dir() -> Path:
    for p in POSSIBLE_DATA_DIRS:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"Не найден каталог с данными. Проверил: {POSSIBLE_DATA_DIRS}"
    )


def process_one_ticker(csv_path: Path) -> pd.DataFrame:
    """Считать один TICKER.csv, посчитать ret и лаги, вернуть очищенный df."""
    ticker = csv_path.stem

    df = pd.read_csv(csv_path)
    # Пытаемся угадать имя колонки даты
    if "open_date" in df.columns:
        date_col = "open_date"
    elif "time" in df.columns:
        date_col = "time"
    else:
        raise ValueError(f"Не найден столбец с датой в {csv_path}")

    if "close" not in df.columns:
        raise ValueError(f"Не найден столбец 'close' в {csv_path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Дневная доходность
    df["ret"] = df["close"].pct_change()

    # Лаги доходности ret_lag_1 ... ret_lag_10
    for k in range(1, SEQ_LEN + 1):
        df[f"ret_lag_{k}"] = df["ret"].shift(k)

    # Убираем строки, где нет ретёрна или лагов
    needed_cols = ["ret"] + [f"ret_lag_{k}" for k in range(1, SEQ_LEN + 1)]
    df = df.dropna(subset=needed_cols)

    if df.empty:
        return df  # пусто — пропустим выше

    df["ticker"] = ticker
    df = df[[date_col, "ticker", "ret"] + [f"ret_lag_{k}" for k in range(1, SEQ_LEN + 1)]]
    df = df.rename(columns={date_col: "open_date"})
    return df


def main():
    data_dir = find_data_dir()
    print(f"[INFO] Использую каталог с данными: {data_dir}")

    all_dfs = []

    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"[INFO] Найдено CSV-файлов: {len(csv_files)}")

    for i, csv_path in enumerate(csv_files, start=1):
        try:
            df_t = process_one_ticker(csv_path)
        except Exception as e:
            print(f"[WARN] Пропускаю {csv_path.name}: {e}")
            continue

        if df_t.empty:
            print(f"[WARN] {csv_path.name}: после очистки нет данных, пропускаю.")
            continue

        all_dfs.append(df_t)

        if i % 50 == 0:
            print(f"[INFO] Обработано тикеров: {i}/{len(csv_files)}")

    if not all_dfs:
        raise RuntimeError("Нет данных ни по одному тикеру после предобработки.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["open_date"] = pd.to_datetime(df_all["open_date"])
    df_all = df_all.sort_values(["open_date", "ticker"]).reset_index(drop=True)

    print(f"[INFO] Итоговая таблица df_all: {df_all.shape[0]} строк")
    print(df_all.head())

    # --- Train / Val / Test сплит по датам ---
    train_mask = df_all["open_date"] < pd.Timestamp("2023-01-01")
    val_mask = (df_all["open_date"] >= pd.Timestamp("2023-01-01")) & (
        df_all["open_date"] < pd.Timestamp("2024-01-01")
    )
    test_mask = df_all["open_date"] >= pd.Timestamp("2024-01-01")

    print("[INFO] Разбиение по датам:")
    print("  train:", train_mask.sum())
    print("  val:  ", val_mask.sum())
    print("  test: ", test_mask.sum())

    feat_cols = [f"ret_lag_{k}" for k in range(1, SEQ_LEN + 1)]
    X_all = df_all[feat_cols].to_numpy(dtype=np.float32)
    y_all = df_all["ret"].to_numpy(dtype=np.float32)

    # --- Стандартизация по train ---
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_all[train_mask])
    X_val_flat = scaler.transform(X_all[val_mask])
    X_test_flat = scaler.transform(X_all[test_mask])

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    # --- Вариант для MLP / LSTM: 2D (N, seq_len) ---
    np.save(OUT_DIR / "X_train_mlp.npy", X_train_flat)
    np.save(OUT_DIR / "X_val_mlp.npy", X_val_flat)
    np.save(OUT_DIR / "X_test_mlp.npy", X_test_flat)

    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "y_test.npy", y_test)

    # --- Вариант для StockMixer: 3D (N, seq_len, 1) ---
    def to_3d(x_flat: np.ndarray) -> np.ndarray:
        # x_flat: (N, seq_len) -> (N, seq_len, 1)
        return x_flat.reshape(x_flat.shape[0], SEQ_LEN, 1)

    X_train_sm = to_3d(X_train_flat)
    X_val_sm = to_3d(X_val_flat)
    X_test_sm = to_3d(X_test_flat)

    np.save(OUT_DIR / "X_train_sm.npy", X_train_sm)
    np.save(OUT_DIR / "X_val_sm.npy", X_val_sm)
    np.save(OUT_DIR / "X_test_sm.npy", X_test_sm)

    # --- Дополнительно сохраняем даты и тикеры (полезно для стратегии) ---
    np.save(OUT_DIR / "dates_train.npy", df_all.loc[train_mask, "open_date"].to_numpy("datetime64[D]"))
    np.save(OUT_DIR / "dates_val.npy", df_all.loc[val_mask, "open_date"].to_numpy("datetime64[D]"))
    np.save(OUT_DIR / "dates_test.npy", df_all.loc[test_mask, "open_date"].to_numpy("datetime64[D]"))

    np.save(OUT_DIR / "tickers_train.npy", df_all.loc[train_mask, "ticker"].to_numpy())
    np.save(OUT_DIR / "tickers_val.npy", df_all.loc[val_mask, "ticker"].to_numpy())
    np.save(OUT_DIR / "tickers_test.npy", df_all.loc[test_mask, "ticker"].to_numpy())

    print("[INFO] Сохранено в", OUT_DIR.resolve())
    print("[INFO] Формы массивов:")
    print("  X_train_mlp:", X_train_flat.shape)
    print("  X_train_sm: ", X_train_sm.shape)
    print("  y_train:    ", y_train.shape)


if __name__ == "__main__":
    main()
