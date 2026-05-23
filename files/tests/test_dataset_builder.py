from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder import build_and_save_dataset, load_price_panel


def _write_ticker_csv(path: Path, closes):
    dates = pd.to_datetime(
        [
            "2022-12-28",
            "2022-12-29",
            "2022-12-30",
            "2022-12-31",
            "2023-01-02",
            "2023-01-03",
            "2024-01-02",
            "2024-01-03",
        ]
    )
    df = pd.DataFrame({"open_date": dates, "close": closes})
    df.to_csv(path, index=False)


def test_load_price_panel_builds_lagged_rows(tmp_path):
    _write_ticker_csv(tmp_path / "AAA.csv", [100, 101, 102, 103, 104, 105, 106, 107])
    _write_ticker_csv(tmp_path / "BBB.csv", [200, 198, 202, 201, 205, 210, 211, 212])

    df = load_price_panel(data_dir=str(tmp_path), lags=2)

    assert set(df["ticker"].unique()) == {"AAA", "BBB"}
    assert {"open_date", "ticker", "ret", "ret_lag_1", "ret_lag_2"}.issubset(df.columns)
    assert len(df) == 10


def test_build_and_save_dataset_writes_expected_arrays(tmp_path):
    data_dir = tmp_path / "csv"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    _write_ticker_csv(data_dir / "AAA.csv", [100, 101, 102, 103, 104, 105, 106, 107])
    _write_ticker_csv(data_dir / "BBB.csv", [200, 198, 202, 201, 205, 210, 211, 212])

    result = build_and_save_dataset(data_dir=str(data_dir), out_dir=str(out_dir), lags=2)

    assert result.shapes["X_train"] == (2, 2)
    assert result.shapes["X_val"] == (4, 2)
    assert result.shapes["X_test"] == (4, 2)
    assert len(result.files_written) >= 25

    x_train = np.load(out_dir / "X_train.npy")
    y_train = np.load(out_dir / "y_train.npy")
    val_dates = np.load(out_dir / "val_dates.npy", allow_pickle=True)

    assert x_train.shape == (2, 2)
    assert y_train.shape == (2,)
    assert len(val_dates) == 4
    assert np.allclose(x_train.mean(axis=0), 0.0, atol=1e-9)


def test_build_and_save_dataset_writes_files_required_by_run_all_models(tmp_path):
    data_dir = tmp_path / "csv"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    _write_ticker_csv(data_dir / "AAA.csv", [100, 101, 102, 103, 104, 105, 106, 107])
    _write_ticker_csv(data_dir / "BBB.csv", [200, 198, 202, 201, 205, 210, 211, 212])

    build_and_save_dataset(data_dir=str(data_dir), out_dir=str(out_dir), lags=2)

    required_run_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "date_id_train.npy",
        "date_id_val.npy",
        "date_id_test.npy",
        "ticker_id_train.npy",
        "ticker_id_val.npy",
        "ticker_id_test.npy",
        "unique_tickers.npy",
    ]

    missing = [name for name in required_run_files if not (out_dir / name).exists()]
    assert missing == []

    assert np.load(out_dir / "date_id_train.npy").shape == np.load(out_dir / "y_train.npy").shape
    assert np.load(out_dir / "ticker_id_val.npy").shape == np.load(out_dir / "y_val.npy").shape
    assert set(np.load(out_dir / "unique_tickers.npy").astype(str)) == {"AAA", "BBB"}
