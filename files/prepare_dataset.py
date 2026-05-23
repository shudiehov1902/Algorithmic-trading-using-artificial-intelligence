import argparse
import os

from dataset_builder import build_and_save_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = "/data/alpaca/alpaca_sp500_etf_2025_1day_open_filled"
DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(BASE_DIR, "data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lagged return arrays from per-ticker OHLC CSV files."
    )
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help=f"Directory with input ticker CSV files. Defaults to DATA_DIR env var or {DEFAULT_DATA_DIR}.",
    )
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("OUT_DIR", os.path.join(BASE_DIR, "data")),
        help="Directory where prepared .npy arrays are written. Defaults to OUT_DIR env var or ./data.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=int(os.environ.get("LAGS", "10")),
        help="Number of lagged return features to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_and_save_dataset(data_dir=args.data_dir, out_dir=args.out_dir, lags=args.lags)

    print("Dataset build summary:")
    for name, shape in result.shapes.items():
        print(f"{name}: {shape}")

    print("\nFeature columns:")
    print(", ".join(result.feature_cols))

    print("\nWritten files:")
    for path in result.files_written:
        print(path)


if __name__ == "__main__":
    main()
