#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
# Repro (as much as CUDA allows)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
START_FROM="${1:-}"
FOUND_START=0
RAN_ANY=0

if [[ -z "$START_FROM" ]]; then
  FOUND_START=1
fi

mkdir -p logs

required_files=(
  "data/X_train.npy"
  "data/y_train.npy"
  "data/X_val.npy"
  "data/y_val.npy"
  "data/X_test.npy"
  "data/y_test.npy"
  "data/date_id_train.npy"
  "data/date_id_val.npy"
  "data/date_id_test.npy"
  "data/ticker_id_train.npy"
  "data/ticker_id_val.npy"
  "data/ticker_id_test.npy"
  "data/unique_tickers.npy"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

run() {
  local name="$1"
  shift
  if [[ "$FOUND_START" -eq 0 ]]; then
    if [[ "$name" != "$START_FROM" ]]; then
      echo "=== Skipping $name (waiting for $START_FROM) ==="
      return 0
    fi
    FOUND_START=1
  fi
  RAN_ANY=1
  echo
  echo "=== Running $name ==="
  echo "Command: $*"
  "$@" 2>&1 | tee "logs/${name}.log"
}

run "mlp_mse_2" \
  python3 mlp_mse_2.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "mlp_mae_2" \
  python3 mlp_mae_2.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "lstm_mse_2" \
  python3 lstm_mse_2.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "lstm_mae_2" \
  python3 lstm_mae_2.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "mlp_sharpe" \
  python3 mlp_sharpe.py \
  --data_dir data --epochs 50 --seed 42 \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --K_list 5,10,20,30,40,50,75,100 \
  --reb_list 5,10,20 \
  --buf_list 0,10,20,40 \
  --save_path data/mlp_sharpe.pt

run "mlp_sortino" \
  python3 mlp_sortino.py \
  --data_dir data --epochs 50 --seed 42 \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --K_list 5,10,20,30,40,50,75,100 \
  --reb_list 5,10,20 \
  --buf_list 0,10,20,40 \
  --save_path data/mlp_sortino.pt

run "lstm_sharpe" \
  python3 lstm_sharpe.py \
  --charge_entry_cost --cost_bps 10 \
  --days_per_batch 10 \
  --seq_len 20 \
  --mse_lambda 0.01 \
  --ramp_start 5 --ramp_end 10

run "lstm_sortino" \
  python3 lstm_sortino.py \
  --charge_entry_cost --cost_bps 10 \
  --days_per_batch 20 \
  --seq_len 20 \
  --mse_lambda 0.01 \
  --ramp_start 5 --ramp_end 10

run "stock_mixer_mse_with_fee" \
  python3 stock_mixer_mse_with_fee.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "stock_mixer_mae_with_fee" \
  python3 stock_mixer_mae_with_fee.py \
  --charge_entry_cost --cost_bps 10 \
  --select_metric net_sortino \
  --grid_reb 5,10,20 --grid_buf 0,10,20,40 --grid_K 5,10,20,30,40,50,75,100 \
  --max_avg_turnover 0.12

run "stock_mixer_sharpe" \
  python3 stock_mixer_sharpe.py \
    --data_dir data \
    --seed 42 --epochs 50 \
    --charge_entry_cost \
    --cost_bps 10 \
    --select_metric net_sortino \
    --grid_K 5,10,20,30,40,50,75,100 \
    --grid_reb 5,10,20 \
    --grid_buf 0,10,20,40 \
    --max_avg_turnover 0.12 \
    --save_path data/stock_mixer_sharpe.pt

run "stock_mixer_sortino" \
  python3 stock_mixer_sortino.py \
    --data_dir data \
    --seed 42 --epochs 50 \
    --charge_entry_cost \
    --cost_bps 10 \
    --select_metric net_sortino \
    --grid_K 5,10,20,30,40,50,75,100 \
    --grid_reb 5,10,20 \
    --grid_buf 0,10,20,40 \
    --max_avg_turnover 0.12 \
    --save_path data/stock_mixer_sortino.pt

if [[ "$RAN_ANY" -eq 0 ]]; then
  echo "Unknown start model: $START_FROM" >&2
  echo "Available models:" >&2
  echo "  mlp_mse_2 mlp_mae_2 lstm_mse_2 lstm_mae_2 mlp_sharpe mlp_sortino lstm_sharpe lstm_sortino stock_mixer_mse_with_fee stock_mixer_mae_with_fee stock_mixer_sharpe stock_mixer_sortino" >&2
  exit 1
fi

echo
echo "All runs completed. Logs are in $ROOT_DIR/logs/"
