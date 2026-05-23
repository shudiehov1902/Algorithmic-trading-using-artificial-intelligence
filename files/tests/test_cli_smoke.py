import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINTS = [
    "prepare_dataset.py",
    "mlp_mse_2.py",
    "mlp_mae_2.py",
    "lstm_mse_2.py",
    "lstm_mae_2.py",
    "mlp_sharpe.py",
    "mlp_sortino.py",
    "lstm_sharpe.py",
    "lstm_sortino.py",
    "stock_mixer_mse_with_fee.py",
    "stock_mixer_mae_with_fee.py",
    "stock_mixer_sharpe.py",
    "stock_mixer_sortino.py",
]


@pytest.mark.smoke
@pytest.mark.parametrize("script_name", ENTRYPOINTS)
def test_entrypoint_help_works(script_name):
    script_path = REPO_ROOT / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
