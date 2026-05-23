import numpy as np
import pandas as pd

from portfolio_utils import (
    backtest_longonly_buffer_cost,
    build_daily_cache,
    robust_metric,
    sharpe_annual_series,
    sortino_annual_series,
)


def test_robust_metric_uses_worse_half():
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-07-02", "2023-07-03"])
    net = pd.Series([0.10, 0.10, -0.05, -0.05], index=idx)
    mkt = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)

    score = robust_metric(
        net_series=net,
        mkt_series=mkt,
        split_date=pd.Timestamp("2023-07-01"),
        metric="net_cum",
    )

    expected_h2 = (1.0 - 0.05) * (1.0 - 0.05) - 1.0
    assert np.isclose(score, expected_h2)


def test_backtest_applies_turnover_costs():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "pred": [0.9, 0.1, 0.1, 0.9],
            "ret": [0.10, 0.00, 0.00, 0.20],
        }
    )
    cache = build_daily_cache(df)

    gross, net, turnover = backtest_longonly_buffer_cost(
        cache=cache,
        k=1,
        rebalance_every=1,
        buffer=0,
        cost_bps=100.0,
        charge_entry_cost=True,
    )

    assert np.isclose(float(gross.iloc[0]), 0.10)
    assert np.isclose(float(net.iloc[0]), 0.09)
    assert np.isclose(float(turnover.iloc[0]), 1.0)

    assert np.isclose(float(gross.iloc[1]), 0.20)
    assert np.isclose(float(net.iloc[1]), 0.19)
    assert np.isclose(float(turnover.iloc[1]), 1.0)


def test_zero_transaction_cost_makes_gross_equal_net():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "pred": [0.9, 0.1, 0.1, 0.9],
            "ret": [0.10, 0.00, 0.00, 0.20],
        }
    )
    cache = build_daily_cache(df)

    gross, net, turnover = backtest_longonly_buffer_cost(
        cache=cache,
        k=1,
        rebalance_every=1,
        buffer=0,
        cost_bps=0.0,
        charge_entry_cost=True,
    )

    assert np.allclose(gross.to_numpy(), net.to_numpy())
    assert float(turnover.sum()) > 0.0


def test_backtest_has_zero_turnover_without_rebalance():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "pred": [0.9, 0.1, 0.1, 0.9],
            "ret": [0.10, 0.00, 0.05, 0.20],
        }
    )
    cache = build_daily_cache(df)

    gross, net, turnover = backtest_longonly_buffer_cost(
        cache=cache,
        k=1,
        rebalance_every=10,
        buffer=0,
        cost_bps=100.0,
        charge_entry_cost=True,
    )

    assert np.isclose(float(turnover.iloc[1]), 0.0)
    assert np.isclose(float(gross.iloc[1]), 0.05)
    assert np.isclose(float(net.iloc[1]), 0.05)


def test_buffer_reduces_turnover_by_keeping_previous_holding():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "pred": [0.9, 0.2, 0.1, 0.8, 0.9, 0.1],
            "ret": [0.01, 0.00, 0.00, 0.02, 0.03, 0.00],
        }
    )
    cache = build_daily_cache(df)

    _, _, turnover_no_buffer = backtest_longonly_buffer_cost(
        cache=cache,
        k=1,
        rebalance_every=1,
        buffer=0,
        cost_bps=10.0,
        charge_entry_cost=False,
    )
    _, _, turnover_with_buffer = backtest_longonly_buffer_cost(
        cache=cache,
        k=1,
        rebalance_every=1,
        buffer=1,
        cost_bps=10.0,
        charge_entry_cost=False,
    )

    assert np.isclose(float(turnover_no_buffer.iloc[1]), 1.0)
    assert np.isclose(float(turnover_with_buffer.iloc[1]), 0.0)


def test_sharpe_and_sortino_are_finite_on_mixed_returns():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    sharpe = sharpe_annual_series(returns)
    sortino = sortino_annual_series(returns)

    assert np.isfinite(sharpe)
    assert np.isfinite(sortino)
