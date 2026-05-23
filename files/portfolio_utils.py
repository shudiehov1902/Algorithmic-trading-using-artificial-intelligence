import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def sharpe_annual_series(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return float("nan")
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    return (mu / sd) * math.sqrt(252) if sd > 0 else float("nan")


def sortino_annual_series(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2:
        return float("nan")
    mu = float(daily_ret.mean())
    downside = daily_ret[daily_ret < 0.0]
    if len(downside) < 2:
        return float("nan")
    dd = float(downside.std(ddof=1))
    return (mu / dd) * math.sqrt(252) if dd > 0 else float("nan")


def cumulative_return(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) == 0:
        return float("nan")
    return float((1.0 + daily_ret).cumprod().iloc[-1] - 1.0)


def align_two(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    a2, b2 = a.align(b, join="inner")
    return a2.dropna(), b2.dropna()


def alpha_series(port_daily: pd.Series, mkt_daily: pd.Series) -> pd.Series:
    p, m = align_two(port_daily, mkt_daily)
    return p - m


def info_ratio_annual(port_daily: pd.Series, mkt_daily: pd.Series) -> float:
    a = alpha_series(port_daily, mkt_daily)
    return sharpe_annual_series(a)


def cumulative_excess_wealth(port_daily: pd.Series, mkt_daily: pd.Series) -> float:
    p, m = align_two(port_daily, mkt_daily)
    if len(p) == 0:
        return float("nan")
    wp = (1.0 + p).cumprod().iloc[-1]
    wm = (1.0 + m).cumprod().iloc[-1]
    return float(wp - wm)


def metric_on_series(net_series: pd.Series, mkt_series: pd.Series, metric: str) -> float:
    if metric == "net_sortino":
        return sortino_annual_series(net_series)
    if metric == "net_sharpe":
        return sharpe_annual_series(net_series)
    if metric == "alpha_ir_net":
        return info_ratio_annual(net_series, mkt_series)
    if metric == "net_cum":
        return cumulative_return(net_series)
    raise ValueError(f"Unknown select metric: {metric}")


def robust_metric(net_series: pd.Series, mkt_series: pd.Series, split_date: pd.Timestamp, metric: str) -> float:
    net_series = net_series.dropna()
    mkt_series = mkt_series.dropna()
    h1_net = net_series[net_series.index < split_date]
    h1_mkt = mkt_series[mkt_series.index < split_date]
    h2_net = net_series[net_series.index >= split_date]
    h2_mkt = mkt_series[mkt_series.index >= split_date]
    m1 = metric_on_series(h1_net, h1_mkt, metric)
    m2 = metric_on_series(h2_net, h2_mkt, metric)
    return float(min(m1, m2))


@dataclass
class DailyCache:
    dates: List[pd.Timestamp]
    tickers_sorted: List[List[str]]
    rank_maps: List[Dict[str, int]]
    ret_maps: List[Dict[str, float]]
    mkt_daily: pd.Series


def build_daily_cache(df: pd.DataFrame) -> DailyCache:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values("date")

    dates: List[pd.Timestamp] = []
    tickers_sorted: List[List[str]] = []
    rank_maps: List[Dict[str, int]] = []
    ret_maps: List[Dict[str, float]] = []

    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("pred", ascending=False)
        tickers = g["ticker"].astype(str).tolist()
        rets = g["ret"].astype(float).to_numpy()

        dates.append(pd.Timestamp(d))
        tickers_sorted.append(tickers)
        rank_maps.append({ticker: i + 1 for i, ticker in enumerate(tickers)})
        ret_maps.append({ticker: float(ret) for ticker, ret in zip(tickers, rets)})

    mkt = df.groupby("date", sort=True)["ret"].mean().sort_index()
    return DailyCache(
        dates=dates,
        tickers_sorted=tickers_sorted,
        rank_maps=rank_maps,
        ret_maps=ret_maps,
        mkt_daily=mkt,
    )


def backtest_longonly_buffer_cost(
    cache: DailyCache,
    k: int,
    rebalance_every: int,
    buffer: int,
    cost_bps: float,
    charge_entry_cost: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    holdings: Set[str] = set()
    gross: List[float] = []
    net: List[float] = []
    turnover: List[float] = []

    for i in range(len(cache.dates)):
        top_list = cache.tickers_sorted[i]
        rank_map = cache.rank_maps[i]
        ret_map = cache.ret_maps[i]

        if i == 0:
            new_holdings = set(top_list[:k])
            to = 1.0 if charge_entry_cost else 0.0
            holdings = new_holdings
        else:
            do_reb = i % rebalance_every == 0
            if do_reb:
                prev = holdings
                kept = set()
                threshold = k + buffer
                for ticker in prev:
                    if rank_map.get(ticker, 10**9) <= threshold:
                        kept.add(ticker)

                new_holdings = set(kept)
                for ticker in top_list:
                    if len(new_holdings) >= k:
                        break
                    if ticker not in new_holdings:
                        new_holdings.add(ticker)

                overlap = len(prev.intersection(new_holdings))
                to = 1.0 - overlap / float(k)
                holdings = new_holdings
            else:
                to = 0.0

        held_rets = [ret_map[ticker] for ticker in holdings if ticker in ret_map]
        gross_ret = float(np.mean(held_rets)) if held_rets else 0.0
        cost = to * (cost_bps / 10000.0)
        net_ret = gross_ret - cost

        gross.append(gross_ret)
        net.append(net_ret)
        turnover.append(to)

    idx = pd.to_datetime(cache.dates)
    return pd.Series(gross, index=idx), pd.Series(net, index=idx), pd.Series(turnover, index=idx)
