import numpy as np
import pandas as pd

from backtest.engine import (calculate_metrics, momentum_weight_matrix,
                             run_backtest, run_portfolio_backtest)
from tests.conftest import make_bars


def test_flat_position_keeps_capital(bars):
    result = run_backtest(bars, pd.Series(0, index=bars.index), 100_000)
    assert np.allclose(result["equity"], 100_000)
    assert calculate_metrics(result)["Trades"] == 0


def test_buy_and_hold_matches_price_return(bars):
    result = run_backtest(bars, pd.Series(1, index=bars.index), 100_000)
    price_return = bars["close"].iloc[-1] / bars["close"].iloc[0]
    # first day is skipped by the one-day shift; align accordingly
    expected = 100_000 * bars["close"].iloc[-1] / bars["close"].iloc[0]
    assert abs(result["equity"].iloc[-1] / expected - 1) < 1e-9


def test_no_lookahead_shift(bars):
    """A signal firing on day t must only earn day t+1's return."""
    position = pd.Series(0, index=bars.index)
    position.iloc[10] = 1  # long at close of day 10 only
    result = run_backtest(bars, position, 100_000)
    day11_return = bars["close"].pct_change().iloc[11]
    assert abs(result["returns"].iloc[11] - day11_return) < 1e-12
    assert (result["returns"].drop(result["returns"].index[11]) == 0).all()


def test_portfolio_backtest_weights_and_metrics():
    prices = pd.DataFrame({
        "AAA": make_bars(drift=0.003, seed=1)["close"],
        "BBB": make_bars(drift=-0.003, seed=2)["close"],
        "CCC": make_bars(drift=0.0, seed=3)["close"],
    })
    weights = momentum_weight_matrix(prices, lookback=20, trend_filter=50, top_n=2)

    assert (weights >= 0).all().all()                      # long-only
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()       # no leverage
    # the persistent downtrend should essentially never be held
    assert weights["BBB"].sum() < weights["AAA"].sum()

    result = run_portfolio_backtest(prices, weights, 100_000)
    metrics = calculate_metrics(result)
    assert result["equity"].iloc[0] > 0
    assert -1 <= metrics["Maximum Drawdown"] <= 0
    assert metrics["Trades"] >= 0
