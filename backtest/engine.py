"""Backtest engine and performance metrics.

Two backtesters, both long-only with no leverage:
  * run_backtest            — single asset, 0/1 position series (used by the
                              ML research strategy)
  * run_portfolio_backtest  — multi-asset weight matrix (used by the live
                              momentum strategy)
Pure pandas/numpy: unit-testable without Streamlit or Alpaca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- single asset
def run_backtest(df: pd.DataFrame, position: pd.Series,
                 initial_capital: float = 100_000) -> dict:
    """Vectorised long-only backtest for one asset.

    ``position[t]`` is decided at the close of day ``t`` and shifted forward
    one day, so realised P&L is ``position[t] * return[t+1]`` — no look-ahead.
    """
    data = df.copy()
    aligned = position.reindex(data.index).fillna(0).shift(1).fillna(0)
    daily_returns = data["close"].pct_change().fillna(0)
    strategy_returns = daily_returns * aligned
    equity = initial_capital * (1 + strategy_returns).cumprod()

    return {
        "position": aligned,
        "returns": strategy_returns,
        "equity": equity,
        "pnl": equity.diff().fillna(0),
        "cumulative_pnl": equity - initial_capital,
        "drawdown": equity / equity.cummax() - 1,
        "trades": aligned.diff().abs().fillna(0),
        "initial_capital": initial_capital,
    }


# ------------------------------------------------------------------ portfolio
def momentum_weight_matrix(prices: pd.DataFrame, lookback: int = 20,
                           trend_filter: int = 50, top_n: int = 3) -> pd.DataFrame:
    """Historical daily target weights for the cross-sectional momentum rules.

    Mirrors strategy.momentum.MomentumStrategy: rank by ``lookback``-day
    return, require close > SMA(trend_filter) and positive momentum, hold the
    top ``top_n`` equal-weighted.
    """
    momentum = prices / prices.shift(lookback) - 1
    above_trend = prices > prices.rolling(trend_filter).mean()
    eligible = momentum.where(above_trend & (momentum > 0))

    ranks = eligible.rank(axis=1, ascending=False)
    selected = ranks <= top_n
    weights = selected.astype(float) / top_n
    weights[eligible.isna()] = 0.0
    return weights.fillna(0.0)


def run_portfolio_backtest(prices: pd.DataFrame, weights: pd.DataFrame,
                           initial_capital: float = 100_000) -> dict:
    """Backtest a daily weight matrix over a price panel (no look-ahead)."""
    weights = weights.reindex(prices.index).fillna(0).shift(1).fillna(0)
    asset_returns = prices.pct_change().fillna(0)
    strategy_returns = (weights * asset_returns).sum(axis=1)
    equity = initial_capital * (1 + strategy_returns).cumprod()
    turnover = weights.diff().abs().sum(axis=1).fillna(0)

    return {
        "position": (weights > 0).sum(axis=1),
        "weights": weights,
        "returns": strategy_returns,
        "equity": equity,
        "pnl": equity.diff().fillna(0),
        "cumulative_pnl": equity - initial_capital,
        "drawdown": equity / equity.cummax() - 1,
        "trades": (weights.diff().abs() > 1e-9).sum(axis=1).astype(float),
        "turnover": turnover,
        "initial_capital": initial_capital,
    }


# -------------------------------------------------------------------- metrics
def calculate_metrics(result: dict) -> dict:
    returns = result["returns"]
    equity = result["equity"]
    initial = result.get("initial_capital", 100_000)
    total_return = equity.iloc[-1] / initial - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = (equity.iloc[-1] / initial) ** (1 / years) - 1
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / volatility if volatility else np.nan
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (returns.mean() * TRADING_DAYS) / downside if downside else np.nan
    invested = returns[result["position"] > 0]
    win_rate = (invested > 0).mean() if not invested.empty else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": float(result["drawdown"].min()),
        "Win Rate": win_rate,
        "Trades": int(result["trades"].sum()),
    }


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for col in ["Total Return", "CAGR", "Volatility", "Maximum Drawdown", "Win Rate"]:
        formatted[col] = formatted[col].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
    for col in ["Sharpe Ratio", "Sortino Ratio"]:
        formatted[col] = formatted[col].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
    formatted["Trades"] = formatted["Trades"].astype(int)
    return formatted


__all__ = ["run_backtest", "run_portfolio_backtest", "momentum_weight_matrix",
           "calculate_metrics", "format_metrics"]
