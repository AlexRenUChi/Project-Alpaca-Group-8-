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


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 100_000,
    *,
    max_gross_notional: float | None = None,
    max_position_notional: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    cooldown_days: int = 1,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    rebalance_band: float = 250.0,
) -> dict:
    """Backtest daily targets with the same notional/risk rules as paper mode.

    Target rows are decided at close *t* and become effective for return *t+1*.
    When notional caps are supplied, input weights are fractions of the allowed
    gross budget (matching ``TradingEngine._rebalance``), not fractions of the
    entire account. Stops are close-to-close approximations and trigger a
    configurable trading-day re-entry cooldown. Costs apply to dollar turnover.
    """
    prices = prices.sort_index().copy()
    raw_targets = weights.reindex(index=prices.index, columns=prices.columns).fillna(0)
    asset_returns = prices.pct_change(fill_method=None).fillna(0)

    # Preserve the small, fast vectorised path for callers that do not request
    # live risk parity (including the single-purpose unit tests).
    use_live_rules = any(v is not None for v in (
        max_gross_notional, max_position_notional, stop_loss_pct, take_profit_pct,
    )) or transaction_cost_bps or slippage_bps
    if not use_live_rules:
        executed_weights = raw_targets.shift(1).fillna(0)
        strategy_returns = (executed_weights * asset_returns).sum(axis=1)
        equity = initial_capital * (1 + strategy_returns).cumprod()
        turnover = executed_weights.diff().abs().sum(axis=1).fillna(0)
        trades = (executed_weights.diff().abs() > 1e-9).sum(axis=1).astype(float)
    else:
        current = pd.Series(0.0, index=prices.columns)
        entry_prices: dict[str, float] = {}
        cooldown_until: dict[str, int] = {}
        equity_value = float(initial_capital)
        cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
        weight_rows: list[pd.Series] = []
        return_rows: list[float] = []
        equity_rows: list[float] = []
        turnover_rows: list[float] = []
        trade_rows: list[float] = []

        for i, date in enumerate(prices.index):
            # Holdings chosen yesterday earn today's close-to-close return.
            weight_rows.append(current.copy())
            gross_return = float((current * asset_returns.loc[date]).sum())
            equity_value *= 1 + gross_return
            denominator = max(1 + gross_return, 1e-12)
            post_weights = (current * (1 + asset_returns.loc[date]) / denominator)

            triggered: set[str] = set()
            for symbol in prices.columns:
                if current[symbol] <= 0 or symbol not in entry_prices:
                    continue
                px = prices.at[date, symbol]
                entry = entry_prices[symbol]
                if pd.isna(px) or entry <= 0:
                    continue
                pnl_pct = float(px / entry - 1)
                if ((stop_loss_pct is not None and pnl_pct <= -stop_loss_pct)
                        or (take_profit_pct is not None and pnl_pct >= take_profit_pct)):
                    triggered.add(symbol)
                    cooldown_until[symbol] = i + max(1, cooldown_days)

            target = raw_targets.loc[date].clip(lower=0).copy()
            for symbol in prices.columns:
                if symbol in triggered or i < cooldown_until.get(symbol, -1):
                    target[symbol] = 0.0

            gross_budget = (equity_value if max_gross_notional is None
                            else min(max_gross_notional, equity_value))
            target_dollars = target * gross_budget
            if max_position_notional is not None:
                target_dollars = target_dollars.clip(upper=max_position_notional)
            total_target = float(target_dollars.sum())
            if total_target > gross_budget and total_target > 0:
                target_dollars *= gross_budget / total_target
            current_dollars = post_weights * equity_value
            desired_dollars = current_dollars.copy()
            for symbol in prices.columns:
                if target_dollars[symbol] <= 0:
                    desired_dollars[symbol] = 0.0
                elif target_dollars[symbol] - current_dollars[symbol] > rebalance_band:
                    desired_dollars[symbol] = target_dollars[symbol]
            desired = desired_dollars / max(equity_value, 1e-12)

            turnover_value = float((desired - post_weights).abs().sum())
            cost_return = turnover_value * cost_rate
            equity_value *= 1 - cost_return
            net_return = (1 + gross_return) * (1 - cost_return) - 1
            return_rows.append(net_return)
            equity_rows.append(equity_value)
            turnover_rows.append(turnover_value)
            trade_rows.append(float(((desired - post_weights).abs() > 1e-9).sum()))

            for symbol in prices.columns:
                if post_weights[symbol] <= 0 < desired[symbol]:
                    px = prices.at[date, symbol]
                    if not pd.isna(px):
                        entry_prices[symbol] = float(px)
                elif desired[symbol] <= 0:
                    entry_prices.pop(symbol, None)
            current = desired

        executed_weights = pd.DataFrame(weight_rows, index=prices.index)
        strategy_returns = pd.Series(return_rows, index=prices.index)
        equity = pd.Series(equity_rows, index=prices.index)
        turnover = pd.Series(turnover_rows, index=prices.index)
        trades = pd.Series(trade_rows, index=prices.index)

    return {
        "position": (executed_weights > 0).sum(axis=1),
        "weights": executed_weights,
        "returns": strategy_returns,
        "equity": equity,
        "pnl": equity.diff().fillna(0),
        "cumulative_pnl": equity - initial_capital,
        "drawdown": equity / equity.cummax() - 1,
        "trades": trades,
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
