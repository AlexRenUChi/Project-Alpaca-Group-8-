"""Cross-sectional momentum strategy (the live trading strategy).

Intuition
---------
Stocks that outperformed their peers over the past ~1 month tend to keep
outperforming over short horizons (the momentum anomaly, Jegadeesh & Titman
1993). We combine this with a trend filter — only hold names trading above
their 50-day moving average — so we avoid buying "winners" that have already
rolled over. Long-only, equal-weight, no leverage.

Rules
-----
1. For each symbol compute the ``lookback_days`` total return.
2. Discard symbols whose close is below the ``trend_filter_days`` SMA or
   whose momentum is negative.
3. Rank the survivors and hold the top ``top_n`` names, equal-weighted.
Everything else is flat. Rebalances every engine cycle (daily bars mean
targets only change once a day).
"""

from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, lookback_days: int = 20, trend_filter_days: int = 50,
                 top_n: int = 3):
        self.lookback_days = lookback_days
        self.trend_filter_days = trend_filter_days
        self.top_n = max(1, top_n)
        self.min_history = max(lookback_days, trend_filter_days) + 5

    def score_symbol(self, bars: pd.DataFrame) -> float | None:
        """Momentum score for one symbol, or None if it fails the filters."""
        close = bars["close"].dropna()
        if len(close) < self.min_history:
            return None
        momentum = close.iloc[-1] / close.iloc[-1 - self.lookback_days] - 1
        sma = close.rolling(self.trend_filter_days).mean().iloc[-1]
        if close.iloc[-1] <= sma or momentum <= 0:
            return None
        return float(momentum)

    def generate_targets(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for symbol, bars in history.items():
            score = self.score_symbol(bars)
            if score is not None:
                scores[symbol] = score

        winners = sorted(scores, key=scores.get, reverse=True)[: self.top_n]
        weight = 1.0 / self.top_n  # equal weight; unused slots stay in cash
        targets = {symbol: 0.0 for symbol in history}
        targets.update({symbol: weight for symbol in winners})
        return targets

    def describe(self) -> str:
        return (f"Momentum(lookback={self.lookback_days}d, "
                f"trend_filter=SMA{self.trend_filter_days}, top_n={self.top_n})")


__all__ = ["MomentumStrategy"]
