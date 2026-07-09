"""Strategy interface.

A strategy consumes per-symbol OHLCV history and produces target portfolio
weights. It never talks to Alpaca directly — the execution engine turns
targets into orders and the risk module decides whether they are allowed.
This separation lets the same strategy run in backtest and paper mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all systematic strategies."""

    #: minimum number of bars per symbol needed before signals are valid
    min_history: int = 60

    @abstractmethod
    def generate_targets(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Map {symbol: OHLCV DataFrame} -> {symbol: target weight in [0, 1]}.

        Weights are fractions of allowed gross exposure. Symbols omitted or
        given weight 0 should be flat. Sum of weights must be <= 1 (long-only,
        no leverage).
        """

    def describe(self) -> str:
        return self.__class__.__name__
