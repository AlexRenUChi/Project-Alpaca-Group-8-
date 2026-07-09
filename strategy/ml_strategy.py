"""ML strategy adapter: wraps the PCA + classifier pipeline as a Strategy.

Per symbol: 22 engineered features -> StandardScaler -> PCA (>=80% variance)
-> classifier -> P(next-day up). Long if that probability clears the
threshold, flat otherwise. Slower than the momentum strategy (a model is
trained per symbol), so it is mainly used for backtesting/research, but it
implements the same interface and can run live.
"""

from __future__ import annotations

import logging

import pandas as pd

from strategy.base import Strategy
from strategy.indicators import build_feature_frame
from strategy.ml_pipeline import latest_feature_row, train_full_model

log = logging.getLogger(__name__)


class MLStrategy(Strategy):
    min_history = 300  # a model needs a real training window

    def __init__(self, model_type: str = "Random Forest",
                 prob_threshold: float = 0.60, variance_threshold: float = 0.80):
        self.model_type = model_type
        self.prob_threshold = prob_threshold
        self.variance_threshold = variance_threshold

    def generate_targets(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        longs: list[str] = []
        for symbol, bars in history.items():
            if len(bars) < self.min_history:
                continue
            try:
                features = build_feature_frame(bars)
                model = train_full_model(features, self.model_type,
                                         self.prob_threshold, self.variance_threshold)
                proba = float(model.predict_proba_long(latest_feature_row(features))[0])
                log.info("ml | %s P(up)=%.3f threshold=%.2f",
                         symbol, proba, self.prob_threshold)
                if proba > self.prob_threshold:
                    longs.append(symbol)
            except Exception as err:
                log.error("ml | %s failed: %s", symbol, err)

        targets = {symbol: 0.0 for symbol in history}
        if longs:
            weight = 1.0 / len(history)  # conservative: never concentrate
            targets.update({symbol: weight for symbol in longs})
        return targets

    def describe(self) -> str:
        return (f"ML({self.model_type}, P>{self.prob_threshold}, "
                f"PCA>={self.variance_threshold:.0%} var)")


__all__ = ["MLStrategy"]
