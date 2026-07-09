"""Strategy factory."""

from __future__ import annotations

from config.settings import StrategyConfig
from strategy.base import Strategy


def make_strategy(cfg: StrategyConfig) -> Strategy:
    if cfg.name == "ml":
        from strategy.ml_strategy import MLStrategy
        return MLStrategy(cfg.ml.model_type, cfg.ml.prob_threshold,
                          cfg.ml.variance_threshold)
    from strategy.momentum import MomentumStrategy
    return MomentumStrategy(cfg.momentum.lookback_days,
                            cfg.momentum.trend_filter_days, cfg.momentum.top_n)


__all__ = ["make_strategy", "Strategy"]
