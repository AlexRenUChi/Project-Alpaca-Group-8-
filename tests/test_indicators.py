import numpy as np

from strategy.indicators import (FEATURE_COLUMNS, build_feature_frame,
                                 compute_indicators)


def test_indicators_columns_and_bounds(bars):
    data = compute_indicators(bars)
    for col in ["sma_20", "sma_50", "macd", "rsi", "stoch_k", "adx", "atr", "cmf"]:
        assert col in data.columns

    rsi = data["rsi"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    stoch = data["stoch_k"].dropna()
    assert ((stoch >= 0) & (stoch <= 100)).all()


def test_feature_frame_complete(bars):
    features = build_feature_frame(bars)
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    # after warm-up there must be usable rows with no NaN/inf
    usable = features[FEATURE_COLUMNS].dropna()
    assert len(usable) > 200
    assert np.isfinite(usable.to_numpy()).all()


def test_no_lookahead_in_features(bars):
    """Truncating the last bar must not change earlier feature values."""
    full = build_feature_frame(bars)
    trunc = build_feature_frame(bars.iloc[:-1])
    common = trunc.index[-5:]
    assert np.allclose(
        full.loc[common, FEATURE_COLUMNS].fillna(0).to_numpy(),
        trunc.loc[common, FEATURE_COLUMNS].fillna(0).to_numpy(),
        atol=1e-9,
    )
