"""Technical indicators and ML feature engineering.

Pure pandas/numpy — no Streamlit or Alpaca imports, so it can be reused by the
Streamlit app, the paper-trading script, and unit tests alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature groups used by the ML pipeline (task 2 + task 3 inputs)
# ---------------------------------------------------------------------------
# Return-based features: log returns plus rolling mean / rolling std (required).
RETURN_FEATURES = [
    "log_return",
    "roll_mean_5",
    "roll_mean_10",
    "roll_std_5",
    "roll_std_10",
    "momentum_5",
    "momentum_10",
]

# Indicator-based features, normalised so they are (roughly) stationary and
# comparable across price regimes — much better PCA / ML inputs than raw price
# levels like SMA or OBV. Covers trend, momentum, volatility and volume.
INDICATOR_FEATURES = [
    "rsi",          # momentum
    "stoch_k",      # momentum
    "stoch_d",      # momentum
    "williams_r",   # momentum
    "macd",         # trend
    "macd_signal",  # trend
    "macd_hist",    # trend
    "adx",          # trend strength
    "cmf",          # volume
    "obv_z",        # volume
    "atr_pct",      # volatility
    "bb_pos",       # volatility
    "sma_ratio",    # trend
    "close_sma20",  # trend
    "ema_ratio",    # trend
]

# Full feature matrix the ML model consumes (22 features).
FEATURE_COLUMNS = RETURN_FEATURES + INDICATOR_FEATURES


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise an OHLCV frame: tz-naive sorted index, canonical columns."""
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    columns = ["open", "high", "low", "close", "volume"]
    return df[[c for c in columns if c in df.columns]].dropna()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the ten technical indicators used across the app."""
    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_range = (high_14 - low_14).replace(0, np.nan)
    data["stoch_k"] = 100 * (close - low_14) / stoch_range
    data["stoch_d"] = data["stoch_k"].rolling(3).mean()
    data["williams_r"] = -100 * (high_14 - close) / stoch_range

    sma_20 = data["sma_20"]
    std_20 = close.rolling(20).std()
    data["bb_upper"] = sma_20 + 2 * std_20
    data["bb_lower"] = sma_20 - 2 * std_20

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr"] = tr.rolling(14).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=data.index).rolling(14).sum() / atr_14
    minus_di = 100 * pd.Series(minus_dm, index=data.index).rolling(14).sum() / atr_14
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    data["adx"] = dx.rolling(14).mean()

    data["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    hl_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume
    data["cmf"] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum()

    return data


def add_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the return-based and normalised indicator features for the model.

    Expects a frame that has already been through ``compute_indicators``.
    """
    d = data.copy()
    close = d["close"]

    # Return features (task 2: log returns, rolling mean, rolling std).
    d["log_return"] = np.log(close / close.shift(1))
    d["roll_mean_5"] = d["log_return"].rolling(5).mean()
    d["roll_mean_10"] = d["log_return"].rolling(10).mean()
    d["roll_std_5"] = d["log_return"].rolling(5).std()
    d["roll_std_10"] = d["log_return"].rolling(10).std()
    d["momentum_5"] = close / close.shift(5) - 1
    d["momentum_10"] = close / close.shift(10) - 1

    # Normalised indicator features.
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["atr_pct"] = d["atr"] / close
    bb_range = (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["bb_pos"] = (close - d["bb_lower"]) / bb_range
    d["sma_ratio"] = d["sma_20"] / d["sma_50"] - 1
    d["close_sma20"] = close / d["sma_20"] - 1
    d["ema_ratio"] = d["ema_12"] / d["ema_26"] - 1

    obv = d["obv"]
    obv_std = obv.rolling(20).std().replace(0, np.nan)
    d["obv_z"] = (obv - obv.rolling(20).mean()) / obv_std

    return d.replace([np.inf, -np.inf], np.nan)


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Convenience: normalise bars, then indicators, then ML features."""
    return add_ml_features(compute_indicators(normalize_bars(raw)))
