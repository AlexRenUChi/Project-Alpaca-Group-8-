import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_bars(n: int = 300, drift: float = 0.0005, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV bars — deterministic, no network needed."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))
    high = close * (1 + 0.005 * rng.random(n))
    low = close * (1 - 0.005 * rng.random(n))
    open_ = low + (high - low) * rng.random(n)
    volume = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def universe_history() -> dict[str, pd.DataFrame]:
    # AAA trends up strongly, BBB drifts down, CCC is flat-ish
    return {
        "AAA": make_bars(drift=0.003, seed=1),
        "BBB": make_bars(drift=-0.003, seed=2),
        "CCC": make_bars(drift=0.0, seed=3),
    }
