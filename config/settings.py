"""Typed configuration loader.

All tunables (tickers, strategy parameters, risk limits) live in
``config/config.yaml``. Secrets live in ``.env`` and are read from the
environment — never from YAML and never hard-coded.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
STORAGE_DIR = PROJECT_ROOT / "storage"
LOGS_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class DataConfig:
    feed: str = "iex"
    timeframe: str = "1Day"
    history_days: int = 200
    poll_interval_sec: int = 60


@dataclass
class MomentumConfig:
    lookback_days: int = 20
    trend_filter_days: int = 50
    top_n: int = 3


@dataclass
class MLConfig:
    model_type: str = "Random Forest"
    prob_threshold: float = 0.60
    variance_threshold: float = 0.80


@dataclass
class StrategyConfig:
    name: str = "momentum"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    ml: MLConfig = field(default_factory=MLConfig)


@dataclass
class RiskConfig:
    max_position_notional: float = 5_000
    max_gross_notional: float = 15_000
    max_order_notional: float = 5_000
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_positions: int = 5
    cooldown_hours: int = 24


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000
    years: int = 5
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0


@dataclass
class Settings:
    universe: list[str] = field(default_factory=lambda: ["SPY"])
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def load_settings(path: Path | str = CONFIG_PATH) -> Settings:
    """Load config.yaml into a typed Settings object (missing keys use defaults)."""
    raw = {}
    path = Path(path)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    strat_raw = raw.get("strategy", {})
    return Settings(
        universe=[s.upper() for s in raw.get("universe", ["SPY"])],
        data=DataConfig(**raw.get("data", {})),
        strategy=StrategyConfig(
            name=strat_raw.get("name", "momentum"),
            momentum=MomentumConfig(**strat_raw.get("momentum", {})),
            ml=MLConfig(**strat_raw.get("ml", {})),
        ),
        risk=RiskConfig(**raw.get("risk", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )


def save_risk_limits(risk: RiskConfig, path: Path | str = CONFIG_PATH) -> None:
    """Persist risk limits edited from the UI back into config.yaml."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["risk"] = {
        "max_position_notional": risk.max_position_notional,
        "max_gross_notional": risk.max_gross_notional,
        "max_order_notional": risk.max_order_notional,
        "stop_loss_pct": risk.stop_loss_pct,
        "take_profit_pct": risk.take_profit_pct,
        "max_positions": risk.max_positions,
        "cooldown_hours": risk.cooldown_hours,
    }
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False))


def alpaca_keys() -> tuple[str, str]:
    """Read Alpaca PAPER keys from the environment. Raises if missing."""
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca keys. Copy .env.example to .env and set "
            "ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys only)."
        )
    return key, secret


def setup_logging(name: str, filename: str = "system.log") -> logging.Logger:
    """Console + rotating file logging under logs/."""
    from logging.handlers import RotatingFileHandler

    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        RotatingFileHandler(LOGS_DIR / filename, maxBytes=2_000_000, backupCount=3),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
