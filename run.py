"""Command-line entry point.

    python run.py --mode paper            # continuous paper-trading loop
    python run.py --mode paper --once     # single cycle (good for the video)
    python run.py --mode backtest         # historical backtest of the config

PAPER TRADING ONLY — no real money can be traded from this codebase.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import load_settings, setup_logging


def run_backtest_mode(settings) -> None:
    from backtest.engine import (calculate_metrics, format_metrics,
                                 momentum_weight_matrix, run_portfolio_backtest)
    from data.connector import AlpacaDataConnector

    log = setup_logging("backtest")
    cfg = settings
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(cfg.backtest.years * 365.25))

    log.info("Fetching %d years of daily bars for %s", cfg.backtest.years, cfg.universe)
    connector = AlpacaDataConnector(feed=cfg.data.feed)
    closes = {}
    for symbol in cfg.universe:
        bars = connector.get_historical(symbol, start=start, end=end, timeframe="1Day")
        if not bars.empty:
            closes[symbol] = bars["close"]
    prices = pd.DataFrame(closes).dropna(how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    m = cfg.strategy.momentum
    weights = momentum_weight_matrix(prices, m.lookback_days,
                                     m.trend_filter_days, m.top_n)
    live_risk = dict(
        max_gross_notional=cfg.risk.max_gross_notional,
        max_position_notional=cfg.risk.max_position_notional,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        take_profit_pct=cfg.risk.take_profit_pct,
        cooldown_days=max(1, (cfg.risk.cooldown_hours + 23) // 24),
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
    )
    result = run_portfolio_backtest(
        prices, weights, cfg.backtest.initial_capital, **live_risk)

    # Equal-weight buy & hold benchmark on the same universe
    bench_weights = pd.DataFrame(1 / prices.shape[1], index=prices.index,
                                 columns=prices.columns)
    bench = run_portfolio_backtest(
        prices, bench_weights, cfg.backtest.initial_capital,
        max_gross_notional=cfg.risk.max_gross_notional,
        max_position_notional=cfg.risk.max_position_notional,
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
    )

    table = pd.DataFrame({
        "Momentum": calculate_metrics(result),
        "Exposure-Matched Equal Weight": calculate_metrics(bench),
    }).T
    print("\n" + format_metrics(table).to_string())
    print(f"\nFinal equity: ${result['equity'].iloc[-1]:,.0f} "
          f"(started ${cfg.backtest.initial_capital:,.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic trading system (Alpaca paper only).")
    parser.add_argument("--mode", choices=["paper", "backtest"], default="paper")
    parser.add_argument("--once", action="store_true",
                        help="paper mode: run one cycle and exit")
    parser.add_argument("--config", default=None, help="path to an alternate config.yaml")
    args = parser.parse_args()

    settings = load_settings(args.config) if args.config else load_settings()

    if args.mode == "backtest":
        run_backtest_mode(settings)
    else:
        from execution.engine import TradingEngine
        TradingEngine(settings).run(once=args.once)


if __name__ == "__main__":
    main()
