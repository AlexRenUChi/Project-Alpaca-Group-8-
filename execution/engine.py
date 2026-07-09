"""Trading engine: the continuous loop that ties everything together.

Each cycle (paper mode):
  1. DATA     fetch latest quotes for the universe -> log to SQLite
  2. DATA     refresh rolling bar history per symbol
  3. RISK     check open positions for stop-loss / take-profit -> exit orders
  4. SIGNALS  strategy turns history into target weights -> log signals
  5. EXECUTE  diff targets vs positions -> order intents -> risk check -> orders
  6. MONITOR  snapshot equity/positions to SQLite + state file for the UI

PAPER TRADING ONLY — the broker is hard-wired to paper=True.
"""

from __future__ import annotations

import logging
import time

from config.settings import Settings, load_settings, setup_logging
from data.connector import AlpacaDataConnector
from data.store import TradeStore
from execution import state as engine_state
from execution.broker import PaperBroker
from risk.manager import OrderIntent, RiskManager
from strategy import make_strategy

log = logging.getLogger(__name__)

#: ignore target/held differences smaller than this many dollars
REBALANCE_BAND = 250.0


class TradingEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.connector = AlpacaDataConnector(feed=self.settings.data.feed)
        self.broker = PaperBroker()
        self.store = TradeStore()
        self.risk = RiskManager(self.settings.risk)
        self.strategy = make_strategy(self.settings.strategy)
        self.history: dict = {}
        self.cycles = 0

    # ------------------------------------------------------------- helpers
    def _execute(self, intent: OrderIntent) -> None:
        """Run one intent through risk, then to the broker; log everything."""
        positions = self.broker.positions()
        approved, reason = self.risk.check_order(intent, positions,
                                                 self.broker.equity())
        if not approved:
            log.warning("risk | REJECTED %s %s: %s", intent.side, intent.symbol, reason)
            self.store.log_order(intent.symbol, intent.side, "risk_rejected",
                                 notional=intent.notional, reason=reason)
            return

        if intent.side == "buy":
            result = self.broker.buy_notional(intent.symbol, intent.notional)
            realized = None
        else:
            held = positions.get(intent.symbol)
            realized = held.unrealized_pl if held else None
            result = self.broker.close_position(intent.symbol)
        self.store.log_order(result.symbol, result.side, result.status,
                             order_id=result.order_id, notional=result.notional,
                             qty=result.qty, reason=intent.reason,
                             realized_pnl=realized if result.status == "filled" else None)

    def _rebalance(self, targets: dict[str, float]) -> None:
        """Turn target weights into buy/close intents and execute them."""
        positions = self.broker.positions()
        budget = self.settings.risk.max_gross_notional

        # exits first (frees budget and reduces risk)
        for symbol, pos in positions.items():
            if targets.get(symbol, 0.0) <= 0 and symbol in self.settings.universe:
                self._execute(OrderIntent(symbol, "sell", 0, "signal exit"))

        positions = self.broker.positions()
        for symbol, weight in sorted(targets.items(), key=lambda kv: -kv[1]):
            if weight <= 0:
                continue
            target_notional = min(weight * budget,
                                  self.settings.risk.max_position_notional)
            held = positions.get(symbol)
            gap = target_notional - (held.market_value if held else 0.0)
            if gap > REBALANCE_BAND:
                self._execute(OrderIntent(symbol, "buy", gap, "signal entry"))

    def _snapshot(self, targets: dict[str, float], market_open: bool) -> None:
        """Persist equity + UI state."""
        try:
            acct = self.broker.account()
            positions = self.broker.positions()
            gross = sum(abs(p.market_value) for p in positions.values())
            self.store.log_equity(float(acct.equity), float(acct.cash), gross)
            engine_state.write_state({
                "status": "running",
                "mode": "paper",
                "market_open": market_open,
                "strategy": self.strategy.describe(),
                "cycle": self.cycles,
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "gross_exposure": gross,
                "targets": targets,
                "positions": {s: {"qty": p.qty, "market_value": p.market_value,
                                  "unrealized_pl": p.unrealized_pl,
                                  "pnl_pct": p.pnl_pct}
                              for s, p in positions.items()},
                "universe": self.settings.universe,
            })
        except Exception as err:
            log.error("engine | snapshot failed: %s", err)

    # ---------------------------------------------------------------- loop
    def run_cycle(self) -> dict[str, float]:
        """One full data -> risk -> signal -> execution -> monitor pass."""
        cfg = self.settings
        self.cycles += 1
        log.info("engine | ===== cycle %d =====", self.cycles)

        # 1. quotes -> structured storage (the live data pipeline log)
        try:
            quotes = self.connector.get_latest_quotes(cfg.universe)
            self.store.log_quotes(quotes)
            log.info("data | logged %d quotes", len(quotes))
        except Exception as err:
            log.error("data | quote fetch failed: %s", err)

        # 2. rolling bar history
        self.history = self.connector.get_history_universe(
            cfg.universe, cfg.data.history_days, cfg.data.timeframe)

        market_open = self.broker.market_open()

        # 3. stop-loss / take-profit before anything else
        for exit_intent in self.risk.stop_exits(self.broker.positions()):
            self.store.log_signal(exit_intent.symbol, "EXIT", exit_intent.reason)
            self._execute(exit_intent)

        # 4. strategy signals
        targets = self.strategy.generate_targets(self.history)
        longs = [s for s, w in targets.items() if w > 0]
        log.info("signal | %s -> long %s", self.strategy.describe(), longs or "nothing")
        for symbol, weight in targets.items():
            if weight > 0:
                self.store.log_signal(symbol, "LONG", f"target weight {weight:.2f}")

        # 5. execution (orders queue if the market is closed; that's fine
        #    for the demo, and stop checks still ran above)
        self._rebalance(targets)

        # 6. monitoring snapshot
        self._snapshot(targets, market_open)
        return targets

    def run(self, once: bool = False) -> None:
        setup_logging("")  # root logger -> console + logs/system.log
        log.info("engine | PAPER TRADING ONLY — no real money is used")
        log.info("engine | universe=%s strategy=%s", self.settings.universe,
                 self.strategy.describe())
        engine_state.clear_stop()
        engine_state.write_pid()
        try:
            while True:
                started = time.time()
                try:
                    self.run_cycle()
                except Exception as err:
                    log.exception("engine | cycle failed: %s", err)
                    engine_state.write_state({"status": "error", "mode": "paper",
                                              "error": str(err)})
                if once:
                    break
                # sleep in 1s slices so a stop request is honoured quickly
                while time.time() - started < self.settings.data.poll_interval_sec:
                    if engine_state.stop_requested():
                        log.info("engine | stop requested — shutting down")
                        return
                    time.sleep(1)
                if engine_state.stop_requested():
                    log.info("engine | stop requested — shutting down")
                    return
        finally:
            engine_state.write_state({"status": "stopped", "mode": "paper"})
            engine_state.clear_pid()
            engine_state.clear_stop()
            log.info("engine | stopped. This was paper trading only.")


__all__ = ["TradingEngine"]
