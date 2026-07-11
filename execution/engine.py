"""Trading engine: the continuous loop that ties everything together.

Each cycle (paper mode):
  1. DATA     fetch latest quotes for the universe -> log to SQLite
  2. DATA     refresh rolling bar history per symbol
  3. ORDERS   reconcile non-terminal orders from prior cycles/restarts
  4. RISK     stop-loss/take-profit -> exit + persistent cooldown
  5. SIGNALS  strategy turns history into target weights -> log signals
  6. EXECUTE  diff targets vs positions/open orders -> risk check -> orders
  7. MONITOR  snapshot equity/positions to SQLite + state file for the UI

PAPER TRADING ONLY — the broker is hard-wired to paper=True.
"""

from __future__ import annotations

import logging
import time

from config.settings import Settings, load_settings, setup_logging
from data.connector import AlpacaDataConnector
from data.store import TradeStore
from execution import state as engine_state
from execution.broker import OrderEvent, OrderResult, PaperBroker
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
    def _record_result(self, result: OrderResult, reason: str = "",
                       entry_price: float | None = None) -> None:
        """Persist every newly observed state and actual fill economics."""
        events = result.events or [OrderEvent(
            result.status, result.qty or 0.0, result.filled_avg_price)]
        for event in events:
            realized = None
            if (result.side == "sell" and event.filled_qty > 0
                    and entry_price is not None
                    and event.filled_avg_price is not None):
                realized = ((event.filled_avg_price - entry_price)
                            * event.filled_qty)
            self.store.log_order_event(
                result.order_id, result.symbol, result.side, event.status,
                requested_notional=result.notional,
                filled_qty=event.filled_qty,
                filled_avg_price=event.filled_avg_price,
                entry_price=entry_price,
                reason=(f"{reason}: {result.error}" if reason and result.error
                        else reason or result.error),
                realized_pnl=realized,
            )

    def _reconcile_orders(self) -> None:
        """Recover and advance all non-terminal orders before new decisions."""
        positions = self.broker.positions()
        try:
            for result in self.broker.open_orders():
                context = self.store.order_context(result.order_id)
                held = positions.get(result.symbol)
                entry = context.get("entry_price") or (
                    held.avg_entry_price if held and result.side == "sell" else None)
                self._record_result(result, context.get("reason", "recovered open order"),
                                    entry)
        except Exception as err:
            log.warning("broker | open-order recovery failed: %s", err)

        for order_id in self.store.pending_order_ids():
            try:
                context = self.store.order_context(order_id)
                result = self.broker.get_order(order_id)
                self._record_result(result, context.get("reason", "reconciled"),
                                    context.get("entry_price"))
            except Exception as err:
                log.warning("broker | reconcile %s failed: %s", order_id[:8], err)

    def _cancel_order(self, order: OrderResult, reason: str) -> None:
        try:
            context = self.store.order_context(order.order_id)
            result = self.broker.cancel_order(order.order_id)
            self._record_result(result, reason,
                                context.get("entry_price") if context else None)
            log.info("broker | canceled stale %s order for %s",
                     order.side, order.symbol)
        except Exception as err:
            log.warning("broker | cancel %s failed: %s", order.order_id[:8], err)

    def _execute(self, intent: OrderIntent) -> OrderResult | None:
        """Run one intent through risk, then to the broker; log everything."""
        open_orders = self.broker.open_orders()
        same_symbol = [o for o in open_orders if o.symbol == intent.symbol]
        if intent.side == "sell":
            # A queued buy must never refill a position after a risk/signal exit.
            for order in [o for o in same_symbol if o.side == "buy"]:
                self._cancel_order(order, f"canceled before {intent.reason}")
            open_orders = self.broker.open_orders()
            if any(o.symbol == intent.symbol and o.side == "buy" for o in open_orders):
                log.error("broker | defer SELL %s; queued buy could not be canceled",
                          intent.symbol)
                return None
            if any(o.symbol == intent.symbol and o.side == "sell" for o in open_orders):
                log.info("broker | skip duplicate SELL %s; open sell exists", intent.symbol)
                return None
        elif same_symbol:
            log.info("broker | skip duplicate BUY %s; open order exists", intent.symbol)
            return None

        positions = self.broker.positions()
        account = self.broker.account()
        pending_buys: dict[str, float] = {}
        for order in open_orders:
            if order.side == "buy":
                pending_buys[order.symbol] = (
                    pending_buys.get(order.symbol, 0.0) + float(order.notional or 0.0))
        approved, reason = self.risk.check_order(
            intent, positions, float(account.equity), pending_buys,
            cash=float(account.cash),
        )
        if not approved:
            log.warning("risk | REJECTED %s %s: %s", intent.side, intent.symbol, reason)
            self.store.log_order_event("", intent.symbol, intent.side, "risk_rejected",
                                       requested_notional=intent.notional, reason=reason)
            return None

        if intent.side == "buy":
            result = self.broker.buy_notional(intent.symbol, intent.notional)
            entry_price = None
        else:
            held = positions.get(intent.symbol)
            if held is None:
                log.info("broker | skip SELL %s; no open position", intent.symbol)
                return None
            entry_price = held.avg_entry_price
            result = self.broker.close_position(intent.symbol)
        self._record_result(result, intent.reason, entry_price)
        return result

    def _rebalance(self, targets: dict[str, float]) -> None:
        """Turn target weights into buy/close intents and execute them."""
        positions = self.broker.positions()
        budget = self.settings.risk.max_gross_notional

        self._cancel_stale_entries(targets)

        # exits first (frees budget and reduces risk)
        for symbol, pos in positions.items():
            if targets.get(symbol, 0.0) <= 0 and symbol in self.settings.universe:
                self._execute(OrderIntent(symbol, "sell", 0, "signal exit"))

        positions = self.broker.positions()
        pending: dict[str, float] = {}
        for order in self.broker.open_orders():
            if order.side == "buy":
                pending[order.symbol] = (pending.get(order.symbol, 0.0)
                                         + float(order.notional or 0.0))
        for symbol, weight in sorted(targets.items(), key=lambda kv: -kv[1]):
            if weight <= 0:
                continue
            target_notional = min(weight * budget,
                                  self.settings.risk.max_position_notional)
            held = positions.get(symbol)
            gap = target_notional - (held.market_value if held else 0.0) \
                - pending.get(symbol, 0.0)
            if gap > REBALANCE_BAND:
                self._execute(OrderIntent(symbol, "buy", gap, "signal entry"))

    def _cancel_stale_entries(self, targets: dict[str, float]) -> None:
        """Cancel queued buys that are flat or oversized under current targets."""
        budget = self.settings.risk.max_gross_notional
        positions = self.broker.positions()
        by_symbol: dict[str, list[OrderResult]] = {}
        for order in self.broker.open_orders():
            if order.side == "buy" and order.symbol in self.settings.universe:
                by_symbol.setdefault(order.symbol, []).append(order)

        for symbol, orders in by_symbol.items():
            target = min(targets.get(symbol, 0.0) * budget,
                         self.settings.risk.max_position_notional)
            held = positions.get(symbol)
            projected = (held.market_value if held else 0.0) + sum(
                float(order.notional or 0.0) for order in orders)
            if target <= 0 or projected > target + REBALANCE_BAND:
                reason = ("target changed to flat" if target <= 0
                          else "queued buys exceed current target")
                for order in orders:
                    self._cancel_order(order, reason)

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
                "active_cooldowns": self.store.active_cooldowns(),
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
        self._reconcile_orders()

        # 3. stop-loss / take-profit before anything else
        active_cooldowns = self.store.active_cooldowns()
        risk_exits = self.risk.stop_exits(self.broker.positions())
        for exit_intent in risk_exits:
            self.store.log_signal(exit_intent.symbol, "EXIT", exit_intent.reason)
            self.store.set_cooldown(exit_intent.symbol,
                                    cfg.risk.cooldown_hours, exit_intent.reason)
            active_cooldowns[exit_intent.symbol] = exit_intent.reason
            if market_open:
                self._execute(exit_intent)
            else:
                log.warning("risk | %s deferred until market opens", exit_intent.symbol)

        # 4. strategy signals
        targets = self.strategy.generate_targets(self.history)
        for symbol in active_cooldowns:
            if symbol in targets:
                targets[symbol] = 0.0
        longs = [s for s, w in targets.items() if w > 0]
        log.info("signal | %s -> long %s", self.strategy.describe(), longs or "nothing")
        for symbol, weight in targets.items():
            if weight > 0:
                self.store.log_signal(symbol, "LONG", f"target weight {weight:.2f}")

        # Cancel obsolete queued entries even while closed, before they can fill
        # at the next open.
        self._cancel_stale_entries(targets)

        # 5. Never queue market orders while closed. This avoids repeated DAY
        #    orders accumulating before the next session.
        if market_open:
            self._rebalance(targets)
        else:
            log.info("engine | market closed; signals recorded, no orders submitted")

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
