"""Alpaca paper-trading broker wrapper (execution module).

The trading client is HARD-WIRED to paper=True — there is no code path in
this repository that can trade real money. Handles order submission, order
state polling (submitted → filled / partially_filled / canceled / rejected),
and network-error retries.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config.settings import alpaca_keys
from risk.manager import PositionView

log = logging.getLogger(__name__)

TERMINAL_STATES = {
    "filled", "canceled", "expired", "rejected", "replaced",
    "done_for_day", "stopped", "suspended", "calculated",
}


@dataclass(frozen=True)
class OrderEvent:
    """One observed state in an Alpaca order's lifecycle."""
    status: str
    filled_qty: float = 0.0
    filled_avg_price: float | None = None


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    status: str
    notional: float | None = None
    qty: float | None = None
    filled_avg_price: float | None = None
    error: str = ""
    events: list[OrderEvent] = field(default_factory=list)


def _retry(fn, attempts: int = 3, base_wait: float = 1.0):
    """Retry transient network/API errors with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as err:  # alpaca-py raises APIError subclasses
            last = err
            msg = str(err).lower()
            transient = any(t in msg for t in (
                "timeout", "connection", "temporarily", "429",
                "500", "502", "503", "504",
            ))
            if not transient or i == attempts - 1:
                raise
            wait = base_wait * 2 ** i
            log.warning("broker | transient error (%s), retrying in %.0fs", err, wait)
            time.sleep(wait)
    raise last  # pragma: no cover


class PaperBroker:
    """All order routing and account access goes through this class."""

    def __init__(self):
        key, secret = alpaca_keys()
        # paper=True is deliberate and must never be changed.
        self.client = TradingClient(key, secret, paper=True)

    # ------------------------------------------------------------- account
    def account(self):
        return _retry(self.client.get_account)

    def equity(self) -> float:
        return float(self.account().equity)

    def positions(self) -> dict[str, PositionView]:
        out: dict[str, PositionView] = {}
        for p in _retry(self.client.get_all_positions):
            out[p.symbol] = PositionView(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
            )
        return out

    def market_open(self) -> bool:
        try:
            return bool(_retry(self.client.get_clock).is_open)
        except Exception as err:
            log.error("broker | clock check failed: %s", err)
            return False

    # -------------------------------------------------------------- orders
    def _submit_market(self, request: MarketOrderRequest, client_order_id: str):
        """Submit idempotently and recover a timeout by client order ID."""
        try:
            return _retry(lambda: self.client.submit_order(request))
        except Exception as submit_err:
            # A timeout can occur after Alpaca accepted the request. Retrying
            # with the same client ID is safe; if Alpaca reports a duplicate,
            # recover the original instead of creating another order.
            try:
                return _retry(
                    lambda: self.client.get_order_by_client_id(client_order_id),
                    attempts=2,
                )
            except Exception:
                raise submit_err

    def buy_notional(self, symbol: str, notional: float) -> OrderResult:
        """Submit a notional market BUY and poll it to a terminal state."""
        client_order_id = f"g8-{symbol.lower()}-buy-{uuid.uuid4().hex[:20]}"
        try:
            request = MarketOrderRequest(
                symbol=symbol, notional=round(float(notional), 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id)
            order = self._submit_market(request, client_order_id)
        except Exception as err:
            log.error("broker | BUY %s rejected: %s", symbol, err)
            event = OrderEvent("rejected")
            return OrderResult("", symbol, "buy", "rejected", notional,
                               error=str(err), events=[event])
        return self._poll(order, "buy", notional=notional)

    def close_position(self, symbol: str) -> OrderResult:
        """Close an entire position with a market SELL."""
        client_order_id = f"g8-{symbol.lower()}-sell-{uuid.uuid4().hex[:20]}"
        try:
            position = _retry(lambda: self.client.get_open_position(symbol))
            available = getattr(position, "qty_available", None)
            qty = float(position.qty if available is None else available)
            if qty <= 0:
                raise RuntimeError("position quantity is held by another open order")
            request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id)
            order = self._submit_market(request, client_order_id)
        except Exception as err:
            log.error("broker | CLOSE %s failed: %s", symbol, err)
            event = OrderEvent("rejected")
            return OrderResult("", symbol, "sell", "rejected",
                               error=str(err), events=[event])
        return self._poll(order, "sell")

    @staticmethod
    def _status(order) -> str:
        return str(order.status.value if hasattr(order.status, "value") else order.status)

    @staticmethod
    def _side(order) -> str:
        side = order.side.value if hasattr(order.side, "value") else order.side
        return str(side)

    @classmethod
    def _event(cls, order) -> OrderEvent:
        qty = float(order.filled_qty) if getattr(order, "filled_qty", None) else 0.0
        avg = (float(order.filled_avg_price)
               if getattr(order, "filled_avg_price", None) else None)
        return OrderEvent(cls._status(order), qty, avg)

    @classmethod
    def _result_from_order(cls, order, notional: float | None = None) -> OrderResult:
        event = cls._event(order)
        requested_notional = notional
        if requested_notional is None and getattr(order, "notional", None):
            requested_notional = float(order.notional)
        return OrderResult(
            order_id=str(order.id), symbol=str(order.symbol), side=cls._side(order),
            status=event.status, notional=requested_notional,
            qty=event.filled_qty or None, filled_avg_price=event.filled_avg_price,
            events=[event],
        )

    def open_orders(self) -> list[OrderResult]:
        """Return every currently open order for idempotency and risk checks."""
        orders = _retry(lambda: self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)))
        return [self._result_from_order(order) for order in orders]

    def get_order(self, order_id: str) -> OrderResult:
        """Fetch one order regardless of whether it is still open."""
        order = _retry(lambda: self.client.get_order_by_id(order_id))
        return self._result_from_order(order)

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order and return its latest observable state."""
        _retry(lambda: self.client.cancel_order_by_id(order_id))
        deadline = time.time() + 10
        result = self.get_order(order_id)
        while result.status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(0.5)
            result = self.get_order(order_id)
        return result

    def _poll(self, order, side: str, notional: float | None = None,
              timeout: float = 30.0) -> OrderResult:
        """Poll an order until it reaches a terminal state (or timeout).

        Outside market hours a DAY market order stays 'accepted'/'new' in the
        queue — that is reported as its current (non-terminal) status.
        """
        order_id = str(order.id)
        status = self._status(order)
        events = [self._event(order)]
        deadline = time.time() + timeout
        while status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(1.5)
            try:
                order = _retry(lambda: self.client.get_order_by_id(order_id))
                status = self._status(order)
                event = self._event(order)
                previous = events[-1]
                if (event.status, event.filled_qty, event.filled_avg_price) != (
                        previous.status, previous.filled_qty, previous.filled_avg_price):
                    events.append(event)
            except Exception as err:
                log.warning("broker | order poll failed: %s", err)
                break
        final_event = self._event(order)
        if (final_event.status, final_event.filled_qty, final_event.filled_avg_price) != (
                events[-1].status, events[-1].filled_qty, events[-1].filled_avg_price):
            events.append(final_event)
        qty = final_event.filled_qty or None
        if status == "partially_filled":
            log.warning("broker | %s %s partially filled (%s)", side, order.symbol, qty)
        log.info("broker | order %s %s %s -> %s", side.upper(), order.symbol,
                 order_id[:8], status)
        return OrderResult(order_id, order.symbol, side, status, notional, qty,
                           final_event.filled_avg_price, events=events)


__all__ = ["PaperBroker", "OrderResult", "OrderEvent", "TERMINAL_STATES"]
