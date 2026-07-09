"""Alpaca paper-trading broker wrapper (execution module).

The trading client is HARD-WIRED to paper=True — there is no code path in
this repository that can trade real money. Handles order submission, order
state polling (submitted → filled / partially_filled / canceled / rejected),
and network-error retries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config.settings import alpaca_keys
from risk.manager import PositionView

log = logging.getLogger(__name__)

TERMINAL_STATES = {"filled", "canceled", "expired", "rejected"}


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    status: str
    notional: float | None = None
    qty: float | None = None
    error: str = ""


def _retry(fn, attempts: int = 3, base_wait: float = 1.0):
    """Retry transient network/API errors with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as err:  # alpaca-py raises APIError subclasses
            last = err
            msg = str(err).lower()
            transient = any(t in msg for t in
                            ("timeout", "connection", "temporarily", "429", "50"))
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
    def buy_notional(self, symbol: str, notional: float) -> OrderResult:
        """Submit a notional market BUY and poll it to a terminal state."""
        try:
            order = _retry(lambda: self.client.submit_order(MarketOrderRequest(
                symbol=symbol, notional=round(float(notional), 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY)))
        except Exception as err:
            log.error("broker | BUY %s rejected: %s", symbol, err)
            return OrderResult("", symbol, "buy", "rejected", notional, error=str(err))
        return self._poll(order, "buy", notional=notional)

    def close_position(self, symbol: str) -> OrderResult:
        """Close an entire position with a market SELL."""
        try:
            order = _retry(lambda: self.client.close_position(symbol))
        except Exception as err:
            log.error("broker | CLOSE %s failed: %s", symbol, err)
            return OrderResult("", symbol, "sell", "rejected", error=str(err))
        return self._poll(order, "sell")

    def _poll(self, order, side: str, notional: float | None = None,
              timeout: float = 30.0) -> OrderResult:
        """Poll an order until it reaches a terminal state (or timeout).

        Outside market hours a DAY market order stays 'accepted'/'new' in the
        queue — that is reported as its current (non-terminal) status.
        """
        order_id = str(order.id)
        status = str(order.status.value if hasattr(order.status, "value") else order.status)
        deadline = time.time() + timeout
        while status not in TERMINAL_STATES and time.time() < deadline:
            time.sleep(1.5)
            try:
                order = _retry(lambda: self.client.get_order_by_id(order_id))
                status = str(order.status.value if hasattr(order.status, "value")
                             else order.status)
            except Exception as err:
                log.warning("broker | order poll failed: %s", err)
                break
        qty = float(order.filled_qty) if getattr(order, "filled_qty", None) else None
        if status == "partially_filled":
            log.warning("broker | %s %s partially filled (%s)", side, order.symbol, qty)
        log.info("broker | order %s %s %s -> %s", side.upper(), order.symbol,
                 order_id[:8], status)
        return OrderResult(order_id, order.symbol, side, status, notional, qty)


__all__ = ["PaperBroker", "OrderResult"]
