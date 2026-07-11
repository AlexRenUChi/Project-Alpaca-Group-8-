"""Risk management: pre-trade checks and position monitoring.

Every order the engine wants to send passes through ``check_order`` first.
Every cycle, ``stop_exits`` inspects open positions for stop-loss /
take-profit breaches. Limits come from config.yaml (RiskConfig) — nothing
is hard-coded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class OrderIntent:
    """What the engine wants to do, before risk approval."""
    symbol: str
    side: str            # "buy" | "sell"
    notional: float      # dollar amount (buys) or 0 for full close (sells)
    reason: str = ""


@dataclass
class PositionView:
    """Broker-agnostic snapshot of one open position."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price <= 0:
            return 0.0
        return self.current_price / self.avg_entry_price - 1


class RiskManager:
    def __init__(self, limits: RiskConfig):
        self.limits = limits

    # ---------------------------------------------------------- pre-trade
    def check_order(self, intent: OrderIntent,
                    positions: dict[str, PositionView],
                    equity: float,
                    pending_buys: dict[str, float] | None = None,
                    cash: float | None = None) -> tuple[bool, str]:
        """Approve or reject an order intent. Returns (approved, reason)."""
        limits = self.limits
        pending_buys = pending_buys or {}

        if intent.side == "sell":
            return True, "sell/close orders always allowed (they reduce risk)"

        if intent.notional <= 0:
            return False, "non-positive order notional"

        if intent.notional > limits.max_order_notional:
            return False, (f"order ${intent.notional:,.0f} exceeds max order "
                           f"notional ${limits.max_order_notional:,.0f}")

        held = positions.get(intent.symbol)
        held_value = held.market_value if held else 0.0
        symbol_pending = pending_buys.get(intent.symbol, 0.0)
        projected_position = held_value + symbol_pending + intent.notional
        if projected_position > limits.max_position_notional:
            return False, (f"position plus open buys would reach ${projected_position:,.0f}, "
                           f"above per-asset cap ${limits.max_position_notional:,.0f}")

        gross = sum(abs(p.market_value) for p in positions.values())
        pending_total = sum(pending_buys.values())
        projected_gross = gross + pending_total + intent.notional
        gross_cap = min(limits.max_gross_notional, equity)
        if projected_gross > gross_cap:
            return False, (f"gross exposure including open buys would reach "
                           f"${projected_gross:,.0f}, above no-leverage cap "
                           f"${gross_cap:,.0f}")

        if held is None and len(positions) >= limits.max_positions:
            return False, f"already at max positions ({limits.max_positions})"

        if cash is not None and pending_total + intent.notional > max(cash, 0.0):
            return False, "order plus open buys exceeds available cash (no leverage allowed)"

        return True, "ok"

    # ------------------------------------------------------------ monitors
    def stop_exits(self, positions: dict[str, PositionView]) -> list[OrderIntent]:
        """Positions breaching stop-loss / take-profit -> close intents."""
        exits: list[OrderIntent] = []
        for p in positions.values():
            if p.pnl_pct <= -self.limits.stop_loss_pct:
                exits.append(OrderIntent(p.symbol, "sell", 0,
                                         f"STOP-LOSS {p.pnl_pct:.2%}"))
            elif p.pnl_pct >= self.limits.take_profit_pct:
                exits.append(OrderIntent(p.symbol, "sell", 0,
                                         f"TAKE-PROFIT {p.pnl_pct:.2%}"))
        for e in exits:
            log.warning("risk | %s triggered for %s", e.reason, e.symbol)
        return exits


__all__ = ["RiskManager", "OrderIntent", "PositionView"]
