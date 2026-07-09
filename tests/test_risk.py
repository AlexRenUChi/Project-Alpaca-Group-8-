from config.settings import RiskConfig
from risk.manager import OrderIntent, PositionView, RiskManager


def make_position(symbol="AAPL", value=4000.0, entry=100.0, price=100.0):
    return PositionView(symbol=symbol, qty=value / price, market_value=value,
                        avg_entry_price=entry, current_price=price,
                        unrealized_pl=(price - entry) * value / price)


def manager():
    return RiskManager(RiskConfig(
        max_position_notional=5000, max_gross_notional=10000,
        max_order_notional=3000, stop_loss_pct=0.05, take_profit_pct=0.10,
        max_positions=2))


def test_order_within_limits_approved():
    ok, reason = manager().check_order(
        OrderIntent("MSFT", "buy", 2000), {}, equity=100_000)
    assert ok, reason


def test_order_exceeding_order_cap_rejected():
    ok, reason = manager().check_order(
        OrderIntent("MSFT", "buy", 3500), {}, equity=100_000)
    assert not ok and "max order" in reason


def test_position_cap_rejected():
    positions = {"AAPL": make_position(value=4000)}
    ok, reason = manager().check_order(
        OrderIntent("AAPL", "buy", 2000), positions, equity=100_000)
    assert not ok and "per-asset cap" in reason


def test_gross_cap_rejected():
    positions = {"AAPL": make_position("AAPL", 5000),
                 "MSFT": make_position("MSFT", 4000)}
    ok, reason = manager().check_order(
        OrderIntent("NVDA", "buy", 2000), positions, equity=100_000)
    assert not ok and "gross" in reason


def test_max_positions_rejected():
    positions = {"AAPL": make_position("AAPL", 1000),
                 "MSFT": make_position("MSFT", 1000)}
    ok, reason = manager().check_order(
        OrderIntent("NVDA", "buy", 1000), positions, equity=100_000)
    assert not ok and "max positions" in reason


def test_sells_always_allowed_even_over_limits():
    positions = {"AAPL": make_position("AAPL", 9999)}
    ok, _ = manager().check_order(OrderIntent("AAPL", "sell", 0), positions, 1)
    assert ok


def test_stop_loss_and_take_profit_exits():
    positions = {
        "LOSER": make_position("LOSER", 1000, entry=100, price=94),   # -6%
        "WINNER": make_position("WINNER", 1000, entry=100, price=111),  # +11%
        "HOLD": make_position("HOLD", 1000, entry=100, price=102),    # +2%
    }
    exits = manager().stop_exits(positions)
    symbols = {e.symbol: e.reason for e in exits}
    assert "LOSER" in symbols and "STOP-LOSS" in symbols["LOSER"]
    assert "WINNER" in symbols and "TAKE-PROFIT" in symbols["WINNER"]
    assert "HOLD" not in symbols
