from data.store import TradeStore


def test_order_lifecycle_is_idempotent_and_metrics_use_closed_trades(tmp_path):
    store = TradeStore(tmp_path / "test.db")
    store.log_order_event("buy-1", "AAPL", "buy", "accepted",
                          requested_notional=1000)
    store.log_order_event("buy-1", "AAPL", "buy", "filled",
                          requested_notional=1000, filled_qty=10,
                          filled_avg_price=100)
    store.log_order_event("sell-1", "AAPL", "sell", "accepted",
                          entry_price=100)
    store.log_order_event("sell-1", "AAPL", "sell", "filled",
                          filled_qty=10, filled_avg_price=110,
                          entry_price=100, realized_pnl=100)
    # Reconciliation may observe the same final state again; it must not count twice.
    store.log_order_event("sell-1", "AAPL", "sell", "filled",
                          filled_qty=10, filled_avg_price=110,
                          entry_price=100, realized_pnl=100)

    events = store.recent_orders(20)
    assert len(events) == 4
    metrics = store.performance_metrics()
    assert metrics["num_trades"] == 1
    assert metrics["hit_rate"] == 1.0


def test_pending_orders_and_persistent_cooldowns(tmp_path):
    store = TradeStore(tmp_path / "test.db")
    store.log_order_event("order-1", "MSFT", "buy", "accepted",
                          requested_notional=1000)
    assert store.pending_order_ids() == ["order-1"]
    store.log_order_event("order-1", "MSFT", "buy", "filled",
                          requested_notional=1000, filled_qty=2,
                          filled_avg_price=500)
    assert store.pending_order_ids() == []

    store.set_cooldown("MSFT", 24, "STOP-LOSS")
    assert store.active_cooldowns()["MSFT"] == "STOP-LOSS"


def test_partially_filled_then_canceled_sell_counts_actual_fill(tmp_path):
    store = TradeStore(tmp_path / "test.db")
    store.log_order_event("sell-1", "AAPL", "sell", "partially_filled",
                          filled_qty=4, filled_avg_price=95,
                          entry_price=100, realized_pnl=-20)
    store.log_order_event("sell-1", "AAPL", "sell", "canceled",
                          filled_qty=4, filled_avg_price=95,
                          entry_price=100, realized_pnl=-20)
    metrics = store.performance_metrics()
    assert metrics["num_trades"] == 1
    assert metrics["hit_rate"] == 0.0
