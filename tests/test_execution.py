from types import SimpleNamespace

from config.settings import RiskConfig, Settings
from execution.broker import OrderResult, PaperBroker
from execution.engine import TradingEngine
from risk.manager import OrderIntent, PositionView, RiskManager


class FakeBroker:
    def __init__(self, open_orders=None, positions=None, market_open=True):
        self._open_orders = open_orders or []
        self._positions = positions or {}
        self._market_open = market_open
        self.buy_calls = 0

    def open_orders(self):
        return list(self._open_orders)

    def positions(self):
        return dict(self._positions)

    def account(self):
        return SimpleNamespace(equity=100_000, cash=100_000)

    def buy_notional(self, symbol, notional):
        self.buy_calls += 1
        return OrderResult("new", symbol, "buy", "accepted", notional)

    def market_open(self):
        return self._market_open


class FakeStore:
    def __init__(self):
        self.cooldowns = {}
        self.signals = []

    def active_cooldowns(self):
        return dict(self.cooldowns)

    def set_cooldown(self, symbol, hours, reason):
        self.cooldowns[symbol] = reason

    def log_signal(self, symbol, signal, detail=""):
        self.signals.append((symbol, signal, detail))

    def log_quotes(self, quotes):
        pass


def make_engine(broker):
    engine = TradingEngine.__new__(TradingEngine)
    engine.settings = Settings(universe=["AAPL"])
    engine.settings.risk = RiskConfig()
    engine.broker = broker
    engine.store = FakeStore()
    engine.risk = RiskManager(engine.settings.risk)
    engine.cycles = 0
    return engine


def test_existing_open_order_blocks_duplicate_buy():
    open_buy = OrderResult("open-1", "AAPL", "buy", "accepted", 5000)
    broker = FakeBroker(open_orders=[open_buy])
    engine = make_engine(broker)

    result = engine._execute(OrderIntent("AAPL", "buy", 5000, "signal entry"))
    assert result is None
    assert broker.buy_calls == 0


def test_pending_buy_is_subtracted_from_rebalance_gap():
    open_buy = OrderResult("open-1", "AAPL", "buy", "accepted", 5000)
    engine = make_engine(FakeBroker(open_orders=[open_buy]))
    intents = []
    engine._execute = intents.append

    engine._rebalance({"AAPL": 1 / 3})
    assert intents == []


def test_risk_exit_cooldown_overrides_same_cycle_long_signal():
    loser = PositionView("AAPL", 10, 940, 100, 94, -60)
    broker = FakeBroker(positions={"AAPL": loser}, market_open=False)
    engine = make_engine(broker)
    engine.connector = SimpleNamespace(
        get_latest_quotes=lambda symbols: {},
        get_history_universe=lambda symbols, days, timeframe: {"AAPL": object()},
    )
    engine.strategy = SimpleNamespace(
        generate_targets=lambda history: {"AAPL": 1.0},
        describe=lambda: "test",
    )
    engine._reconcile_orders = lambda: None
    engine._snapshot = lambda targets, market_open: None

    targets = engine.run_cycle()
    assert targets["AAPL"] == 0.0
    assert engine.store.cooldowns["AAPL"].startswith("STOP-LOSS")


def test_close_uses_idempotent_market_order_with_available_quantity():
    class Client:
        def get_open_position(self, symbol):
            return SimpleNamespace(qty="10", qty_available="8")

        def submit_order(self, request):
            self.request = request
            return SimpleNamespace(
                id="sell-1", symbol="AAPL",
                side=SimpleNamespace(value="sell"),
                status=SimpleNamespace(value="filled"),
                filled_qty="8", filled_avg_price="101.5", notional=None,
            )

    broker = PaperBroker.__new__(PaperBroker)
    broker.client = Client()
    result = broker.close_position("AAPL")

    assert broker.client.request.qty == 8
    assert broker.client.request.client_order_id.startswith("g8-aapl-sell-")
    assert result.status == "filled"
    assert result.filled_avg_price == 101.5
