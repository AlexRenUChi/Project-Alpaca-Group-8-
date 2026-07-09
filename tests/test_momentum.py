from strategy.momentum import MomentumStrategy


def test_targets_are_long_only_and_bounded(universe_history):
    strat = MomentumStrategy(lookback_days=20, trend_filter_days=50, top_n=2)
    targets = strat.generate_targets(universe_history)

    assert set(targets) == set(universe_history)
    assert all(w >= 0 for w in targets.values())          # long-only
    assert sum(targets.values()) <= 1.0 + 1e-9            # no leverage


def test_uptrend_selected_downtrend_not(universe_history):
    strat = MomentumStrategy(lookback_days=20, trend_filter_days=50, top_n=2)
    targets = strat.generate_targets(universe_history)
    assert targets["AAA"] > 0        # strong uptrend must be held
    assert targets["BBB"] == 0.0     # downtrend fails both filters


def test_insufficient_history_is_flat(universe_history):
    short = {s: df.iloc[:30] for s, df in universe_history.items()}
    strat = MomentumStrategy()
    targets = strat.generate_targets(short)
    assert all(w == 0 for w in targets.values())
