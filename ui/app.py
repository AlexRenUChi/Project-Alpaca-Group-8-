"""Streamlit monitoring & control dashboard.

Run from the repo root:  streamlit run ui/app.py

Shows system status, positions & P&L, recent signals and orders, and
performance metrics; lets you start/stop the paper-trading engine, run
backtests, and edit risk limits. PAPER TRADING ONLY.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import load_settings, save_risk_limits
from data.store import TradeStore
from execution import state as engine_state

st.set_page_config(page_title="Systematic Trading System", page_icon="📈",
                   layout="wide")

settings = load_settings()
store = TradeStore()

# ----------------------------------------------------------------- sidebar
st.sidebar.title("📈 Trading System")
st.sidebar.caption("Alpaca **paper trading only** — no real money.")

running = engine_state.engine_running()
state = engine_state.read_state() or {}

status_icon = "🟢" if running else "🔴"
st.sidebar.markdown(f"### {status_icon} Engine: {'RUNNING' if running else 'STOPPED'}")
st.sidebar.markdown(f"**Mode:** {state.get('mode', 'paper')}  \n"
                    f"**Strategy:** {state.get('strategy', settings.strategy.name)}  \n"
                    f"**Market:** {'open' if state.get('market_open') else 'closed/unknown'}  \n"
                    f"**Last update:** {state.get('updated_at', '—')[:19]}")
active_cooldowns = state.get("active_cooldowns") or {}
if active_cooldowns:
    st.sidebar.warning("Risk cooldown: " + ", ".join(active_cooldowns))

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶ Start", disabled=running, use_container_width=True):
    subprocess.Popen([sys.executable, str(REPO_ROOT / "run.py"), "--mode", "paper"],
                     cwd=REPO_ROOT)
    st.sidebar.success("Engine starting…")
    time.sleep(2)
    st.rerun()
if col_stop.button("⏹ Stop", disabled=not running, use_container_width=True):
    engine_state.request_stop()
    st.sidebar.warning("Stop requested — engine exits within a few seconds.")
    time.sleep(2)
    st.rerun()

if st.sidebar.button("🔄 Refresh", use_container_width=True):
    st.rerun()
auto = st.sidebar.checkbox("Auto-refresh (10s)", value=False)

st.sidebar.divider()
st.sidebar.markdown(f"**Universe:** {', '.join(settings.universe)}")

# -------------------------------------------------------------------- tabs
tab_dash, tab_backtest, tab_quotes, tab_risk = st.tabs(
    ["Dashboard", "Backtesting", "Live Quotes", "Risk & Config"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    st.subheader("Account & Positions")
    c1, c2, c3, c4 = st.columns(4)
    metrics = store.performance_metrics()
    c1.metric("Equity", f"${state.get('equity', 0):,.0f}" if state.get("equity") else "—")
    c2.metric("Cumulative P&L",
              f"${metrics['cumulative_pnl']:,.0f}" if metrics["cumulative_pnl"] is not None else "—")
    c3.metric("Max Drawdown",
              f"{metrics['max_drawdown']:.2%}" if metrics["max_drawdown"] is not None else "—")
    c4.metric("Closed Trades / Hit Rate",
              f"{metrics['num_trades']}"
              + (f" / {metrics['hit_rate']:.0%}" if metrics["hit_rate"] is not None else ""))

    positions = state.get("positions") or {}
    if positions:
        pos_df = pd.DataFrame(positions).T
        pos_df.index.name = "symbol"
        pos_df = pos_df.rename(columns={"qty": "Qty", "market_value": "Market Value",
                                        "unrealized_pl": "Unrealized P&L",
                                        "pnl_pct": "P&L %"})
        pos_df["P&L %"] = pos_df["P&L %"].map(lambda x: f"{x:.2%}")
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("No open positions (or engine not started yet).")

    eq = store.equity_curve()
    if len(eq) >= 2:
        fig = go.Figure(go.Scatter(x=eq.index, y=eq["equity"], mode="lines",
                                   name="Equity"))
        fig.update_layout(title="Account Equity", height=300,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Recent Signals")
        sig = store.recent_signals(25)
        if not sig.empty:
            st.dataframe(sig, use_container_width=True, height=280)
        else:
            st.caption("No signals logged yet.")
    with right:
        st.subheader("Recent Orders")
        orders = store.recent_orders(25)
        if not orders.empty:
            st.dataframe(orders, use_container_width=True, height=280)
        else:
            st.caption("No orders logged yet.")

# ---------------------------------------------------------------- backtest
with tab_backtest:
    st.subheader("Portfolio Backtest — Momentum vs Exposure-Matched Benchmark")
    st.caption("Uses the live notional caps, stops, cooldown, costs and slippage.")
    years = st.slider("Years of history", 1, 10, settings.backtest.years)
    if st.button("Run Momentum Backtest", type="primary"):
        from backtest.engine import (calculate_metrics, format_metrics,
                                     momentum_weight_matrix, run_portfolio_backtest)
        from data.connector import AlpacaDataConnector

        with st.spinner("Fetching data and running backtest…"):
            connector = AlpacaDataConnector(feed=settings.data.feed)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=int(years * 365.25))
            closes = {}
            for symbol in settings.universe:
                bars = connector.get_historical(symbol, start=start, end=end,
                                                timeframe="1Day")
                if not bars.empty:
                    closes[symbol] = bars["close"]
            prices = pd.DataFrame(closes).dropna(how="all")
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            m = settings.strategy.momentum
            weights = momentum_weight_matrix(prices, m.lookback_days,
                                             m.trend_filter_days, m.top_n)
            result = run_portfolio_backtest(prices, weights,
                                            settings.backtest.initial_capital,
                max_gross_notional=settings.risk.max_gross_notional,
                max_position_notional=settings.risk.max_position_notional,
                stop_loss_pct=settings.risk.stop_loss_pct,
                take_profit_pct=settings.risk.take_profit_pct,
                cooldown_days=max(1, (settings.risk.cooldown_hours + 23) // 24),
                transaction_cost_bps=settings.backtest.transaction_cost_bps,
                slippage_bps=settings.backtest.slippage_bps)
            bench_w = pd.DataFrame(1 / prices.shape[1], index=prices.index,
                                   columns=prices.columns)
            bench = run_portfolio_backtest(prices, bench_w,
                                           settings.backtest.initial_capital,
                max_gross_notional=settings.risk.max_gross_notional,
                max_position_notional=settings.risk.max_position_notional,
                transaction_cost_bps=settings.backtest.transaction_cost_bps,
                slippage_bps=settings.backtest.slippage_bps)

        table = pd.DataFrame({"Momentum": calculate_metrics(result),
                              "Exposure-Matched Equal Weight": calculate_metrics(bench)}).T
        st.dataframe(format_metrics(table), use_container_width=True)

        fig = go.Figure()
        fig.add_scatter(x=result["equity"].index, y=result["equity"], name="Momentum")
        fig.add_scatter(x=bench["equity"].index, y=bench["equity"],
                        name="Exposure-Matched Benchmark", line=dict(dash="dot"))
        fig.update_layout(title="Equity Curves", height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        dd = go.Figure(go.Scatter(x=result["drawdown"].index, y=result["drawdown"],
                                  fill="tozeroy", name="Drawdown"))
        dd.update_layout(title="Momentum Strategy Drawdown", height=250,
                         yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(dd, use_container_width=True)

    st.divider()
    st.subheader("ML Research Backtest (single ticker)")
    st.caption("22 features → PCA (≥80% var) → classifier → Long/Flat, "
               "out-of-sample vs Buy & Hold.")
    mcol1, mcol2, mcol3 = st.columns(3)
    ml_symbol = mcol1.selectbox("Ticker", settings.universe)
    from strategy.ml_pipeline import MODEL_CHOICES
    ml_model = mcol2.selectbox("Model", MODEL_CHOICES)
    ml_threshold = mcol3.slider("P(up) threshold", 0.50, 0.75,
                                settings.strategy.ml.prob_threshold, 0.01)
    if st.button("Run ML Backtest"):
        from backtest.engine import (calculate_metrics, format_metrics,
                                     run_backtest)
        from data.connector import AlpacaDataConnector
        from strategy.indicators import build_feature_frame
        from strategy.ml_pipeline import run_ml_strategy

        with st.spinner("Training model and backtesting out-of-sample…"):
            connector = AlpacaDataConnector(feed=settings.data.feed)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=int(5 * 365.25))
            bars = connector.get_historical(ml_symbol, start=start, end=end,
                                            timeframe="1Day")
            features = build_feature_frame(bars)
            ml = run_ml_strategy(features, ml_model, prob_threshold=ml_threshold)
            test_prices = features.loc[ml["test_index"]]
            result = run_backtest(test_prices, ml["signal"],
                                  settings.backtest.initial_capital)
            bench = run_backtest(test_prices,
                                 pd.Series(1, index=test_prices.index),
                                 settings.backtest.initial_capital)

        st.write(f"Out-of-sample accuracy **{ml['accuracy']:.1%}** "
                 f"(base rate {ml['base_rate']:.1%}), "
                 f"PCA kept **{ml['n_components']}** components.")
        table = pd.DataFrame({"ML Strategy": calculate_metrics(result),
                              "Buy & Hold": calculate_metrics(bench)}).T
        st.dataframe(format_metrics(table), use_container_width=True)
        fig = go.Figure()
        fig.add_scatter(x=result["equity"].index, y=result["equity"], name="ML")
        fig.add_scatter(x=bench["equity"].index, y=bench["equity"],
                        name="Buy & Hold", line=dict(dash="dot"))
        fig.update_layout(title=f"{ml_symbol} — Out-of-Sample Equity", height=350,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------- quotes
with tab_quotes:
    st.subheader("Latest Quotes")
    st.caption("Bid/ask snapshot for the universe (REST). The engine logs these "
               "to SQLite every cycle; recent stored quotes shown below.")
    if st.button("Fetch Latest Quotes"):
        from data.connector import AlpacaDataConnector
        try:
            quotes = AlpacaDataConnector(feed=settings.data.feed) \
                .get_latest_quotes(settings.universe)
            qdf = pd.DataFrame(quotes).T
            qdf.index.name = "symbol"
            st.dataframe(qdf, use_container_width=True)
        except Exception as err:
            st.error(f"Quote fetch failed: {err}")
    st.markdown("**Recently stored quotes (data pipeline log)**")
    recent = store.recent_quotes(50)
    if not recent.empty:
        st.dataframe(recent, use_container_width=True)
    else:
        st.caption("No quotes stored yet — start the engine.")

# --------------------------------------------------------------------- risk
with tab_risk:
    st.subheader("Risk Limits")
    st.caption("Edits are written to config/config.yaml. The engine reads the "
               "config at startup — restart it to apply changes.")
    risk = settings.risk
    with st.form("risk_form"):
        r1, r2, r3 = st.columns(3)
        max_pos = r1.number_input("Max position notional ($)", 100.0, 1e6,
                                  float(risk.max_position_notional), 100.0)
        max_gross = r2.number_input("Max gross notional ($)", 100.0, 1e7,
                                    float(risk.max_gross_notional), 100.0)
        max_order = r3.number_input("Max order notional ($)", 100.0, 1e6,
                                    float(risk.max_order_notional), 100.0)
        r4, r5, r6 = st.columns(3)
        stop = r4.number_input("Stop-loss %", 0.01, 0.50,
                               float(risk.stop_loss_pct), 0.01, format="%.2f")
        take = r5.number_input("Take-profit %", 0.01, 1.00,
                               float(risk.take_profit_pct), 0.01, format="%.2f")
        max_n = r6.number_input("Max simultaneous positions", 1, 20,
                                int(risk.max_positions))
        cooldown = st.number_input("Risk-exit cooldown (hours)", 1, 168,
                                   int(risk.cooldown_hours))
        if st.form_submit_button("Save Risk Limits", type="primary"):
            risk.max_position_notional = max_pos
            risk.max_gross_notional = max_gross
            risk.max_order_notional = max_order
            risk.stop_loss_pct = stop
            risk.take_profit_pct = take
            risk.max_positions = int(max_n)
            risk.cooldown_hours = int(cooldown)
            save_risk_limits(risk)
            st.success("Saved to config/config.yaml. Restart the engine to apply.")

    st.divider()
    st.markdown("**Current configuration**")
    st.code((REPO_ROOT / "config" / "config.yaml").read_text(), language="yaml")

# ------------------------------------------------------------- auto-refresh
if auto:
    time.sleep(10)
    st.rerun()
