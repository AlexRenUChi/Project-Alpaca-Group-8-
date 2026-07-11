# Project Alpaca — Group 8

**FINM 25000 Homework 3 — Alpaca-based systematic trading system**

An end-to-end systematic trading system using **Alpaca paper trading only**: a
live data pipeline, a rule-based cross-sectional momentum strategy (plus an ML
research strategy), a risk-managed execution engine, a backtester, and a
Streamlit dashboard to monitor and control everything.

> ⚠️ **Paper trading only. No real money is ever used.** The trading client is
> hard-wired to `paper=True` and there is no code path that can trade a live
> account. Educational use only; not investment advice.

## Architecture

```
                        config/config.yaml   .env (keys, never committed)
                                 │                │
                                 ▼                ▼
 ┌──────────┐  bars/quotes ┌───────────┐ targets ┌──────────┐ intents ┌──────────┐
 │  Alpaca  │─────────────▶│   data/   │────────▶│ strategy/ │────────▶│  risk/   │
 │ Data API │              │ connector │ history │ momentum  │         │ manager  │
 └──────────┘              │  + store  │         │  or ML    │         └────┬─────┘
                           └─────┬─────┘         └──────────┘   approved │
                                 │ SQLite (quotes/signals/orders/equity) ▼
                           ┌─────▼─────┐                        ┌───────────────┐
                           │  ui/app   │◀── state.json ─────────│  execution/   │
                           │ Streamlit │─── start/stop flag ───▶│ engine+broker │
                           └───────────┘                        └───────┬───────┘
                                                                        │ orders
                                                                        ▼
                                                          Alpaca Trading API (paper=True)
```

The engine loop (`execution/engine.py`) runs every `poll_interval_sec`:
**data** (log latest quotes to SQLite, refresh rolling bar history) → **order
reconciliation** (recover open/non-terminal orders) → **risk** (stop-loss /
take-profit exits and persistent re-entry cooldowns) → **signals** (strategy →
target weights, logged) → **execution** (diff targets vs. positions *and open
orders* → risk-checked, idempotent orders) → **monitoring** (equity snapshot +
state file for the UI). Market orders are never queued while the market is
closed.

## Folder structure

```
config/       config.yaml (universe, strategy params, risk limits) + typed loader
data/         Alpaca data connector (REST bars, latest quotes, websocket stream)
              + SQLite store for quotes/signals/orders/equity
strategy/     base interface, momentum strategy (live), ML strategy (research),
              indicators + feature engineering, ML pipeline (PCA + classifier)
risk/         pre-trade checks and stop-loss/take-profit monitoring
execution/    paper broker wrapper (order states, retries), engine loop,
              engine↔UI state files
backtest/     single-asset and portfolio backtesters + performance metrics
ui/           Streamlit dashboard (monitor + control)
tests/        pytest suite (no network required)
run.py        CLI entry point (paper / backtest modes)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in keys generated from the Alpaca
dashboard with **Paper** selected. `.env` is in `.gitignore` — never commit
it. All other settings (tickers, strategy parameters, risk limits) are in
`config/config.yaml`.

## Running

```bash
# Dashboard (recommended): monitor, start/stop the engine, backtest, edit risk
streamlit run ui/app.py

# Engine directly from the CLI
python run.py --mode paper           # continuous paper-trading loop
python run.py --mode paper --once    # single cycle (handy for the video)
python run.py --mode backtest        # historical backtest of the configured strategy

# Tests
pytest
```

## Strategy

**Live strategy — cross-sectional momentum.** Stocks that outperformed peers
over the past month tend to keep outperforming over short horizons (Jegadeesh
& Titman, 1993). Each cycle we compute every symbol's 20-day return, discard
any trading below its 50-day moving average or with negative momentum (trend
filter), and hold the top 3 equal-weighted. Long-only, no leverage, cash
otherwise. Parameters in `config.yaml → strategy.momentum`.

**Research strategy — ML classifier.** Per symbol: 22 engineered features
(log returns, rolling stats, RSI, MACD, ADX, Bollinger position, ATR%, CMF,
OBV z-score, …) → StandardScaler → PCA keeping ≥80% variance → classifier
(Random Forest / LogReg / GBM / SVM / MLP) predicting P(next-day up); long if
P > 0.60. Available in the Backtesting tab and as `strategy.name: ml`. Daily
direction is hard to predict — treat it as a methodology exercise.

Both backtests shift positions one day forward (decide at close *t*, earn
return *t+1*) and fit scalers/PCA/models on the training window only, so
there is no look-ahead.

## Risk controls

All limits live in `config.yaml → risk` and are enforced by `risk/manager.py`
before any order reaches the broker. Existing positions and all open buy
orders are included in every projected-exposure check:

max $ per position (`max_position_notional`), max total exposure with no
leverage (`max_gross_notional`), max $ per order (`max_order_notional`),
max simultaneous positions (`max_positions`), plus per-position **stop-loss**
(−5%) and **take-profit** (+10%) exits checked every cycle before new signals.
A risk exit starts a persistent 24-hour cooldown, so the strategy cannot buy
the symbol back in the same or next cycle. Sell/close orders are always allowed
since they reduce risk. Rejected orders are logged with the reason and shown
in the UI.

The portfolio backtest uses these same gross/position caps, stop and take-profit
rules, and cooldown. It also applies the configured transaction-cost and
slippage assumptions. The equal-weight benchmark is exposure-matched to make
the comparison meaningful.

## Monitoring & logging

Everything is logged to `logs/system.log` and to a SQLite database
(`storage/trading.db`): incoming quotes (timestamps, bid/ask), every signal,
every unique order-state transition (`accepted/new → partially_filled → filled
/ canceled / rejected / risk_rejected`), persistent cooldowns, and periodic
equity snapshots. Non-terminal orders are reconciled in later cycles and after
restarts. Realized P&L uses actual filled quantity and average fill price; the
dashboard's trade count and hit rate are based on completed sell/close trades.

## Example walkthrough

1. `streamlit run ui/app.py` → sidebar shows **🔴 Engine: STOPPED**.
2. Click **▶ Start** — the engine launches as a separate process; the sidebar
   flips to **🟢 RUNNING** and the market-open flag updates.
3. Dashboard tab: equity, positions, and the signals/orders tables fill in as
   cycles complete (run during US market hours to see fills).
4. Backtesting tab: run momentum vs. the exposure-matched equal-weight
   benchmark — metrics table, equity curves, drawdown chart.
5. Risk & Config tab: tighten `max_position_notional`, save, restart the
   engine, and watch oversized orders get rejected with reasons.
6. Click **⏹ Stop** — the engine shuts down cleanly within seconds.

## Error handling

Network errors and rate limits are retried with exponential backoff (data and
broker layers). Every market order has a unique `client_order_id`, so a timeout
can recover the already-accepted order without duplicating it. Rejected orders
are caught and logged, never crash the loop. A failed cycle logs the exception,
surfaces `status: error` to the UI, and the engine continues on the next cycle.
Outside market hours signals are recorded, but no market orders are submitted.

## Limitations & possible improvements

Daily-bar momentum and close-to-close stop simulation ignore intraday paths;
configured transaction costs and slippage are estimates, not an execution
simulator. The engine reconciles orders through REST rather than maintaining a
persistent trade-update websocket (the market-data stream is implemented in
`data/connector.py`); config changes require an engine restart. Natural
extensions: limit orders with resting-order management, trade-update streaming,
walk-forward retraining of the ML strategy, and a proper database for
multi-day history.

## Video

10–15 minutes covering: architecture (this diagram), the strategy rules and
risk controls, a live demo of the dashboard with the engine running in paper
mode (start → signals → orders → fills → stop), the backtest results, and
reflections on limitations. Add the link here: **[video link]**.
