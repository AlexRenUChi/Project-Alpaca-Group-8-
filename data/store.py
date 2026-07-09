"""Structured storage for market data and trading activity (SQLite).

One small database (storage/trading.db) with four tables:
  quotes   — incoming bid/ask snapshots (the data-pipeline log)
  signals  — every signal the strategy generates
  orders   — every order submitted, with its lifecycle status
  equity   — periodic account-equity snapshots for P&L / drawdown

SQLite is used because it is structured, queryable, file-based, and in the
standard library — no server needed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import STORAGE_DIR

DB_PATH = STORAGE_DIR / "trading.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    ts TEXT NOT NULL, symbol TEXT NOT NULL,
    bid REAL, ask REAL, bid_size REAL, ask_size REAL
);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL, symbol TEXT NOT NULL,
    signal TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    ts TEXT NOT NULL, order_id TEXT, symbol TEXT NOT NULL,
    side TEXT NOT NULL, notional REAL, qty REAL,
    status TEXT NOT NULL, reason TEXT, realized_pnl REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL, equity REAL NOT NULL, cash REAL, gross_exposure REAL
);
CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes (ts);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders (ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeStore:
    """Tiny persistence layer shared by the engine and the UI."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    # ------------------------------------------------------------- writers
    def log_quotes(self, quotes: dict[str, dict]) -> None:
        rows = [(_now(), sym, q.get("bid"), q.get("ask"),
                 q.get("bid_size"), q.get("ask_size")) for sym, q in quotes.items()]
        with self._conn() as conn:
            conn.executemany("INSERT INTO quotes VALUES (?,?,?,?,?,?)", rows)

    def log_signal(self, symbol: str, signal: str, detail: str = "") -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO signals VALUES (?,?,?,?)",
                         (_now(), symbol, signal, detail))

    def log_order(self, symbol: str, side: str, status: str, order_id: str = "",
                  notional: float | None = None, qty: float | None = None,
                  reason: str = "", realized_pnl: float | None = None) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                         (_now(), order_id, symbol, side, notional, qty,
                          status, reason, realized_pnl))

    def log_equity(self, equity: float, cash: float | None = None,
                   gross_exposure: float | None = None) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO equity VALUES (?,?,?,?)",
                         (_now(), equity, cash, gross_exposure))

    # ------------------------------------------------------------- readers
    def _read(self, query: str, params: tuple = ()) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def recent_quotes(self, limit: int = 100) -> pd.DataFrame:
        return self._read("SELECT * FROM quotes ORDER BY ts DESC LIMIT ?", (limit,))

    def recent_signals(self, limit: int = 50) -> pd.DataFrame:
        return self._read("SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (limit,))

    def recent_orders(self, limit: int = 50) -> pd.DataFrame:
        return self._read("SELECT * FROM orders ORDER BY ts DESC LIMIT ?", (limit,))

    def equity_curve(self) -> pd.DataFrame:
        df = self._read("SELECT * FROM equity ORDER BY ts")
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts")
        return df

    # ------------------------------------------------------------- metrics
    def performance_metrics(self) -> dict:
        """Cumulative P&L, max drawdown, trade count and hit rate from stored data."""
        out = {"cumulative_pnl": None, "max_drawdown": None,
               "num_trades": 0, "hit_rate": None}
        eq = self.equity_curve()
        if len(eq) >= 2:
            series = eq["equity"]
            out["cumulative_pnl"] = float(series.iloc[-1] - series.iloc[0])
            out["max_drawdown"] = float((series / series.cummax() - 1).min())
        orders = self._read(
            "SELECT realized_pnl FROM orders WHERE status='filled' AND side='sell'")
        closed = orders["realized_pnl"].dropna()
        filled = self._read("SELECT COUNT(*) AS n FROM orders WHERE status='filled'")
        out["num_trades"] = int(filled["n"].iloc[0])
        if len(closed):
            out["hit_rate"] = float((closed > 0).mean())
        return out


__all__ = ["TradeStore", "DB_PATH"]
