"""Structured storage for market data and trading activity (SQLite).

One small database (storage/trading.db) with six tables:
  quotes   — incoming bid/ask snapshots (the data-pipeline log)
  signals  — every signal the strategy generates
  orders   — legacy-compatible one-row order log
  order_events — unique states, fills, prices and realized P&L per order
  equity   — periodic account-equity snapshots for P&L / drawdown
  cooldowns — persistent stop-loss/take-profit re-entry blocks

SQLite is used because it is structured, queryable, file-based, and in the
standard library — no server needed.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
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
CREATE TABLE IF NOT EXISTS order_events (
    ts TEXT NOT NULL, order_id TEXT NOT NULL, symbol TEXT NOT NULL,
    side TEXT NOT NULL, status TEXT NOT NULL,
    requested_notional REAL, filled_qty REAL NOT NULL DEFAULT 0,
    filled_avg_price REAL, entry_price REAL, reason TEXT, realized_pnl REAL,
    UNIQUE(order_id, status, filled_qty)
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL, equity REAL NOT NULL, cash REAL, gross_exposure REAL
);
CREATE TABLE IF NOT EXISTS cooldowns (
    symbol TEXT PRIMARY KEY, until_ts TEXT NOT NULL, reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quotes (ts);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders (ts);
CREATE INDEX IF NOT EXISTS idx_order_events_ts ON order_events (ts);
CREATE INDEX IF NOT EXISTS idx_order_events_id ON order_events (order_id);
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

    def log_order_event(self, order_id: str, symbol: str, side: str, status: str,
                        requested_notional: float | None = None,
                        filled_qty: float = 0.0,
                        filled_avg_price: float | None = None,
                        entry_price: float | None = None,
                        reason: str = "",
                        realized_pnl: float | None = None) -> None:
        """Persist one unique lifecycle state for an order.

        The unique key makes repeated REST reconciliation idempotent while still
        retaining partial-fill progress when filled quantity changes.
        """
        order_id = order_id or f"local-{uuid.uuid4()}"
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO order_events
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (_now(), order_id, symbol, side, status, requested_notional,
                 float(filled_qty or 0.0), filled_avg_price, entry_price,
                 reason, realized_pnl),
            )

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
        events = self._read(
            """SELECT ts, order_id, symbol, side, requested_notional AS notional,
                      filled_qty AS qty, filled_avg_price, status, reason, realized_pnl
               FROM order_events ORDER BY ts DESC LIMIT ?""", (limit,))
        if not events.empty:
            return events
        return self._read("SELECT * FROM orders ORDER BY ts DESC LIMIT ?", (limit,))

    def pending_order_ids(self) -> list[str]:
        """Order IDs whose latest observed state is not terminal."""
        terminal = ("filled", "canceled", "expired", "rejected", "replaced",
                    "done_for_day", "stopped", "suspended", "calculated",
                    "risk_rejected")
        placeholders = ",".join("?" for _ in terminal)
        df = self._read(
            f"""SELECT e.order_id
                FROM order_events e
                JOIN (SELECT order_id, MAX(rowid) AS max_rowid
                      FROM order_events GROUP BY order_id) latest
                  ON e.rowid = latest.max_rowid
                WHERE e.status NOT IN ({placeholders})""", terminal)
        return df["order_id"].tolist() if not df.empty else []

    def order_context(self, order_id: str) -> dict:
        df = self._read(
            """SELECT symbol, side, requested_notional, entry_price, reason
               FROM order_events WHERE order_id=? ORDER BY rowid LIMIT 1""",
            (order_id,),
        )
        if df.empty:
            return {}
        return {key: (None if pd.isna(value) else value)
                for key, value in df.iloc[0].to_dict().items()}

    def set_cooldown(self, symbol: str, hours: int, reason: str) -> None:
        until = datetime.now(timezone.utc) + timedelta(hours=max(1, hours))
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO cooldowns(symbol, until_ts, reason) VALUES (?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET
                     until_ts=excluded.until_ts, reason=excluded.reason""",
                (symbol, until.isoformat(), reason),
            )

    def active_cooldowns(self) -> dict[str, str]:
        now = _now()
        with self._conn() as conn:
            conn.execute("DELETE FROM cooldowns WHERE until_ts <= ?", (now,))
        df = self._read("SELECT symbol, reason FROM cooldowns WHERE until_ts > ?", (now,))
        return dict(zip(df["symbol"], df["reason"])) if not df.empty else {}

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
        terminal = ("filled", "canceled", "expired", "rejected", "replaced",
                    "done_for_day", "stopped", "suspended", "calculated")
        placeholders = ",".join("?" for _ in terminal)
        orders = self._read(
            f"""SELECT e.order_id, e.realized_pnl
                FROM order_events e
                JOIN (SELECT order_id, MAX(rowid) AS max_rowid
                      FROM order_events GROUP BY order_id) latest
                  ON e.rowid = latest.max_rowid
                WHERE e.side='sell' AND e.status IN ({placeholders})
                  AND e.filled_qty > 0""", terminal)
        closed = orders["realized_pnl"].dropna()
        out["num_trades"] = int(len(closed))
        if len(closed):
            out["hit_rate"] = float((closed > 0).mean())
        return out


__all__ = ["TradeStore", "DB_PATH"]
