"""Alpaca Market Data connector (data module).

Handles all communication with Alpaca's data APIs:
  * historical bars (REST, paginated)
  * latest quotes for a multi-symbol universe (REST)
  * realtime quote/trade streaming (websocket)

Credentials come from environment variables (.env) — never hard-coded.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pandas as pd
import requests
import websockets

from config.settings import alpaca_keys

log = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaDataConnector:
    """Thin, retry-aware wrapper around Alpaca's market-data endpoints."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 feed: str = "iex", max_retries: int = 3):
        if api_key and secret_key:
            self.api_key, self.secret_key = api_key, secret_key
        else:
            self.api_key, self.secret_key = alpaca_keys()
        self.feed = feed
        self.max_retries = max_retries

    # ------------------------------------------------------------------ REST
    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def _get(self, url: str, params: dict) -> dict:
        """GET with basic retry on network errors / 5xx / rate limits."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                last_err = err
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    log.warning("Data request failed (%s); retry %d/%d in %ds",
                                err, attempt, self.max_retries, wait)
                    import time
                    time.sleep(wait)
        raise RuntimeError(f"Alpaca data request failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _to_rfc3339(value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        return str(value)

    def get_historical(self, symbol: str, start: Any = None, end: Any = None,
                       timeframe: str = "1Day", limit: int = 10_000,
                       feed: Optional[str] = None) -> pd.DataFrame:
        """Historical bars for one symbol, paging through next_page_token.

        Returns a DataFrame indexed by timestamp with open/high/low/close/volume.
        """
        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
        params = {"timeframe": timeframe, "limit": limit, "feed": feed or self.feed}
        if start:
            params["start"] = self._to_rfc3339(start)
        if end:
            params["end"] = self._to_rfc3339(end)

        all_bars: list[dict] = []
        while True:
            payload = self._get(url, params)
            bars = payload.get("bars") or []
            all_bars.extend(bars)
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token

        df = pd.DataFrame(all_bars)
        if not df.empty and "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t").rename(
                columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df

    def get_history_universe(self, symbols: list[str], days: int = 200,
                             timeframe: str = "1Day") -> dict[str, pd.DataFrame]:
        """Rolling daily history for every symbol in the universe."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(days * 1.6) + 10)  # calendar > trading days
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.get_historical(sym, start=start, end=end, timeframe=timeframe)
                if not df.empty:
                    out[sym] = df
                    log.info("data | %s: %d bars through %s", sym, len(df), df.index[-1].date())
                else:
                    log.warning("data | %s: no bars returned", sym)
            except Exception as err:
                log.error("data | %s: fetch failed: %s", sym, err)
        return out

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Latest bid/ask for many symbols in one request.

        Returns {symbol: {bid, ask, bid_size, ask_size, ts}}.
        """
        url = f"{DATA_BASE_URL}/v2/stocks/quotes/latest"
        payload = self._get(url, {"symbols": ",".join(symbols), "feed": self.feed})
        quotes = {}
        for sym, q in (payload.get("quotes") or {}).items():
            quotes[sym] = {
                "bid": q.get("bp"), "ask": q.get("ap"),
                "bid_size": q.get("bs"), "ask_size": q.get("as"),
                "ts": q.get("t"),
            }
        return quotes

    # ------------------------------------------------------------- streaming
    async def stream_market_data(self, symbols: list[str] | str,
                                 on_message: Optional[Callable[[dict], None]] = None,
                                 stop_event: Optional[Any] = None) -> None:
        """Stream quotes+trades for one or more symbols.

        Each message is normalized to
        {"type": "quote"|"trade", "symbol", "bid", "ask", "price", "raw"}.
        `stop_event` (threading.Event) stops the loop from another thread.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        ws_url = f"wss://stream.data.alpaca.markets/v2/{self.feed}"
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"action": "auth", "key": self.api_key,
                                      "secret": self.secret_key}))
            try:
                await ws.recv()
            except Exception:
                pass
            await ws.send(json.dumps({"action": "subscribe",
                                      "quotes": symbols, "trades": symbols}))

            while stop_event is None or not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                for m in (data if isinstance(data, list) else [data]):
                    msg_type = m.get("T")
                    sym = m.get("S")
                    if msg_type == "t":
                        normalized = {"type": "trade", "symbol": sym, "bid": None,
                                      "ask": None, "price": m.get("p"), "raw": m}
                    elif msg_type == "q":
                        normalized = {"type": "quote", "symbol": sym, "bid": m.get("bp"),
                                      "ask": m.get("ap"), "price": None, "raw": m}
                    else:
                        continue
                    if on_message:
                        try:
                            on_message(normalized)
                        except Exception:
                            pass  # keep streaming even if the callback fails
                    else:
                        print(normalized)


__all__ = ["AlpacaDataConnector"]
