"""
Delta Exchange WebSocket price stream.
Keeps a live price cache updated in a background thread.
REST candles are still used for analysis — WS gives us the real-time price.
"""

import json
import logging
import threading
import time

import websocket

import config

logger = logging.getLogger(__name__)

# Live price cache  { symbol: float }
_prices: dict[str, float] = {}
_lock = threading.Lock()
_ws_app: websocket.WebSocketApp | None = None


def get_price(symbol: str) -> float | None:
    with _lock:
        return _prices.get(symbol)


def _on_open(ws):
    logger.info("WebSocket connected — subscribing to tickers")
    msg = {
        "type": "subscribe",
        "payload": {
            "channels": [
                {"name": "v2/ticker", "symbols": config.SYMBOLS}
            ]
        }
    }
    ws.send(json.dumps(msg))


def _on_message(ws, raw):
    try:
        msg = json.loads(raw)
        if msg.get("type") != "v2/ticker":
            return

        symbol = msg.get("symbol")
        if symbol not in config.SYMBOLS:
            return

        # Delta's v2/ticker payload carries quotes/mark_price/close at the TOP
        # level of the message, not nested under a "data" key. The old code
        # read msg["data"] (always empty), so no price was ever cached and
        # every lookup silently fell back to REST. Handle both shapes.
        data = msg if ("quotes" in msg or "mark_price" in msg or "close" in msg) \
            else (msg.get("data") or {})
        quotes = data.get("quotes") or {}

        # Prefer bid/ask mid — matches what the app shows
        try:
            bid = float(quotes.get("best_bid") or 0)
            ask = float(quotes.get("best_ask") or 0)
            if bid > 0 and ask > 0:
                price = round((bid + ask) / 2, 2)
            else:
                price = float(data.get("mark_price") or data.get("close") or 0)
        except (TypeError, ValueError):
            return

        if price > 0:
            with _lock:
                _prices[symbol] = price

    except Exception:
        logger.exception("WS message parse error")


def _on_error(ws, error):
    logger.warning("WebSocket error: %s", error)


def _on_close(ws, code, msg):
    logger.warning("WebSocket closed (code=%s) — will reconnect", code)


def _run_forever():
    global _ws_app
    while True:
        try:
            _ws_app = websocket.WebSocketApp(
                config.WS_URL,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            _ws_app.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            logger.exception("WebSocket crashed — reconnecting in 10s")
        time.sleep(10)


def start():
    """Start WS in a daemon thread — call once at startup."""
    t = threading.Thread(target=_run_forever, daemon=True, name="ws-stream")
    t.start()
    logger.info("WebSocket stream started for %s", config.SYMBOLS)
