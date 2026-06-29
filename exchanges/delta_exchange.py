"""
Delta Exchange India — REST API wrapper extending BaseExchange.
Supports market data (public) and trading (private, HMAC-SHA256 signed).
Base URL: https://api.india.delta.exchange
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

import config
from exchanges.base import BaseExchange, OHLCV, OrderResult, Ticker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = config.BASE_URL  # switches between live and testnet via config.USE_TESTNET

# Perpetual product IDs on Delta Exchange India
PRODUCT_IDS: Dict[str, int] = {
    "BTCUSD":  27,
    "ETHUSD":  3136,
    "XAUTUSD": 131253,
}

# API resolution string for each timeframe
RESOLUTION_MAP: Dict[str, str] = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# Resolution in minutes (used to calculate start timestamp)
RESOLUTION_MINUTES: Dict[str, int] = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
}

_MAX_RETRY_ATTEMPTS = 3
_RETRY_WAIT_SECONDS = 60


# ---------------------------------------------------------------------------
# DeltaExchange
# ---------------------------------------------------------------------------

class DeltaExchange(BaseExchange):
    """
    Delta Exchange India REST API wrapper.
    Public endpoints (market data) need no authentication.
    Private endpoints (orders, balance) require api_key + api_secret.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        timeout: int = 15,
    ) -> None:
        """
        Parameters
        ----------
        api_key    : Delta Exchange API key (required for private endpoints)
        api_secret : Delta Exchange API secret
        timeout    : HTTP request timeout in seconds
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept":       "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Perform a public GET request with retry logic."""
        url = BASE_URL + endpoint
        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.get(
                    url, params=params, timeout=self._timeout
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("success") is False:
                    logger.warning(
                        "[DeltaExchange] API returned success=false: %s", data
                    )
                    return None
                return data
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "[DeltaExchange] GET %s failed (attempt %d/%d): %s",
                    endpoint, attempt, _MAX_RETRY_ATTEMPTS, exc,
                )
                if attempt < _MAX_RETRY_ATTEMPTS:
                    time.sleep(_RETRY_WAIT_SECONDS)
        return None

    def _signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Perform an authenticated (HMAC-SHA256 signed) request.
        Signature scheme follows Delta Exchange API docs.
        """
        if not self._api_key or not self._api_secret:
            raise RuntimeError(
                "api_key and api_secret are required for private API calls."
            )

        timestamp = str(int(time.time()))
        method_upper = method.upper()

        # Build query string for GET params
        query_string = ""
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))

        # Build body string
        body_string = ""
        if body:
            import json as _json
            body_string = _json.dumps(body, separators=(",", ":"))

        # Signature message: METHOD + timestamp + /endpoint + query_string + body
        sig_message = method_upper + timestamp + endpoint
        if query_string:
            sig_message += "?" + query_string
        sig_message += body_string

        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            sig_message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "api-key":   self._api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json",
        }

        url = BASE_URL + endpoint
        if query_string:
            url += "?" + query_string

        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            try:
                if method_upper == "GET":
                    resp = self._session.get(
                        url, headers=headers, timeout=self._timeout
                    )
                elif method_upper == "POST":
                    resp = self._session.post(
                        url, headers=headers, data=body_string, timeout=self._timeout
                    )
                elif method_upper == "DELETE":
                    resp = self._session.delete(
                        url, headers=headers, data=body_string, timeout=self._timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method_upper}")

                # Don't retry auth errors — they won't fix themselves
                if resp.status_code in (401, 403):
                    logger.warning(
                        "[DeltaExchange] %s %s returned %d — check API key/IP whitelist, not retrying",
                        method_upper, endpoint, resp.status_code,
                    )
                    return None

                resp.raise_for_status()
                data = resp.json()
                if data.get("success") is False:
                    logger.warning(
                        "[DeltaExchange] Private API returned success=false: %s", data
                    )
                    return None
                return data

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "[DeltaExchange] %s %s failed (attempt %d/%d): %s",
                    method_upper, endpoint, attempt, _MAX_RETRY_ATTEMPTS, exc,
                )
                if attempt < _MAX_RETRY_ATTEMPTS:
                    time.sleep(_RETRY_WAIT_SECONDS)

        return None

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candles from /v2/history/candles.
        Returns a list of OHLCV dataclasses, oldest first.
        """
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            logger.error("[DeltaExchange] Unknown symbol: %s", symbol)
            return []

        resolution = RESOLUTION_MAP.get(timeframe)
        if resolution is None:
            logger.error("[DeltaExchange] Unknown timeframe: %s", timeframe)
            return []

        res_minutes = RESOLUTION_MINUTES[timeframe]
        now_ts   = int(datetime.now(timezone.utc).timestamp())
        start_ts = now_ts - res_minutes * 60 * (limit + 5)  # small buffer

        params = {
            "resolution": resolution,
            "symbol":     symbol,
            "start":      start_ts,
            "end":        now_ts,
        }

        data = self._get("/v2/history/candles", params)
        if data is None:
            return []

        candles = data.get("result", [])
        if not candles:
            logger.warning(
                "[DeltaExchange] No candle data for %s %s", symbol, timeframe
            )
            return []

        result: List[OHLCV] = []
        for c in candles:
            try:
                result.append(
                    OHLCV(
                        timestamp=float(c.get("time", 0)),
                        open=float(c.get("open", 0)),
                        high=float(c.get("high", 0)),
                        low=float(c.get("low", 0)),
                        close=float(c.get("close", 0)),
                        volume=float(c.get("volume", 0) or 0),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning("[DeltaExchange] Skipping malformed candle: %s", exc)

        # Sort oldest first, trim to requested limit
        result.sort(key=lambda x: x.timestamp)
        return result[-limit:]

    def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch the latest ticker for a symbol from /v2/tickers.
        Returns a Ticker dataclass.
        """
        data = self._get("/v2/tickers", {"symbol": symbol})
        if data is None:
            raise RuntimeError(f"Failed to fetch ticker for {symbol}")

        result = data.get("result", {})
        if isinstance(result, list):
            ticker_raw = next(
                (item for item in result if item.get("symbol") == symbol), None
            )
            if ticker_raw is None:
                raise RuntimeError(f"Ticker for {symbol} not found in response")
        else:
            ticker_raw = result

        quotes = ticker_raw.get("quotes") or {}
        bid  = float(quotes.get("best_bid") or ticker_raw.get("bid") or 0.0)
        ask  = float(quotes.get("best_ask") or ticker_raw.get("ask") or 0.0)
        last = float(ticker_raw.get("close") or ticker_raw.get("last") or 0.0)

        funding_rate: Optional[float] = None
        raw_fr = ticker_raw.get("funding_rate")
        if raw_fr is not None:
            try:
                funding_rate = float(raw_fr)
            except (TypeError, ValueError):
                pass

        open_interest: Optional[float] = None
        raw_oi = ticker_raw.get("oi_value_usd")
        if raw_oi is not None:
            try:
                open_interest = float(raw_oi)
            except (TypeError, ValueError):
                pass

        mark_price: Optional[float] = None
        raw_mp = ticker_raw.get("mark_price")
        if raw_mp is not None:
            try:
                mark_price = float(raw_mp)
            except (TypeError, ValueError):
                pass

        spot_price: Optional[float] = None
        raw_sp = ticker_raw.get("spot_price")
        if raw_sp is not None:
            try:
                spot_price = float(raw_sp)
            except (TypeError, ValueError):
                pass

        return Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            mark_price=mark_price,
            spot_price=spot_price,
            funding_rate=funding_rate,
            open_interest=open_interest,
        )

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Returns the mid-price from best bid/ask.
        Falls back to mark_price then close if quotes are unavailable.
        Mirrors the logic in the legacy delta_api.py.
        """
        data = self._get("/v2/tickers", {"symbol": symbol})
        if data is None:
            return None

        result = data.get("result", {})
        if isinstance(result, list):
            ticker_raw = next(
                (item for item in result if item.get("symbol") == symbol), None
            )
        else:
            ticker_raw = result

        if ticker_raw is None:
            return None

        quotes = ticker_raw.get("quotes") or {}
        try:
            best_ask = float(quotes.get("best_ask") or 0)
            best_bid = float(quotes.get("best_bid") or 0)
            if best_ask > 0 and best_bid > 0:
                return round((best_ask + best_bid) / 2, 2)
        except (TypeError, ValueError):
            pass

        for field_name in ("mark_price", "close"):
            try:
                val = ticker_raw.get(field_name)
                if val:
                    return float(val)
            except (TypeError, ValueError):
                pass

        logger.warning(
            "[DeltaExchange] Could not parse price from ticker: %s", ticker_raw
        )
        return None

    def fetch_orderbook(self, symbol: str) -> dict:
        """Fetch the order book for a symbol."""
        data = self._get("/v2/l2orderbook/" + symbol, {})
        if data is None:
            return {"bids": [], "asks": []}

        result = data.get("result", {})
        bids = [
            [float(entry["limit_price"]), float(entry["size"])]
            for entry in result.get("buy", [])
        ]
        asks = [
            [float(entry["limit_price"]), float(entry["size"])]
            for entry in result.get("sell", [])
        ]
        return {"bids": bids, "asks": asks}

    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch the current funding rate from the ticker."""
        try:
            ticker = self.fetch_ticker(symbol)
            return ticker.funding_rate or 0.0
        except Exception as exc:
            logger.warning(
                "[DeltaExchange] fetch_funding_rate failed for %s: %s", symbol, exc
            )
            return 0.0

    def fetch_open_interest(self, symbol: str) -> float:
        """Fetch the current open interest (USD) from the ticker."""
        try:
            ticker = self.fetch_ticker(symbol)
            return ticker.open_interest or 0.0
        except Exception as exc:
            logger.warning(
                "[DeltaExchange] fetch_open_interest failed for %s: %s", symbol, exc
            )
            return 0.0

    # ------------------------------------------------------------------
    # Private / authenticated endpoints
    # ------------------------------------------------------------------

    def fetch_balance(self) -> dict:
        """
        Fetch wallet balances for all assets.
        Returns dict of asset → available_balance float.
        """
        data = self._signed_request("GET", "/v2/wallet/balances")
        if data is None:
            return {}

        balances: dict = {}
        for entry in data.get("result", []):
            asset   = entry.get("asset_symbol") or entry.get("currency", "")
            balance = entry.get("available_balance") or entry.get("balance") or 0
            try:
                balances[asset] = float(balance)
            except (TypeError, ValueError):
                pass
        return balances

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a new order on Delta Exchange.
        side:        'buy' or 'sell'
        order_type:  'market_order' or 'limit_order'
        stop_loss:   bracket SL trigger price (optional)
        take_profit: bracket TP trigger price (optional)
        """
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            raise ValueError(f"Unknown symbol '{symbol}' — no product_id mapping.")

        # Delta Exchange uses 'market_order' and 'limit_order'
        order_type_api = order_type
        if order_type == "market":
            order_type_api = "market_order"
        elif order_type == "limit":
            order_type_api = "limit_order"

        payload: dict = {
            "product_id":   product_id,
            "side":         side,
            "order_type":   order_type_api,
            "size":         int(amount),  # Delta uses contract size as integer
        }
        if price is not None and order_type_api == "limit_order":
            payload["limit_price"] = str(price)

        # Bracket orders: attach SL and TP to the entry so Delta manages them
        if stop_loss is not None:
            payload["bracket_stop_loss_price"] = str(stop_loss)
            payload["bracket_stop_loss_limit_price"] = str(stop_loss)
        if take_profit is not None:
            payload["bracket_take_profit_price"] = str(take_profit)
            payload["bracket_take_profit_limit_price"] = str(take_profit)

        data = self._signed_request("POST", "/v2/orders", body=payload)
        if data is None:
            raise RuntimeError(f"place_order failed for {symbol}")

        result = data.get("result", {})
        return OrderResult(
            order_id=str(result.get("id", "")),
            symbol=symbol,
            side=side,
            type=order_type,
            amount=float(result.get("size") or amount),
            price=float(result.get("limit_price") or price or 0.0),
            status=str(result.get("state") or "open"),
            timestamp=time.time(),
        )

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order by order ID."""
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            logger.error(
                "[DeltaExchange] cancel_order: unknown symbol %s", symbol
            )
            return False

        payload = {"id": int(order_id), "product_id": product_id}
        data = self._signed_request(
            "DELETE", f"/v2/orders/{order_id}", body=payload
        )
        if data is None:
            logger.error(
                "[DeltaExchange] cancel_order failed for order %s / %s",
                order_id, symbol,
            )
            return False

        logger.info(
            "[DeltaExchange] Order %s for %s cancelled.", order_id, symbol
        )
        return True

    def fetch_position(self, symbol: str) -> Optional[dict]:
        """
        Fetch open position for a symbol.
        Returns the position dict if size != 0, else None (flat).
        """
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            logger.error("[DeltaExchange] fetch_position: unknown symbol %s", symbol)
            return None
        data = self._signed_request(
            "GET", "/v2/positions/margined",
            params={"product_id": str(product_id)},
        )
        if data is None:
            return None
        result = data.get("result", {})
        positions = result if isinstance(result, list) else ([result] if result else [])
        for pos in positions:
            if pos.get("product_id") == product_id and float(pos.get("size", 0)) != 0:
                return pos
        return None

    def fetch_recent_fills(self, symbol: str, limit: int = 5) -> List[dict]:
        """Fetch recent trade fills for a symbol (used to determine exit price)."""
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            return []
        data = self._signed_request(
            "GET", "/v2/fills",
            params={"product_id": str(product_id), "limit": str(limit)},
        )
        if data is None:
            return []
        result = data.get("result", {})
        fills = result.get("fills", result) if isinstance(result, dict) else result
        return fills if isinstance(fills, list) else []

    def fetch_open_orders(self, symbol: str) -> List[OrderResult]:
        """Fetch all open orders for a symbol."""
        product_id = PRODUCT_IDS.get(symbol)
        if product_id is None:
            logger.error(
                "[DeltaExchange] fetch_open_orders: unknown symbol %s", symbol
            )
            return []

        data = self._signed_request(
            "GET", "/v2/orders", params={"product_id": str(product_id), "state": "open"}
        )
        if data is None:
            return []

        orders: List[OrderResult] = []
        for entry in data.get("result", []):
            order_type_raw = entry.get("order_type", "")
            if order_type_raw == "market_order":
                order_type_normalized = "market"
            elif order_type_raw == "limit_order":
                order_type_normalized = "limit"
            else:
                order_type_normalized = order_type_raw

            orders.append(
                OrderResult(
                    order_id=str(entry.get("id", "")),
                    symbol=symbol,
                    side=str(entry.get("side", "")),
                    type=order_type_normalized,
                    amount=float(entry.get("size") or 0.0),
                    price=float(entry.get("limit_price") or 0.0),
                    status=str(entry.get("state") or "open"),
                    timestamp=time.time(),
                )
            )
        return orders
