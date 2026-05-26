"""
Multi-timeframe market data manager.
Fetches OHLCV data from a BaseExchange, converts to DataFrames,
and provides a simple in-memory cache with TTL.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from exchanges.base import BaseExchange, OHLCV

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Number of candles to fetch per timeframe
CANDLE_LIMITS: Dict[str, int] = {
    "1m":  20,
    "5m":  30,
    "15m": 50,
    "1h":  100,
    "4h":  200,
    "1d":  200,
}

CACHE_TTL_SECONDS = 60


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

class _CacheEntry:
    __slots__ = ("data", "expires_at")

    def __init__(self, data: pd.DataFrame, ttl: float = CACHE_TTL_SECONDS) -> None:
        self.data = data
        self.expires_at = time.monotonic() + ttl

    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


# ---------------------------------------------------------------------------
# MarketDataManager
# ---------------------------------------------------------------------------

class MarketDataManager:
    """
    Manages multi-timeframe OHLCV data for one or more symbols.

    Features
    --------
    - fetch_mtf_candles(symbol) → dict[timeframe, pd.DataFrame]
    - get_current_price(symbol) → float
    - fetch_funding_rate(symbol) → float
    - fetch_open_interest(symbol) → float
    - In-memory per-(symbol, timeframe) cache with 60 s TTL
    """

    def __init__(self, exchange: BaseExchange) -> None:
        """
        Parameters
        ----------
        exchange : Any object implementing BaseExchange.
        """
        self._exchange = exchange
        # Cache keys: (symbol, timeframe) → _CacheEntry
        self._cache: Dict[Tuple[str, str], _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_mtf_candles(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all timeframes in TIMEFRAMES.

        Returns
        -------
        dict mapping timeframe string to a pd.DataFrame with columns:
            timestamp (float64), open, high, low, close, volume (float64)
        Oldest row first.
        """
        result: Dict[str, pd.DataFrame] = {}
        for tf in TIMEFRAMES:
            df = self._fetch_with_cache(symbol, tf)
            if df is not None:
                result[tf] = df
            else:
                logger.warning(
                    "[MarketData] No data returned for %s %s", symbol, tf
                )
        return result

    def get_current_price(self, symbol: str) -> float:
        """
        Return the latest close price for the symbol using the 1m frame.
        Falls back to fetch_ticker if candle data is unavailable.
        """
        df = self._fetch_with_cache(symbol, "1m")
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])

        # Fallback: use exchange ticker
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            mid = (ticker.bid + ticker.ask) / 2.0
            if mid > 0:
                return round(mid, 8)
            if ticker.last and ticker.last > 0:
                return float(ticker.last)
        except Exception as exc:
            logger.error(
                "[MarketData] get_current_price ticker fallback failed for %s: %s",
                symbol, exc,
            )
        return 0.0

    def fetch_funding_rate(self, symbol: str) -> float:
        """Return the current funding rate for a perpetual contract."""
        try:
            rate = self._exchange.fetch_funding_rate(symbol)
            logger.debug(
                "[MarketData] Funding rate for %s: %s", symbol, rate
            )
            return float(rate)
        except Exception as exc:
            logger.error(
                "[MarketData] fetch_funding_rate failed for %s: %s", symbol, exc
            )
            return 0.0

    def fetch_open_interest(self, symbol: str) -> float:
        """Return the current open interest (USD) for a perpetual contract."""
        try:
            oi = self._exchange.fetch_open_interest(symbol)
            logger.debug(
                "[MarketData] Open interest for %s: %s", symbol, oi
            )
            return float(oi)
        except Exception as exc:
            logger.error(
                "[MarketData] fetch_open_interest failed for %s: %s", symbol, exc
            )
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, symbol: str, timeframe: str) -> Tuple[str, str]:
        return (symbol, timeframe)

    def _fetch_with_cache(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Return cached DataFrame if still valid, otherwise fetch fresh data."""
        key = self._cache_key(symbol, timeframe)
        entry = self._cache.get(key)

        if entry is not None and entry.is_valid():
            logger.debug(
                "[MarketData] Cache hit for %s %s", symbol, timeframe
            )
            return entry.data

        # Fetch fresh data
        limit = CANDLE_LIMITS.get(timeframe, 100)
        logger.info(
            "[MarketData] Fetching %s %s (%d candles)…", symbol, timeframe, limit
        )

        try:
            candles = self._exchange.fetch_ohlcv(symbol, timeframe, limit)
        except Exception as exc:
            logger.error(
                "[MarketData] fetch_ohlcv failed for %s %s: %s",
                symbol, timeframe, exc,
            )
            return None

        if not candles:
            logger.warning(
                "[MarketData] Empty candle list for %s %s", symbol, timeframe
            )
            return None

        df = self._candles_to_df(candles)

        # Store in cache
        self._cache[key] = _CacheEntry(df)
        logger.info(
            "[MarketData] Cached %d candles for %s %s",
            len(df), symbol, timeframe,
        )
        return df

    @staticmethod
    def _candles_to_df(candles: list[OHLCV]) -> pd.DataFrame:
        """Convert a list of OHLCV dataclasses to a typed pandas DataFrame."""
        rows = [
            {
                "timestamp": c.timestamp,
                "open":      c.open,
                "high":      c.high,
                "low":       c.low,
                "close":     c.close,
                "volume":    c.volume,
            }
            for c in candles
        ]
        df = pd.DataFrame(rows)
        for col in df.columns:
            df[col] = df[col].astype("float64")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """
        Manually invalidate cache entries.
        If symbol is None, clears the entire cache.
        """
        if symbol is None:
            self._cache.clear()
            logger.info("[MarketData] Full cache cleared.")
        else:
            keys_to_remove = [k for k in self._cache if k[0] == symbol]
            for k in keys_to_remove:
                del self._cache[k]
            logger.info("[MarketData] Cache cleared for %s.", symbol)
