"""
Tests for core.market_data.MarketDataManager, including the concurrent
fetch_mtf_candles() implementation — correctness of per-timeframe data,
caching, and that timeframes are actually fetched in parallel (not
serially) so a regression back to sequential fetching would be caught.

Run: python -m unittest tests.test_market_data -v
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_data import MarketDataManager, TIMEFRAMES  # noqa: E402
from exchanges.base import OHLCV  # noqa: E402


def _candles(n=5, base_price=100.0):
    return [
        OHLCV(timestamp=float(i), open=base_price, high=base_price + 1,
              low=base_price - 1, close=base_price + i, volume=10.0)
        for i in range(n)
    ]


class FetchMtfCandlesTests(unittest.TestCase):
    def test_returns_all_timeframes_with_correct_data(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = lambda symbol, tf, limit: _candles(n=3, base_price=hash(tf) % 100)

        mgr = MarketDataManager(exchange)
        result = mgr.fetch_mtf_candles("BTCUSD")

        self.assertEqual(set(result.keys()), set(TIMEFRAMES))
        for tf in TIMEFRAMES:
            self.assertEqual(len(result[tf]), 3)

    def test_skips_timeframe_that_returns_empty(self):
        def fetch(symbol, tf, limit):
            return [] if tf == "4h" else _candles(n=2)

        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = fetch

        mgr = MarketDataManager(exchange)
        result = mgr.fetch_mtf_candles("BTCUSD")

        self.assertNotIn("4h", result)
        self.assertEqual(len(result), len(TIMEFRAMES) - 1)

    def test_skips_timeframe_whose_fetch_raises(self):
        def fetch(symbol, tf, limit):
            if tf == "1h":
                raise ConnectionError("simulated network failure")
            return _candles(n=2)

        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = fetch

        mgr = MarketDataManager(exchange)
        result = mgr.fetch_mtf_candles("BTCUSD")  # must not raise

        self.assertNotIn("1h", result)
        self.assertEqual(len(result), len(TIMEFRAMES) - 1)

    def test_second_call_hits_cache_not_exchange(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = lambda symbol, tf, limit: _candles(n=2)

        mgr = MarketDataManager(exchange)
        mgr.fetch_mtf_candles("BTCUSD")
        call_count_after_first = exchange.fetch_ohlcv.call_count

        mgr.fetch_mtf_candles("BTCUSD")  # should be fully cached
        self.assertEqual(exchange.fetch_ohlcv.call_count, call_count_after_first)

    def test_fetches_actually_overlap_in_time(self):
        """
        Each timeframe fetch sleeps 100ms. If they ran serially, 6 timeframes
        would take >= 600ms. Running concurrently should take well under
        that — this is the actual behavior the optimization is for.
        """
        delay = 0.1

        def slow_fetch(symbol, tf, limit):
            time.sleep(delay)
            return _candles(n=2)

        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = slow_fetch

        mgr = MarketDataManager(exchange)
        start = time.monotonic()
        result = mgr.fetch_mtf_candles("BTCUSD")
        elapsed = time.monotonic() - start

        self.assertEqual(len(result), len(TIMEFRAMES))
        self.assertLess(elapsed, delay * len(TIMEFRAMES) * 0.6)

    def test_concurrent_calls_use_multiple_threads(self):
        seen_threads = set()
        lock = threading.Lock()

        def fetch(symbol, tf, limit):
            with lock:
                seen_threads.add(threading.get_ident())
            time.sleep(0.05)
            return _candles(n=1)

        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = fetch

        mgr = MarketDataManager(exchange)
        mgr.fetch_mtf_candles("BTCUSD")

        self.assertGreater(len(seen_threads), 1)


class GetCurrentPriceTests(unittest.TestCase):
    def test_uses_1m_close_when_available(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = _candles(n=3, base_price=100.0)

        mgr = MarketDataManager(exchange)
        price = mgr.get_current_price("BTCUSD")
        self.assertEqual(price, 102.0)  # last candle close = base(100) + i(2)

    def test_falls_back_to_ticker_when_no_candles(self):
        from exchanges.base import Ticker

        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = []
        exchange.fetch_ticker.return_value = Ticker(symbol="BTCUSD", bid=99, ask=101, last=100)

        mgr = MarketDataManager(exchange)
        price = mgr.get_current_price("BTCUSD")
        self.assertEqual(price, 100.0)  # mid of bid/ask


class CacheInvalidationTests(unittest.TestCase):
    def test_invalidate_specific_symbol(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = lambda symbol, tf, limit: _candles(n=1)

        mgr = MarketDataManager(exchange)
        mgr.fetch_mtf_candles("BTCUSD")
        mgr.fetch_mtf_candles("ETHUSD")
        count_before = exchange.fetch_ohlcv.call_count

        mgr.invalidate_cache("BTCUSD")
        mgr.fetch_mtf_candles("BTCUSD")   # re-fetched
        mgr.fetch_mtf_candles("ETHUSD")   # still cached

        self.assertEqual(exchange.fetch_ohlcv.call_count, count_before + len(TIMEFRAMES))

    def test_invalidate_all(self):
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = lambda symbol, tf, limit: _candles(n=1)

        mgr = MarketDataManager(exchange)
        mgr.fetch_mtf_candles("BTCUSD")
        mgr.invalidate_cache()
        mgr.fetch_mtf_candles("BTCUSD")

        self.assertEqual(exchange.fetch_ohlcv.call_count, len(TIMEFRAMES) * 2)


if __name__ == "__main__":
    unittest.main()
