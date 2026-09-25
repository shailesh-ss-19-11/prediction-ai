"""
Tests for core.advanced_filters — SessionFilter, VolatilityFilter,
OvertradingFilter, FundingRateFilter, MultiTimeframeFilter, NewsFilter.

Run: python -m unittest tests.test_advanced_filters -v
"""

import datetime as real_datetime
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.advanced_filters import (  # noqa: E402
    FundingRateFilter,
    MultiTimeframeFilter,
    NewsFilter,
    OvertradingFilter,
    SessionFilter,
    VolatilityFilter,
)
from core.indicators import IndicatorResult  # noqa: E402


def _fixed_utc(hour, minute=0, day=15):
    return real_datetime.datetime(2026, 1, day, hour, minute, tzinfo=real_datetime.timezone.utc)


class SessionFilterTests(unittest.TestCase):
    def setUp(self):
        self.sf = SessionFilter()

    @patch("core.advanced_filters.datetime")
    def test_asian_session_active(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(3)
        self.assertTrue(self.sf.is_active_session("asian"))
        self.assertFalse(self.sf.is_active_session("london"))

    @patch("core.advanced_filters.datetime")
    def test_london_ny_overlap_is_high_liquidity(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(13)
        self.assertTrue(self.sf.is_high_liquidity_window())
        sessions = self.sf.get_active_sessions()
        self.assertIn("london", sessions)
        self.assertIn("new_york", sessions)

    def test_unknown_session_name(self):
        self.assertFalse(self.sf.is_active_session("mars"))

    @patch("core.advanced_filters.datetime")
    def test_off_hours_no_active_session(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(23)
        self.assertEqual(self.sf.get_active_sessions(), [])


class VolatilityFilterTests(unittest.TestCase):
    def setUp(self):
        self.vf = VolatilityFilter()

    def _df(self, high, low):
        return pd.DataFrame({"high": [high], "low": [low]})

    def test_normal(self):
        df = self._df(102, 100)  # range=2, atr=1 -> ratio=2 -> normal
        self.assertEqual(self.vf.get_volatility_state(df, atr=1.0), "normal")

    def test_extreme(self):
        df = self._df(110, 100)  # range=10, atr=1 -> ratio=10 -> extreme
        self.assertEqual(self.vf.get_volatility_state(df, atr=1.0), "extreme")
        self.assertTrue(self.vf.is_too_volatile(df, atr=1.0))

    def test_high(self):
        df = self._df(103.5, 100)  # range=3.5, atr=1 -> ratio=3.5 -> high
        self.assertEqual(self.vf.get_volatility_state(df, atr=1.0), "high")

    def test_low(self):
        df = self._df(100.1, 100)  # range=0.1, atr=1 -> ratio=0.1 -> low
        self.assertEqual(self.vf.get_volatility_state(df, atr=1.0), "low")
        self.assertTrue(self.vf.is_too_quiet(df, atr=1.0))

    def test_zero_atr_defaults_normal(self):
        df = self._df(110, 100)
        self.assertEqual(self.vf.get_volatility_state(df, atr=0), "normal")
        self.assertFalse(self.vf.is_too_volatile(df, atr=0))

    def test_empty_df(self):
        self.assertEqual(self.vf.get_volatility_state(pd.DataFrame(), atr=1.0), "normal")


class OvertradingFilterTests(unittest.TestCase):
    def test_allows_until_cap(self):
        ot = OvertradingFilter(max_signals_per_day=2)
        self.assertTrue(ot.can_trade("BTCUSD"))
        ot.record_signal("BTCUSD")
        self.assertTrue(ot.can_trade("BTCUSD"))
        ot.record_signal("BTCUSD")
        self.assertFalse(ot.can_trade("BTCUSD"))

    def test_per_symbol_independent(self):
        ot = OvertradingFilter(max_signals_per_day=1)
        ot.record_signal("BTCUSD")
        self.assertFalse(ot.can_trade("BTCUSD"))
        self.assertTrue(ot.can_trade("ETHUSD"))

    def test_reset_daily_clears_counts(self):
        ot = OvertradingFilter(max_signals_per_day=1)
        ot.record_signal("BTCUSD")
        self.assertFalse(ot.can_trade("BTCUSD"))
        ot.reset_daily()
        self.assertTrue(ot.can_trade("BTCUSD"))

    @patch("core.advanced_filters.datetime")
    def test_auto_resets_on_new_utc_day(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(23, day=15)
        mock_dt.strftime = real_datetime.datetime.strftime
        ot = OvertradingFilter(max_signals_per_day=1)
        ot.record_signal("BTCUSD")
        self.assertFalse(ot.can_trade("BTCUSD"))

        mock_dt.now.return_value = _fixed_utc(1, day=16)
        self.assertTrue(ot.can_trade("BTCUSD"))  # new UTC day -> auto reset


class FundingRateFilterTests(unittest.TestCase):
    def setUp(self):
        self.fr = FundingRateFilter()

    def test_extreme_positive(self):
        self.assertTrue(self.fr.is_funding_extreme(0.002))

    def test_within_normal_range(self):
        self.assertFalse(self.fr.is_funding_extreme(0.0005))

    def test_bias_classification(self):
        self.assertEqual(self.fr.get_funding_bias(0.0002), "long_heavy")
        self.assertEqual(self.fr.get_funding_bias(-0.0002), "short_heavy")
        self.assertEqual(self.fr.get_funding_bias(0.00001), "neutral")


class MultiTimeframeFilterTests(unittest.TestCase):
    def setUp(self):
        self.mtf = MultiTimeframeFilter()

    def test_fully_aligned_bullish(self):
        result = self.mtf.confirm_trend({
            "15m": IndicatorResult(trend_ema="bullish"),
            "1h":  IndicatorResult(trend_ema="bullish"),
            "4h":  IndicatorResult(trend_ema="bullish"),
        })
        self.assertTrue(result["aligned"])
        self.assertEqual(result["direction"], "bullish")
        self.assertEqual(result["strength"], 1.0)

    def test_mixed_not_aligned(self):
        result = self.mtf.confirm_trend({
            "15m": IndicatorResult(trend_ema="bullish"),
            "1h":  IndicatorResult(trend_ema="bearish"),
            "4h":  IndicatorResult(trend_ema="neutral"),
        })
        self.assertFalse(result["aligned"])

    def test_no_data_returns_mixed(self):
        result = self.mtf.confirm_trend({})
        self.assertEqual(result["direction"], "mixed")
        self.assertEqual(result["strength"], 0.0)

    def test_htf_bias_agreement(self):
        ind = IndicatorResult(trend_ema="bullish")
        bias = self.mtf.get_htf_bias(ind, {"bias": "bullish"})
        self.assertEqual(bias, "bullish")

    def test_htf_bias_disagreement_is_neutral(self):
        ind = IndicatorResult(trend_ema="bullish")
        bias = self.mtf.get_htf_bias(ind, {"bias": "bearish"})
        self.assertEqual(bias, "neutral")

    def test_htf_bias_ranging_normalised(self):
        ind = IndicatorResult(trend_ema="neutral")
        bias = self.mtf.get_htf_bias(ind, {"bias": "ranging"})
        self.assertEqual(bias, "neutral")


class NewsFilterTests(unittest.TestCase):
    def setUp(self):
        self.nf = NewsFilter()

    @patch("core.advanced_filters.datetime")
    def test_near_news_time(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(14, 25)  # 5 min before 14:30 release
        self.assertTrue(self.nf.is_near_news_time(buffer_minutes=30))

    @patch("core.advanced_filters.datetime")
    def test_far_from_news_time(self, mock_dt):
        mock_dt.now.return_value = _fixed_utc(20, 0)
        self.assertFalse(self.nf.is_near_news_time(buffer_minutes=30))


if __name__ == "__main__":
    unittest.main()
