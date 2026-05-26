"""
Advanced market filters for signal quality control.

Filters:
  - SessionFilter       : trading session / liquidity window checks
  - VolatilityFilter    : news-spike and low-volume detection
  - MultiTimeframeFilter: trend alignment across timeframes
  - NewsFilter          : proximity to high-impact news release times
  - OvertradingFilter   : per-symbol daily signal cap
  - FundingRateFilter   : extreme funding-rate detection for crypto perpetuals
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import IndicatorResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionFilter
# ---------------------------------------------------------------------------

class SessionFilter:
    """
    Detect which trading sessions are currently active and whether we are
    inside a high-liquidity window.

    All times are in UTC.

    SESSIONS format: {name: (open_hour_utc, close_hour_utc)}
    Hours are [open, close) half-open intervals.
    """

    SESSIONS: dict[str, tuple[int, int]] = {
        "asian":    (0,  8),
        "london":   (7,  16),
        "new_york": (12, 21),
    }

    # London / NY overlap → highest liquidity
    _OVERLAP_START = 12
    _OVERLAP_END   = 16

    def _now_utc_hour_minute(self) -> tuple[int, int]:
        now = datetime.now(tz=timezone.utc)
        return now.hour, now.minute

    def is_active_session(self, session_name: str) -> bool:
        """Return True when *session_name* is currently active (UTC)."""
        session = self.SESSIONS.get(session_name.lower())
        if session is None:
            logger.warning("[SessionFilter] Unknown session: %s", session_name)
            return False
        hour, _ = self._now_utc_hour_minute()
        open_h, close_h = session
        if open_h < close_h:
            return open_h <= hour < close_h
        # Overnight session (not currently applicable, but kept for correctness)
        return hour >= open_h or hour < close_h

    def get_active_sessions(self) -> list[str]:
        """Return list of currently-active session names."""
        return [name for name in self.SESSIONS if self.is_active_session(name)]

    def is_high_liquidity_window(self) -> bool:
        """
        True during the London/New-York overlap (12:00–16:00 UTC).
        This window typically has the deepest order flow.
        """
        hour, _ = self._now_utc_hour_minute()
        return self._OVERLAP_START <= hour < self._OVERLAP_END


# ---------------------------------------------------------------------------
# VolatilityFilter
# ---------------------------------------------------------------------------

class VolatilityFilter:
    """
    Classify current market volatility relative to ATR.

    'extreme' : current range > 4x ATR  (flash crash / massive news spike)
    'high'    : current range > threshold_high x ATR  (default 3.0)
    'low'     : current range < threshold_low  x ATR  (default 0.3)
    'normal'  : everything else
    """

    def _current_range(self, df: pd.DataFrame) -> float:
        """High–Low range of the most-recent candle."""
        if df is None or df.empty:
            return float("nan")
        last = df.iloc[-1]
        return float(last["high"]) - float(last["low"])

    def is_too_volatile(
        self,
        df: pd.DataFrame,
        atr: float,
        threshold_multiplier: float = 3.0,
    ) -> bool:
        """
        True when the current candle range > *threshold_multiplier* x ATR.
        Typical use: avoid trading directly into news spikes.
        """
        if atr == 0 or math.isnan(atr):
            return False
        rng = self._current_range(df)
        if math.isnan(rng):
            return False
        return rng > threshold_multiplier * atr

    def is_too_quiet(
        self,
        df: pd.DataFrame,
        atr: float,
        threshold_multiplier: float = 0.3,
    ) -> bool:
        """
        True when the current candle range < *threshold_multiplier* x ATR.
        Typical use: avoid trading during very low-volume / holiday periods.
        """
        if atr == 0 or math.isnan(atr):
            return False
        rng = self._current_range(df)
        if math.isnan(rng):
            return False
        return rng < threshold_multiplier * atr

    def get_volatility_state(self, df: pd.DataFrame, atr: float) -> str:
        """
        Return one of: 'normal', 'high', 'low', 'extreme'.

        'extreme' is defined as range > 4x ATR (catches flash crashes / massive
        news events that exceed the standard 'high' threshold).
        """
        if atr == 0 or math.isnan(atr):
            return "normal"
        rng = self._current_range(df)
        if math.isnan(rng):
            return "normal"

        ratio = rng / atr
        if ratio > 4.0:
            return "extreme"
        if ratio > 3.0:
            return "high"
        if ratio < 0.3:
            return "low"
        return "normal"


# ---------------------------------------------------------------------------
# MultiTimeframeFilter
# ---------------------------------------------------------------------------

class MultiTimeframeFilter:
    """
    Multi-timeframe trend alignment checks.

    Expects indicator results keyed by timeframe strings:
        {"15m": IndicatorResult, "1h": IndicatorResult, "4h": IndicatorResult}
    """

    _TIMEFRAMES = ("15m", "1h", "4h")

    def confirm_trend(
        self,
        indicators_dict: dict[str, IndicatorResult],
    ) -> dict:
        """
        Check whether the trend direction is aligned across 15m, 1h, and 4h.

        Returns
        -------
        dict with keys:
            aligned           : bool
            direction         : str   – 'bullish' | 'bearish' | 'mixed'
            strength          : float – fraction of timeframes aligned (0–1)
            timeframes_aligned: list[str]
        """
        bullish_tfs: list[str] = []
        bearish_tfs: list[str] = []

        for tf in self._TIMEFRAMES:
            ind = indicators_dict.get(tf)
            if ind is None:
                continue
            if ind.trend_ema == "bullish":
                bullish_tfs.append(tf)
            elif ind.trend_ema == "bearish":
                bearish_tfs.append(tf)

        total = len([tf for tf in self._TIMEFRAMES if tf in indicators_dict])
        if total == 0:
            return {
                "aligned": False,
                "direction": "mixed",
                "strength": 0.0,
                "timeframes_aligned": [],
            }

        if len(bullish_tfs) > len(bearish_tfs):
            dominant_dir = "bullish"
            dominant_tfs = bullish_tfs
        elif len(bearish_tfs) > len(bullish_tfs):
            dominant_dir = "bearish"
            dominant_tfs = bearish_tfs
        else:
            dominant_dir = "mixed"
            dominant_tfs = []

        strength = round(len(dominant_tfs) / total, 3)
        aligned  = len(dominant_tfs) == total  # strict: all TFs must agree

        return {
            "aligned":            aligned,
            "direction":          dominant_dir,
            "strength":           strength,
            "timeframes_aligned": dominant_tfs,
        }

    def get_htf_bias(
        self,
        indicators_4h: IndicatorResult,
        structure_4h: dict,
    ) -> str:
        """
        Combine 4h EMA trend signal with 4h market-structure bias.

        Priority:
        1. If both EMA trend and structure bias agree → return that direction.
        2. If only one is available → return that one.
        3. Disagreement → 'neutral'.

        Returns
        -------
        'bullish' | 'bearish' | 'neutral'
        """
        ema_bias  = getattr(indicators_4h, "trend_ema", "neutral")
        struc_bias = structure_4h.get("bias", "ranging")

        # Normalise 'ranging' to 'neutral' for comparison
        if struc_bias == "ranging":
            struc_bias = "neutral"

        if ema_bias == struc_bias and ema_bias != "neutral":
            return ema_bias
        if ema_bias != "neutral" and struc_bias == "neutral":
            return ema_bias
        if struc_bias != "neutral" and ema_bias == "neutral":
            return struc_bias
        return "neutral"


# ---------------------------------------------------------------------------
# NewsFilter
# ---------------------------------------------------------------------------

class NewsFilter:
    """
    Rudimentary high-impact news time proximity check.

    Real-time economic-calendar data is provided by news_fetcher.py.
    This filter guards against entering around statistically common
    high-volatility UTC release times even when live data is unavailable.

    MAJOR_NEWS_HOURS : list of (hour, minute) UTC tuples
    """

    MAJOR_NEWS_HOURS: list[tuple[int, int]] = [
        (8,  30),   # UK data / EU open releases
        (14,  0),   # US pre-market data
        (14, 30),   # US economic data (NFP, CPI, etc.)
        (15,  0),   # US market open secondary releases
    ]

    def is_near_news_time(self, buffer_minutes: int = 30) -> bool:
        """
        Return True when the current UTC time is within *buffer_minutes* of
        any entry in MAJOR_NEWS_HOURS.

        Parameters
        ----------
        buffer_minutes : int – symmetric window around each news time (default 30)
        """
        now = datetime.now(tz=timezone.utc)
        now_minutes = now.hour * 60 + now.minute

        for hour, minute in self.MAJOR_NEWS_HOURS:
            news_minutes = hour * 60 + minute
            if abs(now_minutes - news_minutes) <= buffer_minutes:
                return True
        return False


# ---------------------------------------------------------------------------
# OvertradingFilter
# ---------------------------------------------------------------------------

class OvertradingFilter:
    """
    Prevent over-trading by capping the number of signals per symbol per day.

    The daily counter is reset either by calling reset_daily() explicitly or
    automatically when the UTC date changes between can_trade() calls.
    """

    def __init__(self, max_signals_per_day: int = 3) -> None:
        self._max = max_signals_per_day
        self._signals: dict[str, int] = defaultdict(int)
        self._last_reset_date: Optional[str] = None

    def _today_utc(self) -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    def _auto_reset_if_new_day(self) -> None:
        today = self._today_utc()
        if self._last_reset_date != today:
            self.reset_daily()
            self._last_reset_date = today

    def can_trade(self, symbol: str) -> bool:
        """
        Return True when the symbol has not yet reached its daily signal cap.
        Automatically resets counters at UTC midnight.
        """
        self._auto_reset_if_new_day()
        return self._signals[symbol.upper()] < self._max

    def record_signal(self, symbol: str) -> None:
        """Increment the daily signal counter for *symbol*."""
        self._auto_reset_if_new_day()
        self._signals[symbol.upper()] += 1

    def reset_daily(self) -> None:
        """Reset all counters.  Call at UTC midnight or on new trading day."""
        self._signals.clear()
        self._last_reset_date = self._today_utc()
        logger.debug("[OvertradingFilter] Daily counters reset.")

    def signals_today(self, symbol: str) -> int:
        """Return today's signal count for *symbol*."""
        self._auto_reset_if_new_day()
        return self._signals[symbol.upper()]


# ---------------------------------------------------------------------------
# FundingRateFilter
# ---------------------------------------------------------------------------

class FundingRateFilter:
    """
    Funding-rate based caution filter for perpetual futures.

    Typical 8-hour funding rate on major exchanges is ≈ 0.01% (0.0001).
    At 0.1% (0.001) the market is considered heavily skewed.

    Negative funding → longs pay shorts → market is short-heavy.
    Positive funding → shorts pay longs → market is long-heavy.
    """

    def is_funding_extreme(
        self,
        funding_rate: float,
        threshold: float = 0.001,
    ) -> bool:
        """
        Return True when abs(funding_rate) exceeds *threshold*.

        Default threshold = 0.001 (0.1%) — extreme funding caution zone.
        """
        return abs(funding_rate) > threshold

    def get_funding_bias(self, funding_rate: float) -> str:
        """
        Classify current funding-rate sentiment.

        Returns
        -------
        'long_heavy'  : funding > +0.0001  (longs pay shorts → crowded long)
        'short_heavy' : funding < -0.0001  (shorts pay longs → crowded short)
        'neutral'     : near zero
        """
        _neutral_band = 0.0001

        if funding_rate > _neutral_band:
            return "long_heavy"
        if funding_rate < -_neutral_band:
            return "short_heavy"
        return "neutral"
