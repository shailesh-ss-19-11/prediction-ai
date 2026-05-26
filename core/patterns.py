"""
Candlestick pattern detection with confidence scores.

Each detector analyses the last 1-3 rows of an OHLCV DataFrame and returns
a CandlePattern describing the pattern name, directional bias, and a
confidence score in [0.0, 1.0].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CandlePattern:
    name: str
    direction: str          # 'bullish' | 'bearish' | 'neutral'
    confidence: float       # 0.0 – 1.0
    description: str


# ---------------------------------------------------------------------------
# OHLC helper functions (operate on scalar values)
# ---------------------------------------------------------------------------

def body_size(open_: float, close: float) -> float:
    """Absolute size of the candle body."""
    return abs(close - open_)


def upper_wick(open_: float, high: float, close: float) -> float:
    """Length of the upper shadow (wick above the body)."""
    return high - max(open_, close)


def lower_wick(open_: float, low: float, close: float) -> float:
    """Length of the lower shadow (wick below the body)."""
    return min(open_, close) - low


def is_bullish(open_: float, close: float) -> bool:
    """True when the candle closes above the open (green candle)."""
    return close > open_


def is_bearish(open_: float, close: float) -> bool:
    """True when the candle closes below the open (red candle)."""
    return close < open_


def total_range(high: float, low: float) -> float:
    """High-to-low range of the candle."""
    return high - low


# ---------------------------------------------------------------------------
# Individual pattern detectors
# Each returns Optional[CandlePattern] – None when the pattern is not found.
# ---------------------------------------------------------------------------

def _detect_hammer(o: float, h: float, l: float, c: float) -> Optional[CandlePattern]:
    """
    Hammer (bullish reversal):
    - Small body situated near the top of the candle.
    - Lower wick >= 2x body.
    - Upper wick <= 0.3x body.
    Confidence scales with the lower-wick-to-body ratio.
    """
    rng = total_range(h, l)
    if rng == 0:
        return None
    bd  = body_size(o, c)
    lw  = lower_wick(o, l, c)
    uw  = upper_wick(o, h, c)

    if bd == 0:
        return None
    if lw < 2.0 * bd:
        return None
    if uw > 0.3 * bd:
        return None

    # Confidence: clamped ratio; 2× → 0.60, 4× → 0.80, 6× → 0.90+
    ratio = lw / bd
    confidence = float(np.clip(0.50 + (ratio - 2.0) * 0.08, 0.60, 0.95))
    return CandlePattern(
        name="Hammer",
        direction="bullish",
        confidence=round(confidence, 3),
        description=(
            f"Hammer: lower wick {lw:.4f} is {ratio:.1f}x the body "
            f"({bd:.4f}); upper wick {uw:.4f}."
        ),
    )


def _detect_shooting_star(o: float, h: float, l: float, c: float) -> Optional[CandlePattern]:
    """
    Shooting Star (bearish reversal):
    - Small body situated near the bottom of the candle.
    - Upper wick >= 2x body.
    - Lower wick <= 0.3x body.
    Confidence scales with the upper-wick-to-body ratio.
    """
    bd = body_size(o, c)
    uw = upper_wick(o, h, c)
    lw = lower_wick(o, l, c)

    if bd == 0:
        return None
    if uw < 2.0 * bd:
        return None
    if lw > 0.3 * bd:
        return None

    ratio = uw / bd
    confidence = float(np.clip(0.50 + (ratio - 2.0) * 0.08, 0.60, 0.95))
    return CandlePattern(
        name="Shooting Star",
        direction="bearish",
        confidence=round(confidence, 3),
        description=(
            f"Shooting Star: upper wick {uw:.4f} is {ratio:.1f}x the body "
            f"({bd:.4f}); lower wick {lw:.4f}."
        ),
    )


def _detect_bullish_engulfing(
    prev_o: float, prev_c: float,
    curr_o: float, curr_c: float,
) -> Optional[CandlePattern]:
    """
    Bullish Engulfing:
    - Previous candle is bearish (red).
    - Current candle is bullish (green) and its body fully covers the previous body.
    Confidence based on how much larger the current body is.
    """
    if not is_bearish(prev_o, prev_c):
        return None
    if not is_bullish(curr_o, curr_c):
        return None
    # Full engulfing: current open <= prev close AND current close >= prev open
    if curr_o > prev_c or curr_c < prev_o:
        return None

    prev_bd = body_size(prev_o, prev_c)
    curr_bd = body_size(curr_o, curr_c)
    if prev_bd == 0:
        return None

    size_ratio = curr_bd / prev_bd
    confidence = float(np.clip(0.55 + (size_ratio - 1.0) * 0.15, 0.60, 0.95))
    return CandlePattern(
        name="Bullish Engulfing",
        direction="bullish",
        confidence=round(confidence, 3),
        description=(
            f"Bullish Engulfing: current body {curr_bd:.4f} is "
            f"{size_ratio:.2f}x the previous body {prev_bd:.4f}."
        ),
    )


def _detect_bearish_engulfing(
    prev_o: float, prev_c: float,
    curr_o: float, curr_c: float,
) -> Optional[CandlePattern]:
    """
    Bearish Engulfing:
    - Previous candle is bullish (green).
    - Current candle is bearish (red) and its body fully covers the previous body.
    Confidence based on size ratio.
    """
    if not is_bullish(prev_o, prev_c):
        return None
    if not is_bearish(curr_o, curr_c):
        return None
    # Full engulfing: current open >= prev close AND current close <= prev open
    if curr_o < prev_c or curr_c > prev_o:
        return None

    prev_bd = body_size(prev_o, prev_c)
    curr_bd = body_size(curr_o, curr_c)
    if prev_bd == 0:
        return None

    size_ratio = curr_bd / prev_bd
    confidence = float(np.clip(0.55 + (size_ratio - 1.0) * 0.15, 0.60, 0.95))
    return CandlePattern(
        name="Bearish Engulfing",
        direction="bearish",
        confidence=round(confidence, 3),
        description=(
            f"Bearish Engulfing: current body {curr_bd:.4f} is "
            f"{size_ratio:.2f}x the previous body {prev_bd:.4f}."
        ),
    )


def _detect_doji(o: float, h: float, l: float, c: float) -> Optional[CandlePattern]:
    """
    Doji:
    - Body < 5% of the total range.
    Confidence inversely proportional to body-to-range ratio.
    """
    rng = total_range(h, l)
    if rng == 0:
        return None
    bd = body_size(o, c)
    body_ratio = bd / rng

    if body_ratio >= 0.05:
        return None

    # Confidence: 0% body → 1.0, 5% body → 0.65
    confidence = float(np.clip(1.0 - (body_ratio / 0.05) * 0.35, 0.65, 1.0))
    return CandlePattern(
        name="Doji",
        direction="neutral",
        confidence=round(confidence, 3),
        description=(
            f"Doji: body {bd:.4f} is only {body_ratio*100:.2f}% "
            f"of the range {rng:.4f}."
        ),
    )


def _detect_morning_star(
    c1_o: float, c1_c: float,
    c2_o: float, c2_h: float, c2_l: float, c2_c: float,
    c3_o: float, c3_c: float,
) -> Optional[CandlePattern]:
    """
    Morning Star (3-candle bullish reversal):
    1. First candle is bearish.
    2. Second candle is a doji or small-bodied candle.
    3. Third candle is bullish and closes above the midpoint of the first candle.
    Confidence range: 0.75 – 0.95.
    """
    if not is_bearish(c1_o, c1_c):
        return None
    if not is_bullish(c3_o, c3_c):
        return None

    # Middle candle must be small (body < 30% of its range)
    c2_rng = total_range(c2_h, c2_l)
    if c2_rng == 0:
        return None
    c2_bd = body_size(c2_o, c2_c)
    if c2_bd / c2_rng > 0.30:
        return None

    # Third candle must close above the midpoint of the first
    c1_mid = (c1_o + c1_c) / 2.0
    if c3_c <= c1_mid:
        return None

    # Score by how far above midpoint the third close lands
    c1_bd = body_size(c1_o, c1_c)
    penetration = (c3_c - c1_mid) / c1_bd if c1_bd > 0 else 0
    confidence = float(np.clip(0.75 + penetration * 0.10, 0.75, 0.95))
    return CandlePattern(
        name="Morning Star",
        direction="bullish",
        confidence=round(confidence, 3),
        description=(
            "Morning Star: bearish C1, small/doji C2, "
            f"bullish C3 closing {penetration*100:.1f}% above C1 midpoint."
        ),
    )


def _detect_evening_star(
    c1_o: float, c1_c: float,
    c2_o: float, c2_h: float, c2_l: float, c2_c: float,
    c3_o: float, c3_c: float,
) -> Optional[CandlePattern]:
    """
    Evening Star (3-candle bearish reversal):
    1. First candle is bullish.
    2. Second candle is a doji or small-bodied candle.
    3. Third candle is bearish and closes below the midpoint of the first candle.
    Confidence range: 0.75 – 0.95.
    """
    if not is_bullish(c1_o, c1_c):
        return None
    if not is_bearish(c3_o, c3_c):
        return None

    c2_rng = total_range(c2_h, c2_l)
    if c2_rng == 0:
        return None
    c2_bd = body_size(c2_o, c2_c)
    if c2_bd / c2_rng > 0.30:
        return None

    c1_mid = (c1_o + c1_c) / 2.0
    if c3_c >= c1_mid:
        return None

    c1_bd = body_size(c1_o, c1_c)
    penetration = (c1_mid - c3_c) / c1_bd if c1_bd > 0 else 0
    confidence = float(np.clip(0.75 + penetration * 0.10, 0.75, 0.95))
    return CandlePattern(
        name="Evening Star",
        direction="bearish",
        confidence=round(confidence, 3),
        description=(
            "Evening Star: bullish C1, small/doji C2, "
            f"bearish C3 closing {penetration*100:.1f}% below C1 midpoint."
        ),
    )


def _detect_pin_bar(
    o: float, h: float, l: float, c: float,
    prev_close: float,
) -> Optional[CandlePattern]:
    """
    Pin Bar:
    - The dominant wick is >= 2x the body.
    - The wick points away from the prevailing micro-trend
      (lower wick = bullish pin; upper wick = bearish pin).
    Confidence based on wick prominence relative to total range.
    """
    bd = body_size(o, c)
    uw = upper_wick(o, h, c)
    lw = lower_wick(o, l, c)
    rng = total_range(h, l)

    if bd == 0 or rng == 0:
        return None

    # Determine dominant wick
    if lw >= 2.0 * bd and lw > uw:
        # Bullish pin: long lower wick pointing down (reversal to upside)
        wick_ratio = lw / rng
        confidence = float(np.clip(0.50 + wick_ratio * 0.50, 0.55, 0.90))
        direction = "bullish"
        desc = (
            f"Bullish Pin Bar: lower wick {lw:.4f} "
            f"({wick_ratio*100:.1f}% of range)."
        )
    elif uw >= 2.0 * bd and uw > lw:
        # Bearish pin: long upper wick pointing up (reversal to downside)
        wick_ratio = uw / rng
        confidence = float(np.clip(0.50 + wick_ratio * 0.50, 0.55, 0.90))
        direction = "bearish"
        desc = (
            f"Bearish Pin Bar: upper wick {uw:.4f} "
            f"({wick_ratio*100:.1f}% of range)."
        )
    else:
        return None

    return CandlePattern(
        name="Pin Bar",
        direction=direction,
        confidence=round(confidence, 3),
        description=desc,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_all(df: pd.DataFrame) -> list[CandlePattern]:
    """
    Run all pattern detectors against the last 3 candles of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Must contain at least 1 row; 3 rows required for 3-candle patterns.

    Returns
    -------
    list[CandlePattern]
        All patterns detected (may be empty).  Sorted by confidence descending.
    """
    patterns: list[CandlePattern] = []

    if df is None or len(df) < 1:
        return patterns

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        logger.error("[Patterns] Missing columns: %s", required - set(df.columns))
        return patterns

    # Extract last three candles (c1=oldest, c3=most recent)
    n = len(df)
    c3 = df.iloc[-1]

    # -----------------------------------------------------------------------
    # Single-candle patterns (current candle only)
    # -----------------------------------------------------------------------
    for fn in (
        lambda: _detect_hammer(c3.open, c3.high, c3.low, c3.close),
        lambda: _detect_shooting_star(c3.open, c3.high, c3.low, c3.close),
        lambda: _detect_doji(c3.open, c3.high, c3.low, c3.close),
    ):
        try:
            result = fn()
            if result is not None:
                patterns.append(result)
        except Exception as exc:
            logger.debug("[Patterns] Single-candle detector error: %s", exc)

    # -----------------------------------------------------------------------
    # Two-candle patterns (require at least 2 candles)
    # -----------------------------------------------------------------------
    if n >= 2:
        c2 = df.iloc[-2]

        for fn in (
            lambda: _detect_bullish_engulfing(c2.open, c2.close, c3.open, c3.close),
            lambda: _detect_bearish_engulfing(c2.open, c2.close, c3.open, c3.close),
            lambda: _detect_pin_bar(c3.open, c3.high, c3.low, c3.close, float(c2.close)),
        ):
            try:
                result = fn()
                if result is not None:
                    patterns.append(result)
            except Exception as exc:
                logger.debug("[Patterns] Two-candle detector error: %s", exc)

    # -----------------------------------------------------------------------
    # Three-candle patterns (require at least 3 candles)
    # -----------------------------------------------------------------------
    if n >= 3:
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]

        for fn in (
            lambda: _detect_morning_star(
                c1.open, c1.close,
                c2.open, c2.high, c2.low, c2.close,
                c3.open, c3.close,
            ),
            lambda: _detect_evening_star(
                c1.open, c1.close,
                c2.open, c2.high, c2.low, c2.close,
                c3.open, c3.close,
            ),
        ):
            try:
                result = fn()
                if result is not None:
                    patterns.append(result)
            except Exception as exc:
                logger.debug("[Patterns] Three-candle detector error: %s", exc)

    # Sort by confidence descending so callers can easily pick the best
    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns


def get_best_pattern(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Return the highest-confidence pattern from detect_all, or None if no
    pattern was detected.
    """
    patterns = detect_all(df)
    return patterns[0] if patterns else None
