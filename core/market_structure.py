"""
Market structure analysis: swing points, BOS / CHOCH detection, and
market bias determination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    price: float
    index: int                              # positional index into the DataFrame
    swing_type: str = "high"               # 'high' | 'low' | 'HH' | 'HL' | 'LH' | 'LL'


@dataclass
class StructureEvent:
    event_type: str                         # 'BOS' | 'CHOCH'
    direction: str                          # 'bullish' | 'bearish'
    price: float                            # price level that was broken
    index: int                              # bar index where the break occurred
    description: str = ""


# ---------------------------------------------------------------------------
# Swing-point detection
# ---------------------------------------------------------------------------

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    """
    Detect swing highs and lows using a rolling-window pivot approach.

    A swing high at bar *i* requires:
        df.high[i] == max(df.high[i-lookback : i+lookback+1])
    A swing low at bar *i* requires:
        df.low[i]  == min(df.low[i-lookback  : i+lookback+1])

    Parameters
    ----------
    df       : OHLCV DataFrame
    lookback : number of bars on each side to compare (default 5)

    Returns
    -------
    list[SwingPoint] sorted by index ascending
    """
    if df is None or len(df) < 2 * lookback + 1:
        return []

    swing_points: list[SwingPoint] = []
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback  : i + lookback + 1]

        if highs[i] == np.max(window_h) and not np.isnan(highs[i]):
            swing_points.append(SwingPoint(price=float(highs[i]), index=i, swing_type="high"))

        if lows[i] == np.min(window_l) and not np.isnan(lows[i]):
            swing_points.append(SwingPoint(price=float(lows[i]), index=i, swing_type="low"))

    # Sort by index; remove exact duplicates (same index could be both high and low
    # only in degenerate data, but guard anyway)
    swing_points.sort(key=lambda s: (s.index, s.swing_type))
    return swing_points


# ---------------------------------------------------------------------------
# Swing classification (HH / HL / LH / LL)
# ---------------------------------------------------------------------------

def classify_swings(swing_points: list[SwingPoint]) -> list[SwingPoint]:
    """
    Classify each swing point as HH / HL / LH / LL based on comparison with
    the previous swing of the same type (high vs. high, low vs. low).

    Parameters
    ----------
    swing_points : output of find_swing_points (mixed highs and lows)

    Returns
    -------
    New list of SwingPoint with swing_type set to 'HH' | 'HL' | 'LH' | 'LL'
    (unclassifiable first swings are dropped from the classified output but
    a copy with the raw type is kept so callers always receive all points).
    """
    classified: list[SwingPoint] = []

    highs = [sp for sp in swing_points if sp.swing_type in ("high", "HH", "LH")]
    lows  = [sp for sp in swing_points if sp.swing_type in ("low",  "HL", "LL")]

    def _classify_sequence(seq: list[SwingPoint], is_high: bool) -> list[SwingPoint]:
        result: list[SwingPoint] = []
        prev: Optional[SwingPoint] = None
        for sp in seq:
            if prev is None:
                # Cannot classify the very first point; keep raw type
                result.append(SwingPoint(price=sp.price, index=sp.index,
                                         swing_type="high" if is_high else "low"))
                prev = sp
                continue
            if is_high:
                label = "HH" if sp.price > prev.price else "LH"
            else:
                label = "HL" if sp.price > prev.price else "LL"
            result.append(SwingPoint(price=sp.price, index=sp.index, swing_type=label))
            prev = sp
        return result

    classified_highs = _classify_sequence(highs, is_high=True)
    classified_lows  = _classify_sequence(lows,  is_high=False)

    # Merge and sort by index
    all_classified = classified_highs + classified_lows
    all_classified.sort(key=lambda s: s.index)
    return all_classified


# ---------------------------------------------------------------------------
# BOS / CHOCH detection
# ---------------------------------------------------------------------------

def detect_bos(
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
) -> list[StructureEvent]:
    """
    Detect Break-of-Structure (BOS) and Change-of-Character (CHOCH) events.

    Rules
    -----
    - Infer the prevailing trend from the classified swing sequence
      (HH+HL = bullish; LH+LL = bearish).
    - BOS   : price breaks the most-recent swing high *in the direction* of the
              existing trend (bullish BOS), or breaks the most-recent swing low
              in a bearish trend (bearish BOS).  Trend continuation.
    - CHOCH : price breaks the most-recent swing high *against* the existing
              bearish trend (potential reversal to bullish), or breaks the
              most-recent swing low against a bullish trend (reversal to bearish).

    Parameters
    ----------
    df           : OHLCV DataFrame (close prices used)
    swing_points : classified swing points from classify_swings()

    Returns
    -------
    list[StructureEvent] sorted by index ascending
    """
    events: list[StructureEvent] = []
    if not swing_points or df is None or df.empty:
        return events

    classified = classify_swings(swing_points)
    if not classified:
        return events

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    # Build a working list of the last few swings to infer current trend
    # and to track the levels that can be broken
    recent_highs = [sp for sp in classified if sp.swing_type in ("HH", "LH", "high")]
    recent_lows  = [sp for sp in classified if sp.swing_type in ("HL", "LL", "low")]

    if not recent_highs or not recent_lows:
        return events

    def _infer_trend(classified_pts: list[SwingPoint]) -> str:
        """Return 'bullish', 'bearish', or 'ranging'."""
        last4 = classified_pts[-4:] if len(classified_pts) >= 4 else classified_pts
        hh = sum(1 for s in last4 if s.swing_type == "HH")
        hl = sum(1 for s in last4 if s.swing_type == "HL")
        lh = sum(1 for s in last4 if s.swing_type == "LH")
        ll = sum(1 for s in last4 if s.swing_type == "LL")
        if hh + hl > lh + ll:
            return "bullish"
        if lh + ll > hh + hl:
            return "bearish"
        return "ranging"

    trend = _infer_trend(classified)

    # Walk through each bar that comes *after* the last swing point and
    # check for breaks
    last_swing_idx = max(sp.index for sp in classified)

    # Determine the most-recent meaningful levels to watch
    last_swing_high = recent_highs[-1] if recent_highs else None
    last_swing_low  = recent_lows[-1]  if recent_lows  else None

    checked_high_level: Optional[float] = last_swing_high.price if last_swing_high else None
    checked_low_level:  Optional[float] = last_swing_low.price  if last_swing_low  else None
    checked_high_src_idx = last_swing_high.index if last_swing_high else -1
    checked_low_src_idx  = last_swing_low.index  if last_swing_low  else -1

    for i in range(last_swing_idx + 1, n):
        bar_close = closes[i]
        bar_high  = highs[i]
        bar_low   = lows[i]

        # ----- Bullish break: close above the last swing high -----
        if checked_high_level is not None and bar_close > checked_high_level:
            if trend == "bullish":
                event_type = "BOS"
                direction  = "bullish"
                desc = (
                    f"Bullish BOS at bar {i}: close {bar_close:.4f} broke "
                    f"swing high {checked_high_level:.4f} (from bar {checked_high_src_idx})."
                )
            else:
                event_type = "CHOCH"
                direction  = "bullish"
                desc = (
                    f"Bullish CHOCH at bar {i}: close {bar_close:.4f} broke "
                    f"swing high {checked_high_level:.4f} against bearish trend."
                )
            events.append(StructureEvent(
                event_type=event_type,
                direction=direction,
                price=checked_high_level,
                index=i,
                description=desc,
            ))
            # Update level for the next potential break
            checked_high_level   = bar_high
            checked_high_src_idx = i
            if event_type == "CHOCH":
                trend = "bullish"

        # ----- Bearish break: close below the last swing low -----
        if checked_low_level is not None and bar_close < checked_low_level:
            if trend == "bearish":
                event_type = "BOS"
                direction  = "bearish"
                desc = (
                    f"Bearish BOS at bar {i}: close {bar_close:.4f} broke "
                    f"swing low {checked_low_level:.4f} (from bar {checked_low_src_idx})."
                )
            else:
                event_type = "CHOCH"
                direction  = "bearish"
                desc = (
                    f"Bearish CHOCH at bar {i}: close {bar_close:.4f} broke "
                    f"swing low {checked_low_level:.4f} against bullish trend."
                )
            events.append(StructureEvent(
                event_type=event_type,
                direction=direction,
                price=checked_low_level,
                index=i,
                description=desc,
            ))
            checked_low_level   = bar_low
            checked_low_src_idx = i
            if event_type == "CHOCH":
                trend = "bearish"

    return events


# ---------------------------------------------------------------------------
# Market bias
# ---------------------------------------------------------------------------

def get_market_bias(swing_points: list[SwingPoint]) -> str:
    """
    Determine overall market bias from the last 4 classified swing points.

    Rules
    -----
    - If the last 4 swings are predominantly HH + HL → 'bullish'
    - If the last 4 swings are predominantly LH + LL → 'bearish'
    - Otherwise → 'ranging'

    Parameters
    ----------
    swing_points : list of classified SwingPoint (output of classify_swings)

    Returns
    -------
    'bullish' | 'bearish' | 'ranging'
    """
    if not swing_points:
        return "ranging"

    last4 = swing_points[-4:] if len(swing_points) >= 4 else swing_points

    bullish_count = sum(1 for s in last4 if s.swing_type in ("HH", "HL"))
    bearish_count = sum(1 for s in last4 if s.swing_type in ("LH", "LL"))

    if bullish_count > bearish_count:
        return "bullish"
    if bearish_count > bullish_count:
        return "bearish"
    return "ranging"


# ---------------------------------------------------------------------------
# Trend strength helper
# ---------------------------------------------------------------------------

def _trend_strength(swing_points: list[SwingPoint]) -> float:
    """
    Return a normalised trend strength score in [0, 1].

    Uses the ratio of aligned swings (HH+HL or LH+LL) vs. total classified
    swings across the last 8 swings.
    """
    if not swing_points:
        return 0.0

    last8 = swing_points[-8:] if len(swing_points) >= 8 else swing_points
    total = len(last8)
    if total == 0:
        return 0.0

    bullish_count = sum(1 for s in last8 if s.swing_type in ("HH", "HL"))
    bearish_count = sum(1 for s in last8 if s.swing_type in ("LH", "LL"))
    dominant = max(bullish_count, bearish_count)
    return round(dominant / total, 3)


# ---------------------------------------------------------------------------
# Public summary
# ---------------------------------------------------------------------------

def get_structure_summary(df: pd.DataFrame) -> dict:
    """
    Full market-structure summary for a single timeframe DataFrame.

    Returns
    -------
    dict with keys:
        bias            : str  – 'bullish' | 'bearish' | 'ranging'
        last_bos        : Optional[StructureEvent]
        last_choch      : Optional[StructureEvent]
        swing_points    : list[SwingPoint]  – last 6 classified swings
        trend_strength  : float  – 0.0 to 1.0
    """
    result: dict = {
        "bias": "ranging",
        "last_bos": None,
        "last_choch": None,
        "swing_points": [],
        "trend_strength": 0.0,
    }

    if df is None or df.empty:
        return result

    raw_swings  = find_swing_points(df)
    class_swings = classify_swings(raw_swings)
    events       = detect_bos(df, raw_swings)

    bias     = get_market_bias(class_swings)
    strength = _trend_strength(class_swings)

    last_bos   = next((e for e in reversed(events) if e.event_type == "BOS"),   None)
    last_choch = next((e for e in reversed(events) if e.event_type == "CHOCH"), None)

    result["bias"]           = bias
    result["last_bos"]       = last_bos
    result["last_choch"]     = last_choch
    result["swing_points"]   = class_swings[-6:] if len(class_swings) > 6 else class_swings
    result["trend_strength"] = strength
    return result
