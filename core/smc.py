"""
Smart Money Concepts (SMC) implementation.

Detects:
  - Order Blocks  (bullish / bearish)
  - Fair Value Gaps (FVGs)
  - Liquidity Sweeps (buy-side / sell-side)
  - Stop Hunts
  - Institutional Zones (combined OB + FVG areas)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.market_structure import SwingPoint

logger = logging.getLogger(__name__)

# Minimum ATR multiplier to classify a move as "strong"
_STRONG_MOVE_ATR_MULT = 1.5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    price_high: float
    price_low: float
    direction: str          # 'bullish' | 'bearish'
    strength: float         # 0.0 – 1.0
    index: int              # bar index in the source DataFrame
    tested: bool = False    # True once price re-enters the OB zone


@dataclass
class FairValueGap:
    price_high: float
    price_low: float
    direction: str          # 'bullish' | 'bearish'
    index: int              # bar index of the middle candle
    filled: bool = False


@dataclass
class LiquiditySweep:
    price: float
    direction: str          # 'buy_side' | 'sell_side'
    index: int
    confirmed: bool = False  # True when candle closes back through swept level


@dataclass
class InstitutionalZone:
    price_high: float
    price_low: float
    zone_type: str          # e.g. 'OB', 'FVG', 'OB+FVG'
    strength: float         # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Internal helper: ATR (Wilder)
# ---------------------------------------------------------------------------

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR series; returns NaN where insufficient data."""
    high      = df["high"].astype(float)
    low       = df["low"].astype(float)
    close     = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Order Block detection
# ---------------------------------------------------------------------------

def detect_order_blocks(df: pd.DataFrame) -> list[OrderBlock]:
    """
    Identify bullish and bearish order blocks.

    - Bullish OB  : the last *bearish* candle immediately before a strong
                    bullish move (move size >= 1.5x ATR).
    - Bearish OB  : the last *bullish* candle immediately before a strong
                    bearish move.
    - Strength    : (move_size / ATR) / 5.0, clamped to [0.20, 1.0].
    - A block is marked 'tested' if a later candle's low (bullish OB) or
      high (bearish OB) re-enters the OB price range.

    Parameters
    ----------
    df : OHLCV DataFrame with at least 30 rows for meaningful ATR.

    Returns
    -------
    list[OrderBlock], sorted by index ascending.
    """
    order_blocks: list[OrderBlock] = []

    if df is None or len(df) < 5:
        return order_blocks

    atr_series = _calculate_atr(df)
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    atrs   = atr_series.values.astype(float)
    n      = len(df)

    for i in range(1, n - 1):
        atr = atrs[i]
        if np.isnan(atr) or atr == 0:
            continue

        # --- Bullish OB: candle i is bearish; candle i+1 is a strong bull move ---
        if closes[i] < opens[i]:                         # candle i is bearish
            move = closes[i + 1] - opens[i + 1]          # next candle body
            if move >= _STRONG_MOVE_ATR_MULT * atr:      # strong bullish move
                strength = float(np.clip(move / atr / 5.0, 0.20, 1.0))
                ob = OrderBlock(
                    price_high=float(max(opens[i], closes[i])),
                    price_low=float(min(opens[i], closes[i])),
                    direction="bullish",
                    strength=round(strength, 3),
                    index=i,
                    tested=False,
                )
                order_blocks.append(ob)

        # --- Bearish OB: candle i is bullish; candle i+1 is a strong bear move ---
        if closes[i] > opens[i]:                         # candle i is bullish
            move = opens[i + 1] - closes[i + 1]          # next candle bearish body
            if move >= _STRONG_MOVE_ATR_MULT * atr:
                strength = float(np.clip(move / atr / 5.0, 0.20, 1.0))
                ob = OrderBlock(
                    price_high=float(max(opens[i], closes[i])),
                    price_low=float(min(opens[i], closes[i])),
                    direction="bearish",
                    strength=round(strength, 3),
                    index=i,
                    tested=False,
                )
                order_blocks.append(ob)

    # Mark tested order blocks
    for ob in order_blocks:
        for j in range(ob.index + 2, n):
            if ob.direction == "bullish":
                if lows[j] <= ob.price_high and lows[j] >= ob.price_low:
                    ob.tested = True
                    break
            else:  # bearish
                if highs[j] >= ob.price_low and highs[j] <= ob.price_high:
                    ob.tested = True
                    break

    return order_blocks


# ---------------------------------------------------------------------------
# Fair Value Gap detection
# ---------------------------------------------------------------------------

def detect_fair_value_gaps(df: pd.DataFrame) -> list[FairValueGap]:
    """
    Detect Fair Value Gaps (FVGs / imbalances).

    A 3-candle FVG exists at bar *i* when:
    - Bullish FVG : candle[i-1].high < candle[i+1].low
                    (gap between C1 high and C3 low — unfilled demand zone)
    - Bearish FVG : candle[i-1].low  > candle[i+1].high
                    (gap between C1 low and C3 high — unfilled supply zone)

    Filled condition:
    - Bullish FVG : a later candle's close dips below fvg.price_low
    - Bearish FVG : a later candle's close rises above fvg.price_high

    Only *unfilled* FVGs are returned.

    Parameters
    ----------
    df : OHLCV DataFrame

    Returns
    -------
    list[FairValueGap] — unfilled only, sorted by index ascending.
    """
    fvgs: list[FairValueGap] = []

    if df is None or len(df) < 3:
        return fvgs

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    for i in range(1, n - 1):
        # Bullish FVG
        if highs[i - 1] < lows[i + 1]:
            fvg = FairValueGap(
                price_low=float(highs[i - 1]),
                price_high=float(lows[i + 1]),
                direction="bullish",
                index=i,
                filled=False,
            )
            fvgs.append(fvg)

        # Bearish FVG
        elif lows[i - 1] > highs[i + 1]:
            fvg = FairValueGap(
                price_high=float(lows[i - 1]),
                price_low=float(highs[i + 1]),
                direction="bearish",
                index=i,
                filled=False,
            )
            fvgs.append(fvg)

    # Mark filled FVGs
    for fvg in fvgs:
        for j in range(fvg.index + 2, n):
            if fvg.direction == "bullish" and closes[j] < fvg.price_low:
                fvg.filled = True
                break
            if fvg.direction == "bearish" and closes[j] > fvg.price_high:
                fvg.filled = True
                break

    # Return only unfilled gaps
    return [f for f in fvgs if not f.filled]


# ---------------------------------------------------------------------------
# Liquidity Sweep detection
# ---------------------------------------------------------------------------

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
) -> list[LiquiditySweep]:
    """
    Detect liquidity sweeps at swing-point levels.

    - Buy-side sweep  : a candle's high exceeds a prior swing high then the
                        candle closes back below that level.
    - Sell-side sweep : a candle's low undercuts a prior swing low then the
                        candle closes back above that level.
    - confirmed       : True when the candle closes back through the swept level
                        (the standard definition; same condition used above).

    Parameters
    ----------
    df           : OHLCV DataFrame
    swing_points : raw or classified swing points

    Returns
    -------
    list[LiquiditySweep] sorted by index ascending.
    """
    sweeps: list[LiquiditySweep] = []

    if df is None or df.empty or not swing_points:
        return sweeps

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    swing_highs = sorted(
        [sp for sp in swing_points if sp.swing_type in ("high", "HH", "LH")],
        key=lambda s: s.index,
    )
    swing_lows = sorted(
        [sp for sp in swing_points if sp.swing_type in ("low", "HL", "LL")],
        key=lambda s: s.index,
    )

    for i in range(1, n):
        # --- Buy-side liquidity sweep ---
        for sh in swing_highs:
            if sh.index >= i:
                continue  # only look at prior swing points
            level = sh.price
            if highs[i] > level and closes[i] <= level:
                confirmed = closes[i] < level
                sweeps.append(LiquiditySweep(
                    price=level,
                    direction="buy_side",
                    index=i,
                    confirmed=confirmed,
                ))
                break  # one sweep per bar per type is enough

        # --- Sell-side liquidity sweep ---
        for sl in swing_lows:
            if sl.index >= i:
                continue
            level = sl.price
            if lows[i] < level and closes[i] >= level:
                confirmed = closes[i] > level
                sweeps.append(LiquiditySweep(
                    price=level,
                    direction="sell_side",
                    index=i,
                    confirmed=confirmed,
                ))
                break

    return sweeps


# ---------------------------------------------------------------------------
# Stop Hunt detection
# ---------------------------------------------------------------------------

def detect_stop_hunts(df: pd.DataFrame) -> list[dict]:
    """
    Detect stop-hunt / wick-spike bars.

    A stop hunt is identified when a candle's wick extends beyond the recent
    high or low (lookback=20 bars, excluding the current bar) by more than
    0.5x ATR and the candle *closes back* inside the prior range.

    Returns
    -------
    list[dict] each with keys:
        price         : float   – level that was spiked through
        direction     : str     – 'sell_side' (spike below) | 'buy_side' (spike above)
        index         : int
        atr_extension : float   – how many ATRs the wick extended beyond the level
    """
    stop_hunts: list[dict] = []

    if df is None or len(df) < 22:
        return stop_hunts

    atr_series = _calculate_atr(df)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    opens  = df["open"].values.astype(float)
    atrs   = atr_series.values.astype(float)
    n      = len(df)
    lookback = 20

    for i in range(lookback, n):
        atr = atrs[i]
        if np.isnan(atr) or atr == 0:
            continue

        prior_high = np.max(highs[i - lookback : i])
        prior_low  = np.min(lows[i - lookback  : i])

        # Sell-side stop hunt: wick below prior low, close back above
        if lows[i] < prior_low and closes[i] > prior_low:
            extension = (prior_low - lows[i]) / atr
            if extension >= 0.5:
                stop_hunts.append({
                    "price": float(prior_low),
                    "direction": "sell_side",
                    "index": i,
                    "atr_extension": round(float(extension), 3),
                })

        # Buy-side stop hunt: wick above prior high, close back below
        if highs[i] > prior_high and closes[i] < prior_high:
            extension = (highs[i] - prior_high) / atr
            if extension >= 0.5:
                stop_hunts.append({
                    "price": float(prior_high),
                    "direction": "buy_side",
                    "index": i,
                    "atr_extension": round(float(extension), 3),
                })

    return stop_hunts


# ---------------------------------------------------------------------------
# Institutional Zone construction
# ---------------------------------------------------------------------------

def get_institutional_zones(
    order_blocks: list[OrderBlock],
    fair_value_gaps: list[FairValueGap],
) -> list[InstitutionalZone]:
    """
    Combine overlapping Order Blocks and Fair Value Gaps into Institutional Zones.

    Overlap check : two zones overlap when one's high > the other's low AND
                    one's low < the other's high (standard interval overlap).

    Strength rules
    --------------
    - OB alone        : ob.strength
    - FVG alone       : 0.50
    - OB + FVG overlap: min(1.0, ob.strength + 0.25)   (bonus for confluence)

    Parameters
    ----------
    order_blocks    : list[OrderBlock]
    fair_value_gaps : list[FairValueGap] (unfilled)

    Returns
    -------
    list[InstitutionalZone]
    """
    zones: list[InstitutionalZone] = []

    def _overlaps(a_hi: float, a_lo: float, b_hi: float, b_lo: float) -> bool:
        return a_hi > b_lo and a_lo < b_hi

    def _merge(a_hi: float, a_lo: float, b_hi: float, b_lo: float):
        return max(a_hi, b_hi), min(a_lo, b_lo)

    # Track which FVGs have been merged
    merged_fvg_indices: set[int] = set()

    for ob in order_blocks:
        combined_high = ob.price_high
        combined_low  = ob.price_low
        has_fvg_overlap = False

        for idx_fvg, fvg in enumerate(fair_value_gaps):
            if fvg.direction != ob.direction:
                continue
            if _overlaps(ob.price_high, ob.price_low, fvg.price_high, fvg.price_low):
                combined_high, combined_low = _merge(
                    combined_high, combined_low, fvg.price_high, fvg.price_low
                )
                merged_fvg_indices.add(idx_fvg)
                has_fvg_overlap = True

        zone_type = "OB+FVG" if has_fvg_overlap else "OB"
        strength  = min(1.0, ob.strength + 0.25) if has_fvg_overlap else ob.strength

        zones.append(InstitutionalZone(
            price_high=round(combined_high, 8),
            price_low=round(combined_low, 8),
            zone_type=zone_type,
            strength=round(strength, 3),
        ))

    # Add standalone FVGs that were not merged into an OB zone
    for idx_fvg, fvg in enumerate(fair_value_gaps):
        if idx_fvg in merged_fvg_indices:
            continue
        zones.append(InstitutionalZone(
            price_high=fvg.price_high,
            price_low=fvg.price_low,
            zone_type="FVG",
            strength=0.50,
        ))

    # Sort by strength descending
    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones
