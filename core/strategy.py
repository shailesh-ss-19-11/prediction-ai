"""
Trend-following strategy with Buy/Sell signal generation.

Evaluates multi-timeframe indicator alignment, candlestick pattern
confirmation, volume spikes, market structure, and SMC confluence
to produce a TradeSetup with entry, SL, TP1/2/3, and R:R.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import IndicatorResult
from core.market_structure import find_swing_points
from core.patterns import CandlePattern
from core.smc import OrderBlock, FairValueGap, LiquiditySweep

logger = logging.getLogger(__name__)

# Minimum conditions that must pass (out of 7) to emit a signal
_MIN_CONDITIONS = 6

# ATR multiplier for the minimum SL distance from entry AND for the buffer
# placed below/above the structural swing point.
_SL_ATR_MULT = 1.5

# Minimum SL distance as a fraction of entry price (0.8% floor for crypto).
_MIN_SL_PCT = 0.008

# Price must be within this fraction of a key swing level to allow entry.
# Trades fired from the middle of a range are blocked.
_PROXIMITY_FILTER_PCT = 0.005

# How many 15m bars to look back for the structural swing low/high used as SL.
_SWING_SL_LOOKBACK = 20

# Minimum acceptable risk-to-reward ratio
_MIN_RR = 2.0

# Candle lookback for volume average
_VOLUME_LOOKBACK = 20

# Volume spike threshold
_VOLUME_SPIKE_MULT = 1.5

# Minimum 4h trend_strength (0-1) required before accepting the structure bias
_MIN_TREND_STRENGTH = 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TradeSetup:
    symbol: str
    direction: str              # 'LONG' | 'SHORT'
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr: float                   # risk-to-reward (TP1 based)
    confidence: float           # conditions_passed / 7
    reasons: list[str] = field(default_factory=list)
    pattern: Optional[CandlePattern] = None
    timeframe: str = "15m"
    structure_bias: str = "neutral"
    smc_confluence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: volume spike check
# ---------------------------------------------------------------------------

def _has_volume_spike(df_15m: Optional[pd.DataFrame], direction: str = "") -> bool:
    """
    Return True when the last COMPLETED candle has a volume spike in the trade direction.

    Uses iloc[-2] (previous closed candle) — the current candle is still forming.
    Also checks that the spike candle's close confirms direction (bearish candle for SHORT,
    bullish candle for LONG) so a reversal-bounce spike doesn't pass for the wrong side.
    """
    if df_15m is None or len(df_15m) < _VOLUME_LOOKBACK + 2:
        return False
    vol    = df_15m["volume"].astype(float)
    recent = float(vol.iloc[-2])
    avg    = float(vol.iloc[-(_VOLUME_LOOKBACK + 2):-2].mean())
    if avg == 0:
        return False
    if recent <= _VOLUME_SPIKE_MULT * avg:
        return False

    # Volume is high — verify the candle closed in the trade direction
    if direction:
        candle = df_15m.iloc[-2]
        if direction == "LONG"  and float(candle["close"]) < float(candle["open"]):
            return False   # spike on a bearish candle — don't use for LONG entry
        if direction == "SHORT" and float(candle["close"]) > float(candle["open"]):
            return False   # spike on a bullish candle — don't use for SHORT entry

    return True


# ---------------------------------------------------------------------------
# Helper: ATR from last bar (scalar)
# ---------------------------------------------------------------------------

def _current_atr(indicators: IndicatorResult) -> float:
    atr = indicators.atr_14
    return atr if (atr and not math.isnan(atr)) else 0.0


# ---------------------------------------------------------------------------
# Helper: current close price
# ---------------------------------------------------------------------------

def _current_close(df_15m: Optional[pd.DataFrame]) -> float:
    if df_15m is None or df_15m.empty:
        return float("nan")
    return float(df_15m["close"].iloc[-1])


def _current_high(df_15m: Optional[pd.DataFrame]) -> float:
    if df_15m is None or df_15m.empty:
        return float("nan")
    return float(df_15m["high"].iloc[-1])


def _current_low(df_15m: Optional[pd.DataFrame]) -> float:
    if df_15m is None or df_15m.empty:
        return float("nan")
    return float(df_15m["low"].iloc[-1])


# ---------------------------------------------------------------------------
# Helper: structural swing low / high for SL placement
# ---------------------------------------------------------------------------

def _nearest_swing_low(df: pd.DataFrame, entry: float, atr: float) -> float:
    """
    Return the highest swing low that is strictly below entry.
    Falls back to entry - 1.5×ATR if no qualifying swing low is found.
    Uses the last 50 bars with a 5-bar pivot lookback.
    """
    if df is None or len(df) < 11:
        return entry - _SL_ATR_MULT * atr
    window = df.iloc[-50:] if len(df) >= 50 else df
    swing_pts = find_swing_points(window, lookback=5)
    candidates = [sp.price for sp in swing_pts
                  if sp.swing_type in ("low", "HL", "LL") and sp.price < entry]
    if candidates:
        return max(candidates)  # nearest (highest) valid swing low below entry
    return entry - _SL_ATR_MULT * atr


def _nearest_swing_high(df: pd.DataFrame, entry: float, atr: float) -> float:
    """
    Return the lowest swing high that is strictly above entry.
    Falls back to entry + 1.5×ATR if no qualifying swing high is found.
    Uses the last 50 bars with a 5-bar pivot lookback.
    """
    if df is None or len(df) < 11:
        return entry + _SL_ATR_MULT * atr
    window = df.iloc[-50:] if len(df) >= 50 else df
    swing_pts = find_swing_points(window, lookback=5)
    candidates = [sp.price for sp in swing_pts
                  if sp.swing_type in ("high", "HH", "LH") and sp.price > entry]
    if candidates:
        return min(candidates)  # nearest (lowest) valid swing high above entry
    return entry + _SL_ATR_MULT * atr


def _is_price_near_key_level(df: pd.DataFrame, entry: float) -> bool:
    """
    Return True only when the current price is within _PROXIMITY_FILTER_PCT
    of at least one structural swing point (support or resistance).
    Trades from the middle of a range return False.
    """
    if df is None or len(df) < 11:
        return False
    window = df.iloc[-50:] if len(df) >= 50 else df
    swing_pts = find_swing_points(window, lookback=5)
    for sp in swing_pts:
        if abs(entry - sp.price) / entry <= _PROXIMITY_FILTER_PCT:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper: SMC confluence strings
# ---------------------------------------------------------------------------

def _build_smc_confluence(
    entry_price: float,
    direction: str,
    smc_data: Optional[dict],
    atr: float,
) -> list[str]:
    """
    Scan order blocks, FVGs, and liquidity sweeps for confluences near entry.
    Returns list of human-readable reason strings.
    """
    reasons: list[str] = []
    if not smc_data or atr == 0:
        return reasons

    proximity = atr * 1.0  # within 1 ATR counts as "near"

    # --- Order blocks ---
    for ob in smc_data.get("order_blocks", []):
        if not isinstance(ob, OrderBlock):
            continue
        if ob.direction != ("bullish" if direction == "LONG" else "bearish"):
            continue
        if ob.price_low <= entry_price <= ob.price_high:
            reasons.append(
                f"Inside {ob.direction} OB [{ob.price_low:.4f} – {ob.price_high:.4f}] "
                f"(str={ob.strength:.2f})"
            )
        elif abs(entry_price - ob.price_high) <= proximity:
            reasons.append(
                f"Near {ob.direction} OB top {ob.price_high:.4f} "
                f"(str={ob.strength:.2f})"
            )

    # --- Fair Value Gaps ---
    for fvg in smc_data.get("fair_value_gaps", []):
        if not isinstance(fvg, FairValueGap):
            continue
        if fvg.direction != ("bullish" if direction == "LONG" else "bearish"):
            continue
        if fvg.price_low <= entry_price <= fvg.price_high:
            reasons.append(
                f"Inside {fvg.direction} FVG "
                f"[{fvg.price_low:.4f} – {fvg.price_high:.4f}]"
            )

    # --- Liquidity sweeps ---
    sweeps = smc_data.get("liquidity_sweeps", [])
    if sweeps:
        latest_sweep: LiquiditySweep = sweeps[-1]
        if direction == "LONG" and latest_sweep.direction == "sell_side":
            reasons.append(
                f"Post sell-side liquidity sweep at {latest_sweep.price:.4f} "
                f"(confirmed={latest_sweep.confirmed})"
            )
        elif direction == "SHORT" and latest_sweep.direction == "buy_side":
            reasons.append(
                f"Post buy-side liquidity sweep at {latest_sweep.price:.4f} "
                f"(confirmed={latest_sweep.confirmed})"
            )

    return reasons


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class TrendStrategy:
    """
    Evaluates multi-timeframe data to produce a TradeSetup or None.

    Usage
    -----
    strategy = TrendStrategy()
    setup = strategy.evaluate(
        symbol="BTCUSDT",
        mtf_data={"15m": df_15m, "1h": df_1h, "4h": df_4h},
        indicators_15m=ind_15m,
        indicators_1h=ind_1h,
        indicators_4h=ind_4h,
        patterns=patterns,
        structure=structure_summary_4h,
        smc_data=smc_dict,
    )
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        mtf_data: dict,                         # {"15m": df, "1h": df, "4h": df}
        indicators_15m: IndicatorResult,
        indicators_1h:  IndicatorResult,
        indicators_4h:  IndicatorResult,
        patterns: list[CandlePattern],
        structure: dict,                         # output of get_structure_summary()
        smc_data: Optional[dict] = None,         # {'order_blocks', 'fair_value_gaps', ...}
    ) -> list[TradeSetup]:
        """
        Evaluate LONG and SHORT independently.
        Returns a list — can contain 0, 1, or 2 setups.
        Both are checked so BUY and SELL signals can fire on the same scan.
        """
        df_15m = mtf_data.get("15m")
        atr    = _current_atr(indicators_15m)
        close  = _current_close(df_15m)

        if math.isnan(close) or close == 0 or atr == 0:
            logger.debug("[Strategy] %s: insufficient price / ATR data.", symbol)
            return []

        kwargs = dict(
            symbol=symbol, df_15m=df_15m,
            indicators_15m=indicators_15m,
            indicators_1h=indicators_1h,
            indicators_4h=indicators_4h,
            patterns=patterns,
            structure=structure,
            smc_data=smc_data,
            close=close, atr=atr,
        )

        setups = []
        long_setup  = self._evaluate_long(**kwargs)
        short_setup = self._evaluate_short(**kwargs)

        if long_setup  is not None: setups.append(long_setup)
        if short_setup is not None: setups.append(short_setup)

        return setups

    # ------------------------------------------------------------------
    # BUY / LONG evaluation
    # ------------------------------------------------------------------

    def _evaluate_long(
        self,
        symbol: str,
        df_15m: Optional[pd.DataFrame],
        indicators_15m: IndicatorResult,
        indicators_1h:  IndicatorResult,
        indicators_4h:  IndicatorResult,
        patterns: list[CandlePattern],
        structure: dict,
        smc_data: Optional[dict],
        close: float,
        atr: float,
    ) -> Optional[TradeSetup]:
        conditions_passed = 0
        reasons: list[str] = []

        # 1. Price above EMA 200 on both 1h and 4h
        c1_ok = (
            not math.isnan(indicators_1h.ema_200)
            and not math.isnan(indicators_4h.ema_200)
            and close > indicators_1h.ema_200
            and close > indicators_4h.ema_200
        )
        if c1_ok:
            conditions_passed += 1
            reasons.append(
                f"Price {close:.4f} > EMA200 1h ({indicators_1h.ema_200:.4f}) "
                f"& 4h ({indicators_4h.ema_200:.4f})"
            )

        # 2. EMA 20 > EMA 50 on 1h
        c2_ok = (
            not math.isnan(indicators_1h.ema_20)
            and not math.isnan(indicators_1h.ema_50)
            and indicators_1h.ema_20 > indicators_1h.ema_50
        )
        if c2_ok:
            conditions_passed += 1
            reasons.append(
                f"1h EMA20 ({indicators_1h.ema_20:.4f}) > EMA50 ({indicators_1h.ema_50:.4f})"
            )

        # 3. RSI 50–65 on 15m — momentum clearly bullish, not just neutral
        rsi_15m = indicators_15m.rsi_14
        c3_ok = not math.isnan(rsi_15m) and 50.0 <= rsi_15m <= 65.0
        if c3_ok:
            conditions_passed += 1
            reasons.append(f"15m RSI {rsi_15m:.1f} in buy zone [50–65]")

        # 4. Bullish pattern on the PREVIOUS completed candle (iloc[-2])
        #    Patterns on the current forming candle (iloc[-1]) are unreliable — the
        #    candle hasn't closed yet and may not complete as that pattern.
        best_bullish_pattern: Optional[CandlePattern] = None
        if df_15m is not None and len(df_15m) >= 2:
            from core.patterns import detect_all as _detect_all
            confirmed_patterns = _detect_all(df_15m.iloc[:-1])   # exclude live candle
            for p in confirmed_patterns:
                if p.direction == "bullish" and p.confidence >= 0.60:
                    best_bullish_pattern = p
                    break
        c4_ok = best_bullish_pattern is not None
        if c4_ok:
            conditions_passed += 1
            reasons.append(
                f"Bullish pattern (confirmed): {best_bullish_pattern.name} "  # type: ignore[union-attr]
                f"(conf={best_bullish_pattern.confidence:.2f})"               # type: ignore[union-attr]
            )

        # 5. Volume spike on the previous completed 15m candle — must be a bullish candle
        c5_ok = _has_volume_spike(df_15m, direction="LONG")
        if c5_ok:
            conditions_passed += 1
            reasons.append("Bullish volume spike on last closed candle > 1.5x 20-bar avg")

        # 6. 4h structure bias = 'bullish' with meaningful trend strength
        bias     = structure.get("bias", "ranging")
        strength = structure.get("trend_strength", 0.0)
        c6_ok    = bias == "bullish" and strength >= _MIN_TREND_STRENGTH
        if c6_ok:
            conditions_passed += 1
            reasons.append(f"4h structure: {bias} (strength={strength:.2f})")

        # 7. MACD bullish on 15m — replaces the redundant 'RSI < 70' check.
        #    Requires MACD line above signal AND histogram positive.
        macd_line = indicators_15m.macd_line
        macd_sig  = indicators_15m.signal_line
        macd_hist = indicators_15m.histogram
        c7_ok = (
            not math.isnan(macd_line)
            and not math.isnan(macd_sig)
            and not math.isnan(macd_hist)
            and macd_line > macd_sig
            and macd_hist > 0
        )
        if c7_ok:
            conditions_passed += 1
            reasons.append(
                f"15m MACD bullish (line={macd_line:.4f} > sig={macd_sig:.4f}, "
                f"hist={macd_hist:.4f})"
            )

        if conditions_passed < _MIN_CONDITIONS:
            logger.debug(
                "[Strategy] %s LONG: only %d/%d conditions met.",
                symbol, conditions_passed, 7,
            )
            return None

        # Pre-trade proximity filter: skip trades fired from the middle of a range.
        if not _is_price_near_key_level(df_15m, close):
            logger.debug(
                "[Strategy] %s LONG: price %.4f not within %.1f%% of a key level — skipping.",
                symbol, close, _PROXIMITY_FILTER_PCT * 100,
            )
            return None

        # ----- Entry / SL / TP calculation -----
        #
        # Three constraints — SL is the LOWEST (farthest from entry) of all three:
        #   1. Below the nearest structural swing low  (a real pivot, not raw rolling min)
        #   2. At least 1.5× ATR below entry           (volatility minimum)
        #   3. At least 0.8% below entry               (crypto price floor)
        #
        entry       = close
        swing_low   = _nearest_swing_low(df_15m, entry, atr)
        struct_sl   = swing_low - 0.2 * atr          # small buffer below the pivot
        atr_sl      = entry - _SL_ATR_MULT * atr     # 1.5× ATR below entry
        pct_sl      = entry * (1.0 - _MIN_SL_PCT)    # 0.8% below entry
        stop_loss   = min(struct_sl, atr_sl, pct_sl) # widest (lowest) wins
        risk        = entry - stop_loss

        if risk <= 0:
            logger.debug("[Strategy] %s LONG: risk <= 0, skipping.", symbol)
            return None

        tp1 = entry + 2.0 * risk
        tp2 = entry + 3.0 * risk
        tp3 = entry + 5.0 * risk
        rr  = round((tp1 - entry) / risk, 3)

        if rr < _MIN_RR:
            logger.debug("[Strategy] %s LONG: R:R %.2f < %.1f, skipping.", symbol, rr, _MIN_RR)
            return None

        confidence    = round(conditions_passed / 7.0, 3)
        smc_confluence = _build_smc_confluence(entry, "LONG", smc_data, atr)

        return TradeSetup(
            symbol=symbol,
            direction="LONG",
            entry=round(entry, 8),
            stop_loss=round(stop_loss, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            rr=rr,
            confidence=confidence,
            reasons=reasons,
            pattern=best_bullish_pattern,
            timeframe="15m",
            structure_bias=bias,
            smc_confluence=smc_confluence,
        )

    # ------------------------------------------------------------------
    # SELL / SHORT evaluation
    # ------------------------------------------------------------------

    def _evaluate_short(
        self,
        symbol: str,
        df_15m: Optional[pd.DataFrame],
        indicators_15m: IndicatorResult,
        indicators_1h:  IndicatorResult,
        indicators_4h:  IndicatorResult,
        patterns: list[CandlePattern],
        structure: dict,
        smc_data: Optional[dict],
        close: float,
        atr: float,
    ) -> Optional[TradeSetup]:
        conditions_passed = 0
        reasons: list[str] = []

        # 1. Price below EMA 200 on both 1h and 4h
        c1_ok = (
            not math.isnan(indicators_1h.ema_200)
            and not math.isnan(indicators_4h.ema_200)
            and close < indicators_1h.ema_200
            and close < indicators_4h.ema_200
        )
        if c1_ok:
            conditions_passed += 1
            reasons.append(
                f"Price {close:.4f} < EMA200 1h ({indicators_1h.ema_200:.4f}) "
                f"& 4h ({indicators_4h.ema_200:.4f})"
            )

        # 2. EMA 20 < EMA 50 on 1h
        c2_ok = (
            not math.isnan(indicators_1h.ema_20)
            and not math.isnan(indicators_1h.ema_50)
            and indicators_1h.ema_20 < indicators_1h.ema_50
        )
        if c2_ok:
            conditions_passed += 1
            reasons.append(
                f"1h EMA20 ({indicators_1h.ema_20:.4f}) < EMA50 ({indicators_1h.ema_50:.4f})"
            )

        # 3. RSI 35–50 on 15m — momentum clearly bearish, not just neutral
        rsi_15m = indicators_15m.rsi_14
        c3_ok = not math.isnan(rsi_15m) and 35.0 <= rsi_15m <= 50.0
        if c3_ok:
            conditions_passed += 1
            reasons.append(f"15m RSI {rsi_15m:.1f} in sell zone [35–50]")

        # 4. Bearish pattern on the PREVIOUS completed candle (iloc[-2])
        best_bearish_pattern: Optional[CandlePattern] = None
        if df_15m is not None and len(df_15m) >= 2:
            from core.patterns import detect_all as _detect_all
            confirmed_patterns = _detect_all(df_15m.iloc[:-1])   # exclude live candle
            for p in confirmed_patterns:
                if p.direction == "bearish" and p.confidence >= 0.60:
                    best_bearish_pattern = p
                    break
        c4_ok = best_bearish_pattern is not None
        if c4_ok:
            conditions_passed += 1
            reasons.append(
                f"Bearish pattern (confirmed): {best_bearish_pattern.name} "  # type: ignore[union-attr]
                f"(conf={best_bearish_pattern.confidence:.2f})"               # type: ignore[union-attr]
            )

        # 5. Volume spike on the previous completed 15m candle — must be a bearish candle
        c5_ok = _has_volume_spike(df_15m, direction="SHORT")
        if c5_ok:
            conditions_passed += 1
            reasons.append("Bearish volume spike on last closed candle > 1.5x 20-bar avg")

        # 6. 4h structure bias = 'bearish' with meaningful trend strength
        bias     = structure.get("bias", "ranging")
        strength = structure.get("trend_strength", 0.0)
        c6_ok    = bias == "bearish" and strength >= _MIN_TREND_STRENGTH
        if c6_ok:
            conditions_passed += 1
            reasons.append(f"4h structure: {bias} (strength={strength:.2f})")

        # 7. MACD bearish on 15m — replaces the redundant 'RSI > 30' check.
        #    Requires MACD line below signal AND histogram negative.
        macd_line = indicators_15m.macd_line
        macd_sig  = indicators_15m.signal_line
        macd_hist = indicators_15m.histogram
        c7_ok = (
            not math.isnan(macd_line)
            and not math.isnan(macd_sig)
            and not math.isnan(macd_hist)
            and macd_line < macd_sig
            and macd_hist < 0
        )
        if c7_ok:
            conditions_passed += 1
            reasons.append(
                f"15m MACD bearish (line={macd_line:.4f} < sig={macd_sig:.4f}, "
                f"hist={macd_hist:.4f})"
            )

        if conditions_passed < _MIN_CONDITIONS:
            logger.debug(
                "[Strategy] %s SHORT: only %d/%d conditions met.",
                symbol, conditions_passed, 7,
            )
            return None

        # Pre-trade proximity filter: skip trades fired from the middle of a range.
        if not _is_price_near_key_level(df_15m, close):
            logger.debug(
                "[Strategy] %s SHORT: price %.4f not within %.1f%% of a key level — skipping.",
                symbol, close, _PROXIMITY_FILTER_PCT * 100,
            )
            return None

        # ----- Entry / SL / TP calculation -----
        #
        # Three constraints — SL is the HIGHEST (farthest from entry) of all three:
        #   1. Above the nearest structural swing high (a real pivot, not raw rolling max)
        #   2. At least 1.5× ATR above entry           (volatility minimum)
        #   3. At least 0.8% above entry               (crypto price floor)
        #
        entry       = close
        swing_high  = _nearest_swing_high(df_15m, entry, atr)
        struct_sl   = swing_high + 0.2 * atr          # small buffer above the pivot
        atr_sl      = entry + _SL_ATR_MULT * atr      # 1.5× ATR above entry
        pct_sl      = entry * (1.0 + _MIN_SL_PCT)     # 0.8% above entry
        stop_loss   = max(struct_sl, atr_sl, pct_sl)  # widest (highest) wins
        risk        = stop_loss - entry

        if risk <= 0:
            logger.debug("[Strategy] %s SHORT: risk <= 0, skipping.", symbol)
            return None

        tp1 = entry - 2.0 * risk
        tp2 = entry - 3.0 * risk
        tp3 = entry - 5.0 * risk
        rr  = round((entry - tp1) / risk, 3)

        if rr < _MIN_RR:
            logger.debug("[Strategy] %s SHORT: R:R %.2f < %.1f, skipping.", symbol, rr, _MIN_RR)
            return None

        confidence    = round(conditions_passed / 7.0, 3)
        smc_confluence = _build_smc_confluence(entry, "SHORT", smc_data, atr)

        return TradeSetup(
            symbol=symbol,
            direction="SHORT",
            entry=round(entry, 8),
            stop_loss=round(stop_loss, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3=round(tp3, 8),
            rr=rr,
            confidence=confidence,
            reasons=reasons,
            pattern=best_bearish_pattern,
            timeframe="15m",
            structure_bias=bias,
            smc_confluence=smc_confluence,
        )
