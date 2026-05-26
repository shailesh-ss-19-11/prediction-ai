"""
Complete technical indicators module.
All calculations use pandas and numpy.
Returns the most-recent (scalar) values inside an IndicatorResult dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IndicatorResult:
    # --- EMA ---
    ema_20:  float = float("nan")
    ema_50:  float = float("nan")
    ema_200: float = float("nan")

    # --- RSI ---
    rsi_14:  float = float("nan")

    # --- MACD ---
    macd_line:   float = float("nan")
    signal_line: float = float("nan")
    histogram:   float = float("nan")

    # --- ATR ---
    atr_14: float = float("nan")

    # --- Bollinger Bands ---
    bb_upper:     float = float("nan")
    bb_middle:    float = float("nan")
    bb_lower:     float = float("nan")
    bb_width:     float = float("nan")
    bb_percent_b: float = float("nan")

    # --- VWAP ---
    vwap: float = float("nan")

    # --- Volume Profile ---
    volume_poc: float = float("nan")   # Point of Control
    volume_vah: float = float("nan")   # Value Area High
    volume_val: float = float("nan")   # Value Area Low

    # --- Derived signals ---
    trend_ema: str = "neutral"         # 'bullish', 'bearish', 'neutral'
    rsi_zone:  str = "neutral"         # 'overbought', 'oversold', 'buy_zone', 'sell_zone', 'neutral'


# ---------------------------------------------------------------------------
# Individual indicator calculations
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average using pandas ewm (min_periods=span)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (equivalent to RMA / SMMA).
    This matches TradingView's RSI implementation.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, and histogram.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast   = series.ewm(span=fast,   adjust=False, min_periods=fast).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False, min_periods=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing.
    Requires columns: high, low, close.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper, middle, lower, width, percent_b).
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std    = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std

    band_range = upper - lower
    width      = band_range / middle.replace(0, float("nan"))
    percent_b  = (series - lower) / band_range.replace(0, float("nan"))

    return upper, middle, lower, width, percent_b


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP (volume-weighted average price).

    If a 'timestamp' column is present, the series is reset (cumulative sum
    restarted) at each calendar day boundary.  Otherwise VWAP is computed
    cumulatively over the entire DataFrame.

    Requires columns: high, low, close, volume.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume        = df["volume"].fillna(0.0)

    if "timestamp" in df.columns:
        # Convert Unix seconds to date string for grouping
        dates = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
        df_tmp = pd.DataFrame(
            {"tp": typical_price, "vol": volume, "date": dates}
        )
        vwap_values = pd.Series(index=df.index, dtype="float64")

        for date, grp in df_tmp.groupby("date", sort=False):
            cum_tpv  = (grp["tp"] * grp["vol"]).cumsum()
            cum_vol  = grp["vol"].cumsum()
            vwap_day = cum_tpv / cum_vol.replace(0, float("nan"))
            vwap_values.loc[grp.index] = vwap_day.values

        return vwap_values
    else:
        cum_tpv = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tpv / cum_vol.replace(0, float("nan"))


def _volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
    value_area_pct: float = 0.70,
) -> Tuple[float, float, float]:
    """
    Volume Profile: Point of Control (POC), Value Area High (VAH),
    Value Area Low (VAL).

    Parameters
    ----------
    df             : DataFrame with columns high, low, close, volume
    bins           : number of price bins
    value_area_pct : fraction of total volume that defines the value area

    Returns
    -------
    (poc_price, vah_price, val_price)
    """
    prices = df["close"].dropna()
    volume = df["volume"].fillna(0.0)

    if prices.empty or volume.sum() == 0:
        nan = float("nan")
        return nan, nan, nan

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_min == price_max:
        nan = float("nan")
        return nan, nan, nan

    bin_edges   = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Assign each candle's volume to the bin matching its close price
    bin_indices = np.digitize(prices.values, bin_edges[1:-1])  # 0-indexed bin
    bin_volume  = np.zeros(bins, dtype=float)

    for idx, vol in zip(bin_indices, volume.values):
        if 0 <= idx < bins:
            bin_volume[idx] += vol

    # Point of Control: bin with the most volume
    poc_idx   = int(np.argmax(bin_volume))
    poc_price = float(bin_centers[poc_idx])

    # Value Area: expand outward from POC until value_area_pct of total volume
    total_vol      = bin_volume.sum()
    target_vol     = total_vol * value_area_pct

    accumulated    = bin_volume[poc_idx]
    low_idx        = poc_idx
    high_idx       = poc_idx

    while accumulated < target_vol:
        can_expand_low  = low_idx  > 0
        can_expand_high = high_idx < bins - 1

        if not can_expand_low and not can_expand_high:
            break

        vol_below = bin_volume[low_idx  - 1] if can_expand_low  else -1.0
        vol_above = bin_volume[high_idx + 1] if can_expand_high else -1.0

        if vol_above >= vol_below:
            high_idx   += 1
            accumulated += bin_volume[high_idx]
        else:
            low_idx    -= 1
            accumulated += bin_volume[low_idx]

    vah_price = float(bin_centers[high_idx])
    val_price = float(bin_centers[low_idx])

    return poc_price, vah_price, val_price


# ---------------------------------------------------------------------------
# Derived signal helpers
# ---------------------------------------------------------------------------

def _trend_ema(
    close: float,
    ema_20_val: float,
    ema_50_val: float,
    ema_200_val: float,
) -> str:
    """
    'bullish'  if close > ema_200 AND ema_20 > ema_50
    'bearish'  if close < ema_200 AND ema_20 < ema_50
    'neutral'  otherwise
    """
    if any(np.isnan(v) for v in [close, ema_20_val, ema_50_val, ema_200_val]):
        return "neutral"

    if close > ema_200_val and ema_20_val > ema_50_val:
        return "bullish"
    if close < ema_200_val and ema_20_val < ema_50_val:
        return "bearish"
    return "neutral"


def _rsi_zone(rsi_val: float) -> str:
    """
    'overbought' if RSI > 70
    'oversold'   if RSI < 30
    'buy_zone'   if 45 <= RSI <= 65
    'sell_zone'  if 35 <= RSI <= 55  (overlaps with buy_zone — buy_zone takes priority)
    'neutral'    otherwise
    """
    if np.isnan(rsi_val):
        return "neutral"

    if rsi_val > 70:
        return "overbought"
    if rsi_val < 30:
        return "oversold"
    # buy_zone is checked before sell_zone so that 45-55 overlap → buy_zone
    if 45.0 <= rsi_val <= 65.0:
        return "buy_zone"
    if 35.0 <= rsi_val <= 55.0:
        return "sell_zone"
    return "neutral"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_all(df: pd.DataFrame) -> IndicatorResult:
    """
    Calculate all indicators from an OHLCV DataFrame and return
    the most-recent scalar values in an IndicatorResult.

    Parameters
    ----------
    df : pd.DataFrame with columns: open, high, low, close, volume
         Optional column: timestamp (Unix seconds, used for daily VWAP reset)

    Returns
    -------
    IndicatorResult with all scalar fields populated (NaN where insufficient data).
    """
    result = IndicatorResult()

    if df is None or df.empty:
        logger.warning("[Indicators] Empty DataFrame passed to calculate_all.")
        return result

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.error("[Indicators] Missing columns: %s", missing)
        return result

    close = df["close"].astype(float)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    try:
        ema20_s  = _ema(close, 20)
        ema50_s  = _ema(close, 50)
        ema200_s = _ema(close, 200)
        result.ema_20  = _last(ema20_s)
        result.ema_50  = _last(ema50_s)
        result.ema_200 = _last(ema200_s)
    except Exception as exc:
        logger.warning("[Indicators] EMA calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    try:
        rsi_s      = _rsi(close, period=14)
        result.rsi_14 = _last(rsi_s)
    except Exception as exc:
        logger.warning("[Indicators] RSI calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    try:
        macd_s, sig_s, hist_s = _macd(close, fast=12, slow=26, signal=9)
        result.macd_line   = _last(macd_s)
        result.signal_line = _last(sig_s)
        result.histogram   = _last(hist_s)
    except Exception as exc:
        logger.warning("[Indicators] MACD calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------
    try:
        atr_s       = _atr(df, period=14)
        result.atr_14 = _last(atr_s)
    except Exception as exc:
        logger.warning("[Indicators] ATR calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    try:
        bb_up, bb_mid, bb_lo, bb_w, bb_pb = _bollinger_bands(close, period=20, std_dev=2.0)
        result.bb_upper     = _last(bb_up)
        result.bb_middle    = _last(bb_mid)
        result.bb_lower     = _last(bb_lo)
        result.bb_width     = _last(bb_w)
        result.bb_percent_b = _last(bb_pb)
    except Exception as exc:
        logger.warning("[Indicators] Bollinger Bands calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------
    try:
        vwap_s      = _vwap(df)
        result.vwap = _last(vwap_s)
    except Exception as exc:
        logger.warning("[Indicators] VWAP calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # Volume Profile
    # ------------------------------------------------------------------
    try:
        poc, vah, val       = _volume_profile(df)
        result.volume_poc   = poc
        result.volume_vah   = vah
        result.volume_val   = val
    except Exception as exc:
        logger.warning("[Indicators] Volume Profile calculation failed: %s", exc)

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------
    try:
        current_close      = float(close.iloc[-1]) if not close.empty else float("nan")
        result.trend_ema   = _trend_ema(
            current_close, result.ema_20, result.ema_50, result.ema_200
        )
        result.rsi_zone    = _rsi_zone(result.rsi_14)
    except Exception as exc:
        logger.warning("[Indicators] Derived signals calculation failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _last(series: pd.Series) -> float:
    """Return the last non-NaN value in the series, or NaN if none exists."""
    if series.empty:
        return float("nan")
    val = series.iloc[-1]
    return float(val) if not pd.isna(val) else float("nan")
