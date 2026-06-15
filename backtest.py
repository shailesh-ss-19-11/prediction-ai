"""
Fast vectorized backtester for DeltaSignalBot.

All indicators are pre-computed once on the full series.
The main loop only indexes into pre-computed arrays — O(n) total,
not O(n^2) like the bar-by-bar approach.

Data source: Binance public API (no key needed), cached to CSV.

Usage:
    python backtest.py
    python backtest.py --days 60 --symbol BTCUSDT
    python backtest.py --days 30 --symbol BTCUSDT ETHUSDT XAUTUSD
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))

import config
from core.indicators import IndicatorResult
from core.patterns import detect_all as detect_patterns
from core.market_structure import get_structure_summary
from core.smc import (
    detect_order_blocks,
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
    get_institutional_zones,
)
from core.strategy import TrendStrategy
from core.market_structure import find_swing_points

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "data", "bt_cache")

# Seconds per timeframe
_SEC = {"15m": 900, "1h": 3600, "4h": 14400}

# Asian session filter: 00:00-07:00 UTC (low liquidity, matches live bot)
_ASIAN_START, _ASIAN_END = 0, 7

# Trade management
_COOLDOWN_SEC    = 4 * 3600
_MAX_PER_DAY     = 3
_MIN_CONDITIONS  = 6            # 6 of 7 must pass
_MIN_TREND_STR   = 0.5
_VOLUME_LOOKBACK = 20
_VOLUME_MULT     = 1.5

# Realistic cost model
TAKER_FEE_RATE = 0.0005   # 0.05% per side
SLIPPAGE_PCT   = 0.0002   # 0.02% slippage on fill


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def _fetch_binance(symbol: str, interval: str, days: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        print(f"  [cache] {symbol} {interval}  {len(df):,} bars")
        return df.astype(float)

    print(f"  [fetch] {symbol} {interval}  {days}d...")
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 86400 * 1000
    rows: dict[int, dict] = {}
    cur = start_ms
    for _ in range(300):
        r = requests.get(BINANCE_URL, params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": now_ms, "limit": 1000,
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for k in batch:
            ts = int(k[0]) // 1000
            rows[ts] = {"timestamp": float(ts), "open": float(k[1]),
                        "high": float(k[2]),  "low": float(k[3]),
                        "close": float(k[4]), "volume": float(k[5])}
        newest = max(int(k[0]) for k in batch) // 1000
        if newest * 1000 >= now_ms - _SEC[interval] * 1000:
            break
        cur = (newest + _SEC[interval]) * 1000
        time.sleep(0.1)

    df = pd.DataFrame(sorted(rows.values(), key=lambda x: x["timestamp"])).astype(float)
    df.to_csv(cache, index=False)
    print(f"         saved {len(df):,} bars -> {cache}")
    return df


# ---------------------------------------------------------------------------
# Pre-vectorized indicators
# ---------------------------------------------------------------------------

def _ewm(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, float("nan")))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _macd(s: pd.Series):
    fast = s.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = s.ewm(span=26, adjust=False, min_periods=26).mean()
    line = fast - slow
    sig  = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line, sig, line - sig


def precompute(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to a copy of df. Runs once per timeframe."""
    d = df.copy()
    c = d["close"]
    d["ema20"]  = _ewm(c, 20)
    d["ema50"]  = _ewm(c, 50)
    d["ema200"] = _ewm(c, 200)
    d["rsi14"]  = _rsi(c)
    d["atr14"]  = _atr(d)
    d["macd_line"], d["macd_sig"], d["macd_hist"] = _macd(c)
    d["vol_ma20"] = d["volume"].rolling(20, min_periods=20).mean()
    return d


# ---------------------------------------------------------------------------
# Per-4h-bar structure (run once, forward-fill to 15m)
# ---------------------------------------------------------------------------

def build_structure_series(df4h: pd.DataFrame) -> pd.DataFrame:
    """
    For each fully-closed 4h bar, compute the bias and trend_strength.
    Uses a 100-bar rolling window (covers 400h ~ 17 days of context).
    Returns a DataFrame indexed by 4h bar index with 'bias' and 'strength'.
    """
    biases    = []
    strengths = []
    n = len(df4h)
    for k in range(n):
        window = df4h.iloc[max(0, k - 100): k + 1]
        s = get_structure_summary(window)
        biases.append(s["bias"])
        strengths.append(s["trend_strength"])
    return pd.DataFrame({
        "bias":     biases,
        "strength": strengths,
    }, index=df4h.index)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol:      str
    direction:   str
    entry:       float
    sl:          float
    initial_sl:  float
    tp1:         float
    tp2:         float
    size:        float
    entry_bar:   int
    exit_bar:    Optional[int] = None
    exit_price:  Optional[float] = None
    exit_reason: str = ""
    pnl:         float = 0.0
    fees:        float = 0.0


# ---------------------------------------------------------------------------
# Walk-forward simulator
# ---------------------------------------------------------------------------

class FastBacktester:
    def __init__(self, symbol: str, days: int, balance: float):
        self.symbol  = symbol
        self.days    = days
        self.balance = balance
        self.peak    = balance
        self.strategy = TrendStrategy()

        self._last_signal: dict[str, float] = {}
        self._day_count:   dict[object, int] = {}
        self.open_trades:   list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.signals = 0

        self.d15: pd.DataFrame
        self.d1h: pd.DataFrame
        self.d4h: pd.DataFrame
        self.struct4h: pd.DataFrame

    # ------------------------------------------------------------------
    def load(self) -> None:
        print(f"\n[{self.symbol}] Fetching data...")
        warmup = 60   # extra days for indicator warm-up
        total  = self.days + warmup
        self.d15 = _fetch_binance(self.symbol, "15m", total)
        self.d1h = _fetch_binance(self.symbol, "1h",  total)
        self.d4h = _fetch_binance(self.symbol, "4h",  total)

        print(f"  Pre-computing indicators...")
        self.d15 = precompute(self.d15)
        self.d1h = precompute(self.d1h)
        self.d4h = precompute(self.d4h)

        print(f"  Computing 4h structure (once per 4h bar)...")
        self.struct4h = build_structure_series(self.d4h)

        print(f"  Data ready: {len(self.d15):,} x 15m | {len(self.d1h):,} x 1h | {len(self.d4h):,} x 4h")

    # ------------------------------------------------------------------
    def _on_cooldown(self, direction: str, ts: float) -> bool:
        key = f"{self.symbol}_{direction}"
        return (ts - self._last_signal.get(key, 0)) < _COOLDOWN_SEC

    def _can_trade_today(self, ts: float) -> bool:
        day = datetime.fromtimestamp(ts, timezone.utc).date()
        return self._day_count.get(day, 0) < _MAX_PER_DAY

    def _record_signal(self, direction: str, ts: float) -> None:
        key = f"{self.symbol}_{direction}"
        self._last_signal[key] = ts
        day = datetime.fromtimestamp(ts, timezone.utc).date()
        self._day_count[day] = self._day_count.get(day, 0) + 1

    # ------------------------------------------------------------------
    def _update_trades(self, high: float, low: float, bar_idx: int) -> None:
        still_open: list[Trade] = []
        for t in self.open_trades:
            risk = abs(t.entry - t.initial_sl)
            # Breakeven: move SL to entry once +1R is reached
            if t.direction == "LONG":
                if risk > 0 and high >= t.entry + risk and t.sl < t.entry:
                    t.sl = t.entry
            else:
                if risk > 0 and low <= t.entry - risk and t.sl > t.entry:
                    t.sl = t.entry

            filled = False
            if t.direction == "LONG":
                if low <= t.sl:
                    reason = "breakeven" if t.sl >= t.entry else "sl"
                    self._close(t, t.sl, reason, bar_idx)
                    filled = True
                elif high >= t.tp2:
                    self._close(t, t.tp2, "tp2", bar_idx)
                    filled = True
                elif high >= t.tp1:
                    self._close(t, t.tp1, "tp1", bar_idx)
                    filled = True
            else:
                if high >= t.sl:
                    reason = "breakeven" if t.sl <= t.entry else "sl"
                    self._close(t, t.sl, reason, bar_idx)
                    filled = True
                elif low <= t.tp2:
                    self._close(t, t.tp2, "tp2", bar_idx)
                    filled = True
                elif low <= t.tp1:
                    self._close(t, t.tp1, "tp1", bar_idx)
                    filled = True

            if not filled:
                still_open.append(t)
        self.open_trades = still_open

    def _close(self, trade: Trade, exit_price: float, reason: str, bar_idx: int) -> None:
        slip = exit_price * SLIPPAGE_PCT
        if trade.direction == "LONG":
            exit_price -= slip
            pnl = (exit_price - trade.entry) * trade.size
        else:
            exit_price += slip
            pnl = (trade.entry - exit_price) * trade.size
        fees = (trade.entry + exit_price) * trade.size * TAKER_FEE_RATE
        net  = round(pnl - fees, 6)

        trade.exit_price  = round(exit_price, 6)
        trade.exit_bar    = bar_idx
        trade.exit_reason = reason
        trade.pnl         = net
        trade.fees        = round(fees, 6)
        self.balance     += net
        self.peak         = max(self.peak, self.balance)
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------
    def _check_signal(self, i: int) -> None:
        row15 = self.d15.iloc[i]
        close_ts = float(row15["timestamp"]) + _SEC["15m"]

        # Asian session filter
        bar_hour = datetime.fromtimestamp(close_ts, timezone.utc).hour
        if _ASIAN_START <= bar_hour < _ASIAN_END:
            return

        if not self._can_trade_today(close_ts):
            return

        # Map to fully-closed 1h and 4h bars
        ts1h = self.d1h["timestamp"].values
        ts4h = self.d4h["timestamp"].values
        k1 = int((ts1h + _SEC["1h"]  <= close_ts).sum()) - 1
        k4 = int((ts4h + _SEC["4h"]  <= close_ts).sum()) - 1

        if k1 < 210 or k4 < 205:   # need EMA200 warmed up on both
            return

        row1h  = self.d1h.iloc[k1]
        row4h  = self.d4h.iloc[k4]
        struct = self.struct4h.iloc[k4]

        # ---- Build IndicatorResult objects from precomputed columns ----
        ind_15m = IndicatorResult(
            ema_20=float(row15["ema20"])  if not pd.isna(row15["ema20"])  else float("nan"),
            ema_50=float(row15["ema50"])  if not pd.isna(row15["ema50"])  else float("nan"),
            ema_200=float(row15["ema200"])if not pd.isna(row15["ema200"]) else float("nan"),
            rsi_14=float(row15["rsi14"]) if not pd.isna(row15["rsi14"]) else float("nan"),
            atr_14=float(row15["atr14"]) if not pd.isna(row15["atr14"]) else float("nan"),
            macd_line=float(row15["macd_line"]) if not pd.isna(row15["macd_line"]) else float("nan"),
            signal_line=float(row15["macd_sig"]) if not pd.isna(row15["macd_sig"]) else float("nan"),
            histogram=float(row15["macd_hist"])  if not pd.isna(row15["macd_hist"]) else float("nan"),
        )
        ind_1h = IndicatorResult(
            ema_20=float(row1h["ema20"])  if not pd.isna(row1h["ema20"])  else float("nan"),
            ema_50=float(row1h["ema50"])  if not pd.isna(row1h["ema50"])  else float("nan"),
            ema_200=float(row1h["ema200"])if not pd.isna(row1h["ema200"]) else float("nan"),
        )
        ind_4h = IndicatorResult(
            ema_200=float(row4h["ema200"])if not pd.isna(row4h["ema200"]) else float("nan"),
        )

        structure = {
            "bias":           struct["bias"],
            "trend_strength": float(struct["strength"]),
            "last_bos":       None,
            "last_choch":     None,
            "swing_points":   [],
        }

        # Patterns (fast — just last 3 rows, O(1))
        win_15m = self.d15.iloc[max(0, i - 4): i + 1]
        patterns = detect_patterns(win_15m)

        # SMC (fast — only on last 60 bars)
        smc_win = self.d15.iloc[max(0, i - 59): i + 1]
        obs  = detect_order_blocks(smc_win)
        fvgs = detect_fair_value_gaps(smc_win)
        swings = find_swing_points(smc_win)
        smc_data = {
            "order_blocks":     obs,
            "fair_value_gaps":  fvgs,
            "liquidity_sweeps": detect_liquidity_sweeps(smc_win, swings),
            "zones":            get_institutional_zones(obs, fvgs),
        }

        # Full strategy window for SL placement (needs swing history)
        sl_win = self.d15.iloc[max(0, i - 59): i + 1]

        mtf = {"15m": sl_win, "1h": self.d1h.iloc[max(0, k1 - 60): k1 + 1],
               "4h": self.d4h.iloc[max(0, k4 - 60): k4 + 1]}

        setups = self.strategy.evaluate(
            self.symbol, mtf, ind_15m, ind_1h, ind_4h,
            patterns, structure, smc_data,
        )

        for setup in setups:
            if self._on_cooldown(setup.direction, close_ts):
                continue
            # No duplicate position
            if any(t.direction == setup.direction for t in self.open_trades):
                continue
            # Max concurrent positions cap
            if len(self.open_trades) >= 2:
                continue

            # Position sizing (1% risk)
            sl_dist = abs(setup.entry - setup.stop_loss)
            if sl_dist <= 0:
                continue
            risk_usd = self.balance * (config.MAX_RISK_PERCENT / 100.0)
            lot_size = risk_usd / sl_dist
            if lot_size <= 0:
                continue

            # Min R:R
            reward = abs(setup.tp1 - setup.entry)
            if sl_dist > 0 and (reward / sl_dist) < config.MIN_RISK_REWARD:
                continue

            # Apply entry slippage
            slip = setup.entry * SLIPPAGE_PCT
            actual_entry = setup.entry + (slip if setup.direction == "LONG" else -slip)
            entry_fee = actual_entry * lot_size * TAKER_FEE_RATE
            self.balance -= entry_fee

            trade = Trade(
                symbol=self.symbol,
                direction=setup.direction,
                entry=round(actual_entry, 6),
                sl=setup.stop_loss,
                initial_sl=setup.stop_loss,
                tp1=setup.tp1,
                tp2=setup.tp2,
                size=round(lot_size, 6),
                entry_bar=i,
            )
            self.open_trades.append(trade)
            self._record_signal(setup.direction, close_ts)
            self.signals += 1
            logger.info(
                "  [%s] SIGNAL %s @ %.2f  SL=%.2f  TP1=%.2f  lot=%.4f",
                datetime.fromtimestamp(close_ts, timezone.utc).strftime("%m-%d %H:%M"),
                setup.direction, actual_entry, setup.stop_loss, setup.tp1, lot_size,
            )

    # ------------------------------------------------------------------
    def run(self) -> None:
        n = len(self.d15)
        print(f"  Simulating {n:,} bars...", end="", flush=True)
        for i in range(n):
            bar = self.d15.iloc[i]
            self._update_trades(float(bar["high"]), float(bar["low"]), i)
            self._check_signal(i)
            if i % (n // 20) == 0:
                print(".", end="", flush=True)
        print(" done")

        # Force-close remaining positions at last close
        if self.open_trades:
            last_price = float(self.d15["close"].iloc[-1])
            for t in list(self.open_trades):
                self._close(t, last_price, "end_of_test", n - 1)
            self.open_trades.clear()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(bt: FastBacktester) -> None:
    trades = bt.closed_trades
    total  = len(trades)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  BACKTEST — {bt.symbol}  |  {bt.days}d window")
    print(sep)
    print(f"  Balance       : ${config.ACCOUNT_BALANCE:.2f}")
    print(f"  Risk/trade    : {config.MAX_RISK_PERCENT:.1f}%  Min R:R: {config.MIN_RISK_REWARD}")
    print(f"  Fees          : {TAKER_FEE_RATE*100:.3f}% per side   Slippage: {SLIPPAGE_PCT*100:.3f}%")
    print("-" * 64)

    if total == 0:
        print("  No trades — 6/7 strategy conditions never aligned.")
        print("  Try a longer window or a trending market period.")
        print(sep)
        return

    winners    = [t for t in trades if t.pnl > 0]
    losers     = [t for t in trades if t.pnl <= 0]
    breakevens = [t for t in trades if t.exit_reason == "breakeven"]
    sl_hits    = [t for t in trades if t.exit_reason == "sl"]
    total_pnl  = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    win_rate   = len(winners) / total * 100

    rr_vals = []
    for t in trades:
        sl_d = abs(t.entry - t.initial_sl)
        if sl_d > 0 and t.exit_price:
            rr_vals.append(abs(t.exit_price - t.entry) / sl_d)
    avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0

    avg_win  = sum(t.pnl for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl for t in losers)  / len(losers)  if losers  else 0
    expectancy = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss

    gross_wins   = sum(t.pnl for t in winners)
    gross_losses = abs(sum(t.pnl for t in losers))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    equity = config.ACCOUNT_BALANCE
    peak   = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    ret = (bt.balance / config.ACCOUNT_BALANCE - 1) * 100

    print(f"  Signals       : {bt.signals}   Trades closed: {total}")
    print(f"  Win rate      : {win_rate:.1f}%   ({len(winners)}W / {len(losers)}L / {len(breakevens)}BE)")
    print(f"  SL hits       : {len(sl_hits)}   Breakevens: {len(breakevens)}")
    print(f"  Net P&L       : ${total_pnl:+.4f}  (fees ${total_fees:.4f})")
    print(f"  Expectancy    : ${expectancy:+.4f} per trade")
    print(f"  Profit factor : {pf:.3f}")
    print(f"  Avg R         : {avg_rr:.2f}")
    print(f"  Max drawdown  : {max_dd:.2f}%")
    print(f"  Start balance : ${config.ACCOUNT_BALANCE:.2f}")
    print(f"  End balance   : ${bt.balance:.4f}")
    print(f"  Return        : {ret:+.2f}%")
    print("-" * 64)

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("  Exits: " + "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))

    print("-" * 64)
    print(f"  {'#':>3}  {'Date':>14}  {'Dir':<6}  {'Entry':>10}  {'Exit':>10}  {'PnL':>9}  Reason")
    for idx, t in enumerate(trades[-15:], 1):
        dt = ""
        if t.exit_bar and t.exit_bar < len(bt.d15):
            ts = float(bt.d15.iloc[t.exit_bar]["timestamp"])
            dt = datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M")
        ep = f"{t.exit_price:.2f}" if t.exit_price else "open"
        mark = "W" if t.pnl > 0 else ("B" if t.exit_reason == "breakeven" else "L")
        print(f"  {idx:>3}  {dt:>14}  {t.direction:<6}  {t.entry:>10.2f}  {ep:>10}  "
              f"{t.pnl:>+9.4f} [{mark}] {t.exit_reason}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fast vectorized backtester")
    ap.add_argument("--days",    type=int,   default=60,         help="test window in days")
    ap.add_argument("--balance", type=float, default=config.ACCOUNT_BALANCE)
    ap.add_argument("--symbol",  nargs="+",  default=["BTCUSDT"], help="Binance symbol(s)")
    ap.add_argument("--nocache", action="store_true", help="ignore cached CSV files")
    args = ap.parse_args()

    if args.nocache:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("Cache cleared.")

    for sym in args.symbol:
        bt = FastBacktester(sym, args.days, args.balance)
        bt.load()
        t0 = time.time()
        bt.run()
        elapsed = time.time() - t0
        print(f"  Simulation took {elapsed:.1f}s")
        report(bt)


if __name__ == "__main__":
    main()
