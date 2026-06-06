"""
Backtest harness for DeltaSignalBot.

Fetches 30-day + 45-day warmup history from Binance (public API, no key needed),
replays every 15m bar through the live strategy stack, simulates intrabar SL/TP,
and prints a full performance report.

Usage:
    python3 backtest.py
    python3 backtest.py --days 30 --symbol BTCUSDT
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
from core.indicators import calculate_all
from core.patterns import detect_all as detect_patterns
from core.market_structure import get_structure_summary, find_swing_points, classify_swings
from core.smc import (
    detect_order_blocks,
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
)
from core.strategy import TrendStrategy
from core.advanced_filters import VolatilityFilter
from risk.risk_manager import RiskManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "bt_cache")

# Rolling window sizes — mirror MarketDataManager.CANDLE_LIMITS
_WIN_15M, _WIN_1H, _WIN_4H = 60, 250, 300
_WARMUP_4H = 200   # need 200 4h bars for EMA200 to be valid

_SEC = {"15m": 900, "1h": 3600, "4h": 14400}
_COOLDOWN_SEC = 4 * 3600
_MAX_PER_DAY  = 3

# Fees / slippage (realistic for crypto futures)
TAKER_FEE_RATE   = 0.0005   # 0.05% per side
SLIPPAGE_PCT     = 0.0002   # 0.02% per fill


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def _binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    r = requests.get(
        BINANCE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        },
        timeout=30,
    )
    r.raise_for_status()
    rows = []
    for k in r.json():
        rows.append({
            "timestamp": k[0] / 1000.0,   # open time → seconds
            "open":  float(k[1]),
            "high":  float(k[2]),
            "low":   float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return rows


def fetch_history(symbol: str, interval: str, days: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        logger.info("  cache hit  %-10s %-3s  %d candles", symbol, interval, len(df))
        return df

    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms  = now_ms - days * 24 * 3600 * 1000
    by_time: dict[int, dict] = {}

    cur_start = start_ms
    for _ in range(200):
        chunk = _binance_klines(symbol, interval, cur_start, now_ms)
        if not chunk:
            break
        for c in chunk:
            by_time[int(c["timestamp"])] = c
        oldest = min(int(c["timestamp"]) for c in chunk)
        newest = max(int(c["timestamp"]) for c in chunk)
        if newest * 1000 >= now_ms - _SEC[interval] * 1000:
            break
        cur_start = (newest + _SEC[interval]) * 1000
        time.sleep(0.1)

    rows = [v for _, v in sorted(by_time.items())]
    df = pd.DataFrame(rows).astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    logger.info("  fetched    %-10s %-3s  %d candles", symbol, interval, len(df))
    return df


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    size: float
    entry_bar: int   # index in d15
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    fees: float = 0.0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(self, symbol: str, days: int, balance: float = 100.0):
        self.symbol  = symbol
        self.days    = days
        self.balance = balance
        self.peak    = balance
        self.strategy = TrendStrategy()
        self.risk     = RiskManager(balance, config.MAX_RISK_PERCENT)
        self.vol      = VolatilityFilter()
        self.data: dict[str, pd.DataFrame] = {}

        self._last_signal: dict[str, float] = {}
        self._day_count: dict[tuple, int] = {}
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.signals = 0

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    def load(self) -> None:
        fetch_days = self.days + 50   # extra warmup margin
        self.data = {
            "15m": fetch_history(self.symbol, "15m", fetch_days),
            "1h":  fetch_history(self.symbol, "1h",  fetch_days),
            "4h":  fetch_history(self.symbol, "4h",  fetch_days),
        }

    # ------------------------------------------------------------------
    # Cooldown / daily cap helpers
    # ------------------------------------------------------------------
    def _on_cooldown(self, direction: str, close_time: float) -> bool:
        key = f"{self.symbol}_{direction}"
        return (close_time - self._last_signal.get(key, 0)) < _COOLDOWN_SEC

    def _can_trade_today(self, close_time: float) -> bool:
        day = datetime.fromtimestamp(close_time, timezone.utc).date()
        return self._day_count.get(day, 0) < _MAX_PER_DAY

    def _record_signal(self, direction: str, close_time: float) -> None:
        key = f"{self.symbol}_{direction}"
        self._last_signal[key] = close_time
        day = datetime.fromtimestamp(close_time, timezone.utc).date()
        self._day_count[day] = self._day_count.get(day, 0) + 1

    # ------------------------------------------------------------------
    # Intrabar SL/TP check
    # ------------------------------------------------------------------
    def _update_trades(self, bar_high: float, bar_low: float) -> None:
        still_open: list[Trade] = []
        for t in self.open_trades:
            filled = False
            if t.direction == "LONG":
                if bar_low <= t.sl:
                    self._close(t, t.sl, "sl")
                    filled = True
                elif bar_high >= t.tp2:
                    self._close(t, t.tp2, "tp2")
                    filled = True
                elif bar_high >= t.tp1:
                    self._close(t, t.tp1, "tp1")
                    filled = True
            else:   # SHORT
                if bar_high >= t.sl:
                    self._close(t, t.sl, "sl")
                    filled = True
                elif bar_low <= t.tp2:
                    self._close(t, t.tp2, "tp2")
                    filled = True
                elif bar_low <= t.tp1:
                    self._close(t, t.tp1, "tp1")
                    filled = True
            if not filled:
                still_open.append(t)
        self.open_trades = still_open

    def _close(self, trade: Trade, exit_price: float, reason: str) -> None:
        slippage = exit_price * SLIPPAGE_PCT
        if trade.direction == "LONG":
            exit_price -= slippage
            pnl = (exit_price - trade.entry) * trade.size
        else:
            exit_price += slippage
            pnl = (trade.entry - exit_price) * trade.size

        fees = (trade.entry + exit_price) * trade.size * TAKER_FEE_RATE
        trade.exit_price  = exit_price
        trade.exit_reason = reason
        trade.pnl         = round(pnl - fees, 6)
        trade.fees        = round(fees, 6)
        self.balance     += trade.pnl
        self.peak         = max(self.peak, self.balance)
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------
    # Per-bar evaluation (strict no-lookahead)
    # ------------------------------------------------------------------
    def _evaluate(self, i: int) -> None:
        d15, d1h, d4h = self.data["15m"], self.data["1h"], self.data["4h"]
        ts15 = d15["timestamp"].values
        ts1h = d1h["timestamp"].values
        ts4h = d4h["timestamp"].values

        close_time = ts15[i] + _SEC["15m"]   # this bar is fully closed

        # how many 1h / 4h bars are FULLY closed by close_time
        k1 = int((ts1h + _SEC["1h"]  <= close_time).sum())
        k4 = int((ts4h + _SEC["4h"]  <= close_time).sum())

        if k1 < 210 or k4 < _WARMUP_4H:   # need 200+ 4h for EMA200
            return

        if not self._can_trade_today(close_time):
            return

        # Skip Asian session (00:00–07:00 UTC) — matches live bot behaviour
        bar_hour = datetime.fromtimestamp(close_time, timezone.utc).hour
        if 0 <= bar_hour < 7:
            return

        df_15m = d15.iloc[max(0, i - _WIN_15M + 1): i + 1].copy()
        df_1h  = d1h.iloc[max(0, k1 - _WIN_1H):  k1].copy()
        df_4h  = d4h.iloc[max(0, k4 - _WIN_4H):  k4].copy()

        ind_15m = calculate_all(df_15m)
        ind_1h  = calculate_all(df_1h)
        ind_4h  = calculate_all(df_4h)

        if self.vol.get_volatility_state(df_15m, ind_15m.atr_14) == "extreme":
            return

        patterns  = detect_patterns(df_15m)
        structure = get_structure_summary(df_4h)

        raw_swings = find_swing_points(df_15m)
        obs  = detect_order_blocks(df_15m)
        fvgs = detect_fair_value_gaps(df_15m)
        sweeps = detect_liquidity_sweeps(df_15m, raw_swings)

        smc_data = {
            "order_blocks":    obs,
            "fair_value_gaps": fvgs,
            "liquidity_sweeps": sweeps,
        }

        setups = self.strategy.evaluate(
            self.symbol,
            {"15m": df_15m, "1h": df_1h, "4h": df_4h},
            ind_15m, ind_1h, ind_4h,
            patterns, structure, smc_data,
        )

        for setup in setups:
            if self._on_cooldown(setup.direction, close_time):
                continue
            if len(self.open_trades) >= 3:
                continue

            risk = self.risk.evaluate_trade(
                setup.entry, setup.stop_loss, setup.tp1,
                balance=self.balance,
                current_balance=self.balance,
                peak_balance=self.peak,
            )
            if not risk.can_trade or risk.lot_size <= 0:
                continue

            # Apply slippage to entry
            slippage = setup.entry * SLIPPAGE_PCT
            actual_entry = setup.entry + (slippage if setup.direction == "LONG" else -slippage)
            entry_fee = actual_entry * risk.lot_size * TAKER_FEE_RATE
            self.balance -= entry_fee

            trade = Trade(
                symbol=self.symbol,
                direction=setup.direction,
                entry=actual_entry,
                sl=setup.stop_loss,
                tp1=setup.tp1,
                tp2=setup.tp2,
                size=risk.lot_size,
                entry_bar=i,
            )
            self.open_trades.append(trade)
            self._record_signal(setup.direction, close_time)
            self.signals += 1
            logger.info(
                "  SIGNAL %s %s @ %.4f  SL=%.4f  TP1=%.4f  TP2=%.4f",
                setup.direction, self.symbol, actual_entry,
                setup.stop_loss, setup.tp1, setup.tp2,
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        d15 = self.data["15m"]
        n   = len(d15)
        logger.info("Replaying %d bars for %s...", n, self.symbol)

        for i in range(n):
            bar = d15.iloc[i]
            # 1) update open trades with this bar's intrabar range
            self._update_trades(float(bar["high"]), float(bar["low"]))
            # 2) evaluate a fresh signal on the just-closed bar
            self._evaluate(i)

        # Close remaining open trades at last price (end of test window)
        if self.open_trades:
            last_close = float(d15["close"].iloc[-1])
            for t in list(self.open_trades):
                self._close(t, last_close, "end_of_test")
            self.open_trades.clear()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(bt: Backtester) -> None:
    trades = bt.closed_trades
    total  = len(trades)

    print()
    print("=" * 62)
    print(f"  BACKTEST RESULTS — {bt.days}d window")
    print("=" * 62)
    print(f"  Symbol           : {bt.symbol}")
    print(f"  Fees             : {TAKER_FEE_RATE*100:.3f}% per side (taker)")
    print(f"  Slippage         : {SLIPPAGE_PCT*100:.3f}% per fill")
    print(f"  Risk per trade   : {config.MAX_RISK_PERCENT:.1f}% of ${bt.balance:.2f}")
    print("-" * 62)

    if total == 0:
        print("  Signals opened   : 0")
        print("  No trades triggered — 6/7 conditions never aligned in")
        print("  the test window. Strategy requires a strongly trending")
        print("  market with multi-TF alignment.")
        print("=" * 62)
        return

    winners  = [t for t in trades if t.pnl > 0]
    losers   = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    win_rate  = len(winners) / total * 100

    # Avg R achieved
    rr_vals = []
    for t in trades:
        sl_d = abs(t.entry - t.sl)
        if sl_d > 0 and t.exit_price is not None:
            rr_vals.append(abs(t.exit_price - t.entry) / sl_d)
    avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0

    # Expectancy
    avg_win  = sum(t.pnl for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl for t in losers)  / len(losers)  if losers  else 0
    expectancy = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss

    # Profit factor
    gross_wins  = sum(t.pnl for t in winners)
    gross_losses = abs(sum(t.pnl for t in losers))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    # Max drawdown
    equity = config.ACCOUNT_BALANCE
    peak   = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    ret = (bt.balance / config.ACCOUNT_BALANCE - 1) * 100

    print(f"  Signals opened   : {bt.signals}")
    print(f"  Trades closed    : {total}")
    print(f"  Win rate         : {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"  Net P&L          : ${total_pnl:+.4f}  (fees paid ${total_fees:.4f})")
    print(f"  Expectancy/trade : ${expectancy:+.4f}")
    print(f"  Profit factor    : {profit_factor:.3f}")
    print(f"  Avg R achieved   : {avg_rr:.2f}")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print(f"  Start balance    : ${config.ACCOUNT_BALANCE:.2f}")
    print(f"  End balance      : ${bt.balance:.4f}")
    print(f"  Return           : {ret:+.2f}%")
    print("-" * 62)

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("  Exit reasons     : " +
          "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))

    print("-" * 62)
    print("  Trade log (last 10):")
    print(f"  {'#':>3}  {'Dir':<6}  {'Entry':>10}  {'Exit':>10}  "
          f"{'PnL':>8}  {'Reason':<14}")
    for idx, t in enumerate(trades[-10:]):
        ep = f"{t.exit_price:.4f}" if t.exit_price else "open"
        print(f"  {idx+1:>3}  {t.direction:<6}  {t.entry:>10.4f}  {ep:>10}  "
              f"{t.pnl:>+8.4f}  {t.exit_reason:<14}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest DeltaSignalBot")
    ap.add_argument("--days",   type=int, default=30,       help="test window in days")
    ap.add_argument("--symbol", type=str, default="BTCUSDT", help="Binance symbol")
    ap.add_argument("--balance",type=float, default=config.ACCOUNT_BALANCE)
    args = ap.parse_args()

    print(f"Loading {args.days}d history (+50d warmup) for {args.symbol}...")
    bt = Backtester(args.symbol, args.days, args.balance)
    bt.load()
    print("Running backtest...")
    bt.run()
    report(bt)


if __name__ == "__main__":
    main()
