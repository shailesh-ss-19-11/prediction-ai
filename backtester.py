#!/usr/bin/env python3
"""
Walk-forward backtester for TrendStrategy.

Fetches 3 months of 15m BTCUSD candles from Delta Exchange (paginated),
resamples to 1h / 4h in-process, then simulates every trade signal at
15m bar resolution — no look-ahead.

Usage
-----
    python backtester.py                       # defaults
    python backtester.py --symbol BTCUSD --days 90 --balance 1000
    python backtester.py --out my_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from exchanges.delta_exchange import BASE_URL, PRODUCT_IDS
from core.indicators import calculate_all
from core.market_structure import get_structure_summary, find_swing_points
from core.patterns import detect_all as detect_patterns
from core.smc import (detect_fair_value_gaps, detect_liquidity_sweeps,
                       detect_order_blocks, get_institutional_zones)
from core.strategy import TrendStrategy

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    bar_idx:    int
    timestamp:  str
    direction:  str
    entry:      float
    stop_loss:  float
    tp1:        float
    rr_planned: float
    confidence: float
    exit_price: float
    result:     str   # WIN | LOSS | TIMEOUT
    pnl_r:      float # actual R multiple achieved
    bars_held:  int


@dataclass
class BacktestResult:
    symbol:              str
    period_days:         int
    start_date:          str
    end_date:            str
    total_bars_scanned:  int
    total_trades:        int
    wins:                int
    losses:              int
    breakevens:          int
    timeouts:            int
    win_rate_pct:        float
    avg_rr:              float
    avg_winner_r:        float
    avg_loser_r:         float
    max_drawdown_pct:    float
    final_balance:       float
    total_return_pct:    float
    initial_balance:     float
    trades:              list


# ---------------------------------------------------------------------------
# Historical data fetch (paginated)
# ---------------------------------------------------------------------------

def _ts_str(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fetch_15m_candles(symbol: str, days: int) -> pd.DataFrame:
    """
    Pull `days` worth of 15m candles directly from the Delta Exchange
    history endpoint, walking backwards in 500-candle chunks.

    Returns a deduplicated, timestamp-sorted DataFrame with columns:
        timestamp, open, high, low, close, volume  (all float64).
    """
    if symbol not in PRODUCT_IDS:
        raise ValueError(f"Unknown symbol '{symbol}'. "
                         f"Add it to exchanges/delta_exchange.py PRODUCT_IDS.")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json",
                             "Accept":       "application/json"})

    end_ts      = int(datetime.now(timezone.utc).timestamp())
    start_ts    = end_ts - days * 86400
    chunk_secs  = 500 * 15 * 60   # 500 bars × 15 min = 5.2 days per request
    total_chunks = max(1, (end_ts - start_ts) // chunk_secs + 1)

    print(f"\nFetching {days}d of 15m {symbol} candles "
          f"(~{total_chunks} requests)...", flush=True)

    all_rows: list[dict] = []
    current_end = end_ts
    chunk_n = 0

    while current_end > start_ts:
        current_start = max(current_end - chunk_secs, start_ts)
        params = {
            "resolution": "15m",
            "symbol":     symbol,
            "start":      current_start,
            "end":        current_end,
        }
        try:
            resp = session.get(
                f"{BASE_URL}/v2/history/candles",
                params=params, timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") is False:
                logger.warning("API returned success=false for chunk %d", chunk_n)
            candles = data.get("result", [])
            for c in candles:
                all_rows.append({
                    "timestamp": float(c.get("time", 0)),
                    "open":      float(c.get("open", 0)),
                    "high":      float(c.get("high", 0)),
                    "low":       float(c.get("low", 0)),
                    "close":     float(c.get("close", 0)),
                    "volume":    float(c.get("volume", 0) or 0),
                })
            chunk_n += 1
            print(f"  chunk {chunk_n}/{total_chunks} - {len(candles)} bars", flush=True)
        except requests.RequestException as exc:
            logger.warning("Fetch error on chunk %d: %s — skipping", chunk_n, exc)

        current_end = current_start
        time.sleep(0.25)   # stay within rate limits

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).astype("float64")
    df = (df.drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))

    print(f"  Total: {len(df):,} candles  "
          f"[{_ts_str(df['timestamp'].iloc[0])} to "
          f"{_ts_str(df['timestamp'].iloc[-1])}]", flush=True)
    return df


def _resample_15m(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 15m OHLCV DataFrame to '1h' or '4h'.
    Uses left-closed, left-labelled bins so the bar timestamp is the open.
    """
    df = df_15m.copy()
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    r = df.resample(rule, closed="left", label="left").agg({
        "timestamp": "first",
        "open":      "first",
        "high":      "max",
        "low":       "min",
        "close":     "last",
        "volume":    "sum",
    }).dropna(subset=["open"])
    return r.reset_index(drop=True).astype("float64")


# ---------------------------------------------------------------------------
# Single-trade simulator
# ---------------------------------------------------------------------------

def _simulate_trade(
    df_15m:    pd.DataFrame,
    signal_i:  int,
    entry:     float,
    stop_loss: float,
    tp1:       float,
    direction: str,
    max_bars:  int = 200,
) -> tuple[str, float, int, float]:
    """
    Scan bars forward from signal_i + 1.

    Conservative fill order within a bar: SL is checked before TP.
    This slightly underestimates win rate, consistent with real execution.

    Matches the live paper engine: once a bar reaches +1R the stop moves to
    breakeven (applied from the NEXT bar — conservative, no intra-bar ordering
    assumption), so a +1R trade that fully reverses scratches instead of -1R.

    Returns
    -------
    (result, exit_price, bars_held, pnl_r)
    result  : 'WIN' | 'LOSS' | 'BREAKEVEN' | 'TIMEOUT'
    pnl_r   : R-multiple (negative for losses, positive for wins)
    """
    risk = abs(entry - stop_loss)
    if risk == 0:
        return "LOSS", stop_loss, 0, -1.0

    end_i = min(signal_i + max_bars + 1, len(df_15m))
    stop  = stop_loss

    for j in range(signal_i + 1, end_i):
        bar_high = float(df_15m.iloc[j]["high"])
        bar_low  = float(df_15m.iloc[j]["low"])

        if direction == "LONG":
            if bar_low  <= stop:
                if stop >= entry:
                    return "BREAKEVEN", stop, j - signal_i, 0.0
                return "LOSS", stop, j - signal_i, -1.0
            if bar_high >= tp1:
                return "WIN", tp1, j - signal_i, round((tp1 - entry) / risk, 3)
            if bar_high >= entry + risk and stop < entry:
                stop = entry
        else:   # SHORT
            if bar_high >= stop:
                if stop <= entry:
                    return "BREAKEVEN", stop, j - signal_i, 0.0
                return "LOSS", stop, j - signal_i, -1.0
            if bar_low  <= tp1:
                return "WIN", tp1, j - signal_i, round((entry - tp1) / risk, 3)
            if bar_low <= entry - risk and stop > entry:
                stop = entry

    # Timeout — exit at last-bar close
    close_bar = min(signal_i + max_bars, len(df_15m) - 1)
    last_close = float(df_15m.iloc[close_bar]["close"])
    pnl_r = ((last_close - entry) / risk if direction == "LONG"
              else (entry - last_close) / risk)
    return "TIMEOUT", last_close, max_bars, round(pnl_r, 3)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

_WARMUP_BARS  = 300   # skip first 300 15m bars — EMA200 needs 200 1h bars first
_MAX_BARS_HELD = 200  # force-close after 200 bars (~50h) if SL/TP not reached


def run_backtest(
    symbol:          str   = "BTCUSD",
    days:            int   = 90,
    initial_balance: float = 1000.0,
    risk_pct:        float = 1.0,
) -> BacktestResult:
    # ---- Fetch + resample ----
    df_15m = _fetch_15m_candles(symbol, days)
    if df_15m.empty:
        raise RuntimeError(f"No data returned for {symbol}. "
                           "Check API connectivity or symbol name.")

    df_1h = _resample_15m(df_15m, "1h")
    df_4h = _resample_15m(df_15m, "4h")
    df_1d = _resample_15m(df_15m, "1D")

    strategy    = TrendStrategy()
    balance     = initial_balance
    equity      = [balance]
    trades:     list[BacktestTrade] = []
    skip_until  = -1          # bar index: skip signal evaluation while in a trade
    ts_arr      = df_15m["timestamp"].values
    total_bars  = len(df_15m)
    prev_pct    = -1

    print(f"\nWalk-forward simulation on {total_bars:,} 15m bars...", flush=True)

    for i in range(_WARMUP_BARS, total_bars - 1):
        # Progress indicator
        pct = (i - _WARMUP_BARS) * 100 // max(1, total_bars - _WARMUP_BARS)
        if pct // 5 != prev_pct // 5:
            prev_pct = pct
            print(f"  {pct:3d}%  bar {i:,}/{total_bars:,}  "
                  f"trades so far: {len(trades)}", flush=True)

        # One trade at a time — wait until current trade closes
        if i <= skip_until:
            continue

        current_ts = ts_arr[i]

        # Build per-timeframe windows ending at current bar (no look-ahead)
        w15 = df_15m.iloc[max(0, i - 300): i + 1].reset_index(drop=True)

        # Higher-TF: only FULLY CLOSED bars. A bar opening at T closes at
        # T + duration; the signal fires at the current 15m bar close
        # (current_ts + 900). Including the in-progress 1h/4h bar leaked
        # future data into the indicators (look-ahead bias).
        bar_close_ts = current_ts + 900
        idx_1h = int((df_1h["timestamp"].values + 3600   <= bar_close_ts).sum())
        idx_4h = int((df_4h["timestamp"].values + 14400  <= bar_close_ts).sum())
        idx_1d = int((df_1d["timestamp"].values + 86400  <= bar_close_ts).sum())
        w1h = df_1h.iloc[max(0, idx_1h - 250): idx_1h].reset_index(drop=True)
        w4h = df_4h.iloc[max(0, idx_4h - 300): idx_4h].reset_index(drop=True)
        w1d = df_1d.iloc[max(0, idx_1d - 200): idx_1d].reset_index(drop=True)

        # Need enough data for all indicators
        if len(w1h) < 30 or len(w4h) < 20:
            continue

        # Indicators
        ind_15m = calculate_all(w15)
        ind_1h  = calculate_all(w1h)
        ind_4h  = calculate_all(w4h)

        # Patterns + structure + SMC
        patterns  = detect_patterns(w15)
        structure = get_structure_summary(w4h)
        ob_list   = detect_order_blocks(w15)
        fvg_list  = detect_fair_value_gaps(w15)
        smc_data  = {
            "order_blocks":     ob_list,
            "fair_value_gaps":  fvg_list,         # key matches strategy.py expectation
            "liquidity_sweeps": detect_liquidity_sweeps(w15, find_swing_points(w15)),
            "zones":            get_institutional_zones(ob_list, fvg_list),
        }

        mtf    = {"15m": w15, "1h": w1h, "4h": w4h, "1d": w1d}
        setups = strategy.evaluate(
            symbol, mtf, ind_15m, ind_1h, ind_4h,
            patterns, structure, smc_data,
        )
        if not setups:
            continue

        setup = setups[0]   # first setup wins if multiple fire simultaneously

        result, exit_price, bars_held, pnl_r = _simulate_trade(
            df_15m, i, setup.entry, setup.stop_loss, setup.tp1,
            setup.direction, _MAX_BARS_HELD,
        )

        risk_dollars = balance * (risk_pct / 100.0)
        balance     += risk_dollars * pnl_r
        equity.append(balance)

        trades.append(BacktestTrade(
            bar_idx    = i,
            timestamp  = _ts_str(current_ts),
            direction  = setup.direction,
            entry      = round(setup.entry,      4),
            stop_loss  = round(setup.stop_loss,  4),
            tp1        = round(setup.tp1,        4),
            rr_planned = setup.rr,
            confidence = setup.confidence,
            exit_price = round(exit_price, 4),
            result     = result,
            pnl_r      = pnl_r,
            bars_held  = bars_held,
        ))

        skip_until = i + bars_held   # block re-entry until trade closes

    # ---- Statistics ----
    wins       = [t for t in trades if t.result == "WIN"]
    losses     = [t for t in trades if t.result == "LOSS"]
    breakevens = [t for t in trades if t.result == "BREAKEVEN"]
    timeouts   = [t for t in trades if t.result == "TIMEOUT"]
    n          = len(trades)

    win_rate     = round(len(wins) / n * 100, 1)             if n         else 0.0
    avg_rr       = round(sum(t.pnl_r for t in trades) / n, 3) if n         else 0.0
    avg_winner_r = round(sum(t.pnl_r for t in wins)   / len(wins),   3) if wins   else 0.0
    avg_loser_r  = round(sum(t.pnl_r for t in losses) / len(losses), 3) if losses else 0.0

    # Max drawdown from equity curve
    max_dd = 0.0
    peak   = initial_balance
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return BacktestResult(
        symbol             = symbol,
        period_days        = days,
        start_date         = _ts_str(float(ts_arr[_WARMUP_BARS])),
        end_date           = _ts_str(float(ts_arr[-1])),
        total_bars_scanned = total_bars - _WARMUP_BARS,
        total_trades       = n,
        wins               = len(wins),
        losses             = len(losses),
        breakevens         = len(breakevens),
        timeouts           = len(timeouts),
        win_rate_pct       = win_rate,
        avg_rr             = avg_rr,
        avg_winner_r       = avg_winner_r,
        avg_loser_r        = avg_loser_r,
        max_drawdown_pct   = round(max_dd, 2),
        final_balance      = round(balance, 2),
        total_return_pct   = round((balance - initial_balance) / initial_balance * 100, 2),
        initial_balance    = initial_balance,
        trades             = [asdict(t) for t in trades],
    )


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print_report(r: BacktestResult) -> None:
    bar = "-" * 52
    print(f"\n{bar}")
    print(f"  BACKTEST  {r.symbol}  |  {r.period_days}d")
    print(bar)
    print(f"  Period          {r.start_date}  to  {r.end_date}")
    print(f"  Bars scanned    {r.total_bars_scanned:,}")
    print(f"  Total trades    {r.total_trades}")
    if r.total_trades:
        print(f"  Wins            {r.wins}  ({r.win_rate_pct:.1f}%)")
        print(f"  Losses          {r.losses}")
        print(f"  Breakevens      {r.breakevens}")
        print(f"  Timeouts        {r.timeouts}")
        print(bar)
        print(f"  Average R       {r.avg_rr:+.3f}")
        print(f"  Avg winner      {r.avg_winner_r:+.3f} R")
        print(f"  Avg loser       {r.avg_loser_r:+.3f} R")
        print(bar)
        print(f"  Max drawdown    {r.max_drawdown_pct:.2f}%")
        print(f"  Starting bal    ${r.initial_balance:,.2f}")
        print(f"  Final balance   ${r.final_balance:,.2f}")
        print(f"  Total return    {r.total_return_pct:+.2f}%")
        print(bar)

        recent = r.trades[-10:]
        if recent:
            print(f"\n  Last {len(recent)} trades:")
            for t in recent:
                icons = {"WIN": "W", "LOSS": "L", "BREAKEVEN": "B"}
                icon = icons.get(t["result"], "T")
                print(
                    f"    {icon}  {t['timestamp']}  {t['direction']:<5}  "
                    f"entry={t['entry']:>9.2f}  sl={t['stop_loss']:>9.2f}  "
                    f"exit={t['exit_price']:>9.2f}  {t['pnl_r']:+.2f}R  [{t['result']}]"
                )
    else:
        print("  No trades fired during this period.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward backtester for TrendStrategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--symbol",  default="BTCUSD",  help="Trading symbol")
    ap.add_argument("--days",    default=90,  type=int,   help="History window in days")
    ap.add_argument("--balance", default=1000.0, type=float, help="Starting balance USD")
    ap.add_argument("--risk",    default=1.0, type=float, help="Risk %% per trade")
    ap.add_argument("--out",     default="backtest_results.json", help="Output JSON path")
    args = ap.parse_args()

    print(f"TrendStrategy backtester  |  {args.symbol}  "
          f"{args.days}d  ${args.balance:.0f}  {args.risk}% risk/trade")

    try:
        result = run_backtest(
            symbol          = args.symbol,
            days            = args.days,
            initial_balance = args.balance,
            risk_pct        = args.risk,
        )
    except Exception as exc:
        print(f"\nBacktest failed: {exc}", file=sys.stderr)
        logger.exception("run_backtest error")
        sys.exit(1)

    _print_report(result)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)
    print(f"Results saved -> {args.out}\n")


if __name__ == "__main__":
    main()
