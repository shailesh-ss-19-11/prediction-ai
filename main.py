"""
DeltaSignalBot v2 — WebSocket prices + strategy signals.
Sends to Telegram ONLY when 6/7 strategy conditions pass.
Message contains: Entry, SL, TP1, TP2, lot size. Nothing else.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone

import requests
import schedule

import api as api_server
import config
import trade_journal
from exchanges.delta_exchange import DeltaExchange
from exchanges.ws_stream import start as ws_start, get_price as ws_price
from core.market_data import MarketDataManager
from core.indicators import calculate_all
from core.patterns import detect_all as detect_patterns
from core.market_structure import get_structure_summary, find_swing_points
from core.smc import (detect_order_blocks, detect_fair_value_gaps,
                      detect_liquidity_sweeps, get_institutional_zones)
from core.strategy import TrendStrategy
from core.advanced_filters import VolatilityFilter, OvertradingFilter, SessionFilter
from risk.risk_manager import RiskManager
from execution.paper_trading import PaperTradingEngine
from notifications.telegram import TelegramNotifier

def _setup_logging() -> logging.Logger:
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # capture everything; handlers filter by level

    # Console: INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if config.LOG_TO_FILE:
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # Main log file: DEBUG and above (full detail)
        file_handler = logging.handlers.RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        # Error-only log file
        error_handler = logging.handlers.RotatingFileHandler(
            config.LOG_ERROR_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)

    return logging.getLogger("main")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
exchange    = DeltaExchange(api_key=config.DELTA_API_KEY, api_secret=config.DELTA_API_SECRET)
market_data = MarketDataManager(exchange)
strategy    = TrendStrategy()
risk_mgr    = RiskManager(config.ACCOUNT_BALANCE, config.MAX_RISK_PERCENT)
paper       = PaperTradingEngine(config.ACCOUNT_BALANCE)
telegram    = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_IDS)

vol_filter     = VolatilityFilter()
ot_filter      = OvertradingFilter(max_signals_per_day=3)
session_filter = SessionFilter()

# ---------------------------------------------------------------------------
# Live balance refresh
# ---------------------------------------------------------------------------

def _refresh_balance() -> float:
    """
    Fetch real wallet balance from Delta Exchange and sync it into the
    risk manager so position sizing always reflects the actual account.
    Falls back to config.ACCOUNT_BALANCE if the API call fails.
    """
    try:
        balances = exchange.fetch_balance()
        if not balances:
            logger.warning("fetch_balance returned empty — using config balance")
            return risk_mgr.account_balance

        # Delta Exchange India uses USDT as main collateral; fall back to USD
        usd_balance = (
            balances.get("USDT")
            or balances.get("USD")
            or balances.get("usdt")
            or max(balances.values(), default=0)
        )
        usd_balance = round(float(usd_balance), 4)

        if usd_balance > 0 and usd_balance != risk_mgr.account_balance:
            logger.info(
                "Balance updated: $%.4f → $%.4f", risk_mgr.account_balance, usd_balance
            )
            risk_mgr.account_balance = usd_balance

        return usd_balance

    except Exception:
        logger.exception("_refresh_balance failed — keeping current balance")
        return risk_mgr.account_balance


# ---------------------------------------------------------------------------
# Outbound IP monitor
# ---------------------------------------------------------------------------

_prev_ip: str = ""


def check_ip_change() -> None:
    global _prev_ip
    try:
        current_ip = requests.get("https://api.ipify.org", timeout=10).text.strip()
    except Exception as exc:
        logger.warning("IP check failed: %s", exc)
        return

    if not _prev_ip:
        # First run — record IP and notify via Telegram
        _prev_ip = current_ip
        logger.info("Outbound IP recorded: %s", current_ip)
        telegram.send_text(
            f"🌐 <b>Bot Started — Outbound IP</b>\n\n"
            f"IP: <code>{current_ip}</code>\n\n"
            f"Whitelist this IP on Delta Exchange if not already done."
        )
        return

    if current_ip != _prev_ip:
        logger.warning("Outbound IP changed: %s → %s", _prev_ip, current_ip)
        telegram.send_text(
            f"⚠️ <b>Outbound IP Changed!</b>\n\n"
            f"Old IP: <code>{_prev_ip}</code>\n"
            f"New IP: <code>{current_ip}</code>\n\n"
            f"Update Delta Exchange API key whitelist with the new IP, "
            f"otherwise live trading will fail."
        )
        _prev_ip = current_ip


# Cooldown: don't repeat same direction signal within 4 hours
# Key format: "BTCUSD_LONG" or "BTCUSD_SHORT"
# Persisted to disk so a process restart can't bypass the cooldown
# (this previously caused duplicate signals minutes apart).
_COOLDOWN_FILE = "data/cooldowns.json"
_last_signal: dict[str, float] = {}
_COOLDOWN = 4 * 3600


def _load_cooldowns() -> None:
    global _last_signal
    try:
        if os.path.exists(_COOLDOWN_FILE):
            with open(_COOLDOWN_FILE, "r", encoding="utf-8") as fh:
                _last_signal = {k: float(v) for k, v in json.load(fh).items()}
            logger.info("Loaded %d cooldown entries", len(_last_signal))
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("Failed to load cooldowns — starting fresh")
        _last_signal = {}


def _save_cooldowns() -> None:
    try:
        os.makedirs(os.path.dirname(_COOLDOWN_FILE), exist_ok=True)
        with open(_COOLDOWN_FILE, "w", encoding="utf-8") as fh:
            json.dump(_last_signal, fh)
    except OSError:
        logger.exception("Failed to save cooldowns")


def _on_cooldown(key: str) -> bool:
    return (time.time() - _last_signal.get(key, 0)) < _COOLDOWN


def _mark(key: str) -> None:
    _last_signal[key] = time.time()
    _save_cooldowns()


# ---------------------------------------------------------------------------
# Closed-candle helper
# ---------------------------------------------------------------------------

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _drop_forming_candle(df, timeframe: str):
    """
    Delta's candle endpoint includes the still-forming candle (end = now).
    Evaluating indicators/entries on it makes signals repaint and fire
    mid-spike at momentum extremes — a major cause of instant SL hits.
    Drop the last row when its bar period has not yet completed.
    """
    if df is None or df.empty:
        return df
    tf_secs = _TF_SECONDS.get(timeframe)
    if not tf_secs:
        return df
    last_open = float(df["timestamp"].iloc[-1])
    if time.time() < last_open + tf_secs:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-symbol check
# ---------------------------------------------------------------------------

def _place_live_order(symbol: str, setup, risk) -> None:
    """Place a limit entry order on Delta Exchange when AUTO_TRADE is enabled."""
    try:
        side = "buy" if setup.direction == "LONG" else "sell"

        # Delta Exchange India uses integer contract counts (1 contract = 1 USD notional).
        # Cap at MAX_CONTRACTS_PER_TRADE to limit real-money exposure.
        contracts = min(
            max(1, round(risk.lot_size * setup.entry)),
            config.MAX_CONTRACTS_PER_TRADE,
        )

        order = exchange.place_order(
            symbol     = symbol,
            side       = side,
            order_type = "limit",
            amount     = contracts,
            price      = setup.entry,
        )

        logger.info(
            "LIVE ORDER PLACED: %s %s | id=%s | entry=%.2f | contracts=%d | sl=%.2f | tp1=%.2f",
            symbol, setup.direction, order.order_id, setup.entry, contracts,
            setup.stop_loss, setup.tp1,
        )

        telegram.send_text(
            f"✅ <b>Order Placed — Delta Exchange</b>\n"
            f"<b>{symbol} {setup.direction}</b>\n"
            f"├ Order ID: <code>{order.order_id}</code>\n"
            f"├ Entry:    <b>${setup.entry:,.2f}</b> (limit)\n"
            f"├ SL:       ${setup.stop_loss:,.2f}\n"
            f"├ TP1:      ${setup.tp1:,.2f}\n"
            f"├ TP2:      ${setup.tp2:,.2f}\n"
            f"└ Size:     {contracts} contracts"
        )

    except Exception:
        logger.exception("AUTO_TRADE order failed for %s %s", symbol, setup.direction)
        telegram.send_text(
            f"⚠️ <b>Auto-order FAILED</b> — {symbol} {setup.direction}\n"
            f"Place manually at ${setup.entry:,.2f}"
        )


def check_symbol(symbol: str) -> None:
    try:
        # Block trading during Asian session (00:00–07:00 UTC) — low liquidity,
        # erratic moves, most false signals fire here.
        active_sessions = session_filter.get_active_sessions()
        if not active_sessions or active_sessions == ["asian"]:
            logger.info("%s: Asian/off-hours session only (%s) — skipping",
                        symbol, active_sessions or "none")
            return

        signals_used = ot_filter.signals_today(symbol)
        if not ot_filter.can_trade(symbol):
            logger.info("%s: daily signal cap reached (%d/%d) — skipping",
                        symbol, signals_used, 3)
            return

        # Use WebSocket price if available, else REST fallback
        price_source = "WS"
        current_price = ws_price(symbol)
        if not current_price:
            price_source = "REST"
            current_price = market_data.get_current_price(symbol)
        if not current_price:
            logger.warning("%s: no price available from WS or REST", symbol)
            return
        logger.debug("%s: price $%.2f (source=%s)", symbol, current_price, price_source)

        # Fetch candles for strategy (REST)
        mtf = market_data.fetch_mtf_candles(symbol)
        if not mtf or not all(tf in mtf for tf in ("15m", "1h", "4h")):
            missing = [tf for tf in ("15m", "1h", "4h") if tf not in (mtf or {})]
            logger.warning("%s: missing candle data for timeframes: %s", symbol, missing)
            return

        # Analyse CLOSED candles only — the API includes the forming bar,
        # which makes every indicator and the entry price repaint.
        for tf in ("15m", "1h", "4h"):
            mtf[tf] = _drop_forming_candle(mtf[tf], tf)
            if mtf[tf] is None or mtf[tf].empty:
                logger.warning("%s: no closed candles for %s", symbol, tf)
                return

        df_15m, df_1h, df_4h = mtf["15m"], mtf["1h"], mtf["4h"]

        # Indicators
        ind_15m = calculate_all(df_15m)
        ind_1h  = calculate_all(df_1h)
        ind_4h  = calculate_all(df_4h)
        logger.debug(
            "%s indicators — RSI15m:%.1f EMA20_1h:%.2f EMA200_4h:%.2f ATR:%.4f",
            symbol, ind_15m.rsi_14, ind_1h.ema_20, ind_4h.ema_200, ind_15m.atr_14,
        )

        # Skip extreme volatility (news spikes)
        vol_state = vol_filter.get_volatility_state(df_15m, ind_15m.atr_14)
        if vol_state == "extreme":
            logger.info("%s: extreme volatility — skipping", symbol)
            return
        if vol_state != "normal":
            logger.debug("%s: volatility state = %s", symbol, vol_state)

        # Strategy analysis
        pattern   = detect_patterns(df_15m)
        structure = get_structure_summary(df_4h)
        bias      = structure.get("bias", "?")
        logger.debug("%s: 4h structure bias=%s | patterns detected=%d",
                     symbol, bias, len(pattern))

        ob_list = detect_order_blocks(df_15m)
        fvg_list = detect_fair_value_gaps(df_15m)
        swings_15m = find_swing_points(df_15m)
        smc_data  = {
            "order_blocks":     ob_list,
            "fair_value_gaps":  fvg_list,          # key must match strategy.py
            "liquidity_sweeps": detect_liquidity_sweeps(df_15m, swings_15m),
            "zones":            get_institutional_zones(ob_list, fvg_list),
        }
        logger.debug("%s: SMC — OBs=%d FVGs=%d",
                     symbol, len(ob_list), len(fvg_list))

        setups = strategy.evaluate(symbol, mtf, ind_15m, ind_1h, ind_4h,
                                   pattern, structure, smc_data)

        if not setups:
            logger.info("%s $%s | trend:%s | bias:%s | RSI:%.1f | no setup",
                        symbol, f"{current_price:,.2f}",
                        ind_4h.trend_ema, bias, ind_15m.rsi_14)
            return

        logger.info("%s: %d setup(s) found", symbol, len(setups))

        for setup in setups:
            cooldown_key = f"{symbol}_{setup.direction}"
            if _on_cooldown(cooldown_key):
                remaining = int(_COOLDOWN - (time.time() - _last_signal.get(cooldown_key, 0)))
                logger.info("%s %s: cooldown active (%ds remaining)",
                            symbol, setup.direction, remaining)
                continue

            # Never stack a second position in the same symbol+direction
            # (restart-safety: cooldown alone can be lost with the process)
            if any(t.symbol == symbol and t.direction == setup.direction
                   for t in paper.get_open_trades()):
                logger.info("%s %s: position already open — skipping duplicate",
                            symbol, setup.direction)
                continue

            # Entry computed on the last closed candle — if live price has
            # already drifted more than 25% of the risk away, the planned SL
            # no longer protects the same move. Skip the late entry.
            sl_risk = abs(setup.entry - setup.stop_loss)
            if sl_risk > 0 and abs(current_price - setup.entry) > 0.25 * sl_risk:
                logger.info(
                    "%s %s: price drifted %.2f from entry %.2f (> 25%% of risk %.2f) — skipping",
                    symbol, setup.direction, current_price - setup.entry,
                    setup.entry, sl_risk,
                )
                continue

            # Risk check
            risk = risk_mgr.evaluate_trade(setup.entry, setup.stop_loss, setup.tp1)
            if not risk.can_trade:
                logger.info("%s %s: risk check failed — %s",
                            symbol, setup.direction, risk.reason)
                continue

            logger.info(
                "%s %s: setup valid | conf=%.0f%% | entry=%.2f | sl=%.2f | "
                "tp1=%.2f | rr=%.2f | lots=%.4f",
                symbol, setup.direction, setup.confidence * 100,
                setup.entry, setup.stop_loss, setup.tp1, setup.rr, risk.lot_size,
            )
            if setup.reasons:
                logger.debug("%s %s: conditions — %s",
                             symbol, setup.direction, " | ".join(setup.reasons))

            # Paper trade — open position and capture the trade ID for journal
            trade_id = None
            if config.PAPER_TRADING_MODE:
                trade = paper.open_trade(symbol, setup.direction, setup.entry,
                                         setup.stop_loss, setup.tp1, setup.tp2, risk.lot_size)
                trade_id = trade.id
                paper.save_to_file("paper_trades.json")

            # Send signal to Telegram
            sent = telegram.send_signal(
                symbol       = symbol,
                direction    = setup.direction,
                entry        = setup.entry,
                sl           = setup.stop_loss,
                tp1          = setup.tp1,
                tp2          = setup.tp2,
                lot_size     = risk.lot_size,
                risk_dollars = risk.risk_dollars,
            )

            if sent:
                _mark(cooldown_key)
                ot_filter.record_signal(symbol)
                logger.info("SIGNAL SENT: %s %s | Entry %.2f | SL %.2f | TP %.2f | Lots %.3f",
                            symbol, setup.direction, setup.entry,
                            setup.stop_loss, setup.tp1, risk.lot_size)

                # Auto-execute limit order on Delta Exchange
                if config.AUTO_TRADE and not config.PAPER_TRADING_MODE and config.DELTA_API_KEY:
                    _place_live_order(symbol, setup, risk)

                # Persist to trade journal
                trade_journal.record_open(
                    trade_id     = trade_id or f"{symbol}_{int(time.time())}",
                    symbol       = symbol,
                    direction    = setup.direction,
                    entry        = setup.entry,
                    stop_loss    = setup.stop_loss,
                    tp1          = setup.tp1,
                    tp2          = setup.tp2,
                    lot_size     = risk.lot_size,
                    risk_dollars = risk.risk_dollars,
                    confidence   = setup.confidence,
                    rr           = setup.rr,
                    reasons      = setup.reasons,
                )
            else:
                logger.error("%s %s: Telegram send failed", symbol, setup.direction)

    except Exception:
        logger.exception("Error in check_symbol(%s)", symbol)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_status_lines: list[str] = []   # collects per-symbol status each scan


def run_checks() -> None:
    global _status_lines
    _status_lines = []
    scan_start = time.time()
    logger.info("=== Scan started | symbols=%s ===", config.SYMBOLS)

    # Sync real balance before every scan so position sizing is accurate
    if not config.PAPER_TRADING_MODE and config.DELTA_API_KEY:
        _refresh_balance()

    # Close paper trades that hit SL/TP
    try:
        prices = {s: (ws_price(s) or market_data.get_current_price(s))
                  for s in config.SYMBOLS}
        prices = {k: v for k, v in prices.items() if v}
        logger.debug("Live prices: %s", {k: f"{v:,.2f}" for k, v in prices.items()})
        closed_trades = paper.update(prices)
        # Save every scan — update() can also move stops to breakeven
        # without closing, and that state must survive a restart
        paper.save_to_file("paper_trades.json")
        if closed_trades:
            logger.info("Paper trades closed this scan: %d", len(closed_trades))
        for closed in closed_trades:
            # Update journal with outcome before sending Telegram message
            trade_journal.record_close(
                trade_id    = closed.id,
                exit_price  = closed.exit_price or 0.0,
                exit_reason = closed.exit_reason,
                pnl         = closed.pnl,
            )
            telegram.send_paper_closed(
                closed.symbol, closed.direction, closed.pnl, closed.exit_reason,
                entry=closed.entry, exit_price=closed.exit_price or 0.0,
            )
    except Exception:
        logger.exception("Paper update error")

    for symbol in config.SYMBOLS:
        check_symbol(symbol)

    elapsed = time.time() - scan_start
    logger.info("=== Scan complete in %.1fs ===", elapsed)


def send_hourly_status() -> None:
    """Send a brief Telegram status every hour so user knows bot is alive."""
    try:
        lines = ["🤖 <b>Bot Status — No signal yet</b>\n"]
        for symbol in config.SYMBOLS:
            price = ws_price(symbol) or market_data.get_current_price(symbol)
            if price:
                lines.append(f"• <b>{symbol}</b>  ${price:,.2f}")
        lines.append(f"\n⏳ Scanning every {config.CHECK_INTERVAL_MINUTES} min")
        lines.append("🔍 Waiting for 6/7 conditions to align...")
        telegram.send_text("\n".join(lines))
    except Exception:
        logger.exception("Hourly status error")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("DeltaSignalBot v2 — signals only when 6/7 conditions pass")

    if "--setup" in sys.argv:
        print("Fill config.py then run: python main.py")
        sys.exit(0)

    # Restore state — without this every restart forgot open trades and
    # cooldowns, so positions were duplicated and journal entries never closed
    paper.load_from_file("paper_trades.json")
    _load_cooldowns()

    # Start REST API server in background thread
    api_server.start_api_thread(paper)

    # Start WebSocket price stream in background
    ws_start()
    logger.info("Waiting 3s for WebSocket to connect...")
    time.sleep(3)

    telegram.send_text(
        f"✅ <b>DeltaSignalBot v2 started</b>\n"
        f"Symbols: <b>{', '.join(config.SYMBOLS)}</b>\n"
        f"Mode: <b>{'Paper' if config.PAPER_TRADING_MODE else 'Live'}</b> | "
        f"Every <b>{config.CHECK_INTERVAL_MINUTES}min</b>"
    )

    check_ip_change()   # record initial IP at startup

    # Fetch real balance at startup
    if not config.PAPER_TRADING_MODE and config.DELTA_API_KEY:
        live_balance = _refresh_balance()
        telegram.send_text(
            f"💰 <b>Account Balance</b>: <code>${live_balance:.4f}</code>\n"
            f"Risk per trade: <b>{config.MAX_RISK_PERCENT}%</b> "
            f"= <code>${live_balance * config.MAX_RISK_PERCENT / 100:.4f}</code>"
        )

    run_checks()
    schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(run_checks)
    schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(check_ip_change)
    schedule.every(1).hours.do(send_hourly_status)
    schedule.every().day.at("00:00").do(ot_filter.reset_daily)

    logger.info("Running. Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(10)
        except KeyboardInterrupt:
            paper.save_to_file("paper_trades.json")
            logger.info("Stopped. Trades saved.")
            break


if __name__ == "__main__":
    main()
