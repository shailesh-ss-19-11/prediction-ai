"""
DeltaSignalBot v2 — WebSocket prices + strategy signals.
Sends to Telegram ONLY when 6/7 strategy conditions pass.
Message contains: Entry, SL, TP1, TP2, lot size. Nothing else.
"""

import logging
import sys
import time
from datetime import datetime, timezone

import schedule

import config
from exchanges.delta_exchange import DeltaExchange
from exchanges.ws_stream import start as ws_start, get_price as ws_price
from core.market_data import MarketDataManager
from core.indicators import calculate_all
from core.patterns import get_best_pattern
from core.market_structure import get_structure_summary
from core.smc import (detect_order_blocks, detect_fair_value_gaps,
                      detect_liquidity_sweeps, get_institutional_zones)
from core.strategy import TrendStrategy
from core.advanced_filters import VolatilityFilter, OvertradingFilter
from risk.risk_manager import RiskManager
from execution.paper_trading import PaperTradingEngine
from notifications.telegram import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
exchange    = DeltaExchange(api_key=config.DELTA_API_KEY, api_secret=config.DELTA_API_SECRET)
market_data = MarketDataManager(exchange)
strategy    = TrendStrategy()
risk_mgr    = RiskManager(config.ACCOUNT_BALANCE, config.MAX_RISK_PERCENT)
paper       = PaperTradingEngine(config.ACCOUNT_BALANCE)
telegram    = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

vol_filter = VolatilityFilter()
ot_filter  = OvertradingFilter(max_signals_per_day=3)

# Cooldown: don't repeat same direction signal within 4 hours
# Key format: "BTCUSD_LONG" or "BTCUSD_SHORT"
_last_signal: dict[str, float] = {}
_COOLDOWN = 4 * 3600


def _on_cooldown(key: str) -> bool:
    return (time.time() - _last_signal.get(key, 0)) < _COOLDOWN


def _mark(key: str) -> None:
    _last_signal[key] = time.time()


# ---------------------------------------------------------------------------
# Per-symbol check
# ---------------------------------------------------------------------------

def check_symbol(symbol: str) -> None:
    try:
        if not ot_filter.can_trade(symbol):
            return

        # Use WebSocket price if available, else REST fallback
        current_price = ws_price(symbol) or market_data.get_current_price(symbol)
        if not current_price:
            logger.warning("%s: no price available", symbol)
            return

        # Fetch candles for strategy (REST)
        mtf = market_data.fetch_mtf_candles(symbol)
        if not mtf or not all(tf in mtf for tf in ("15m", "1h", "4h")):
            logger.warning("%s: missing candle data", symbol)
            return

        df_15m, df_1h, df_4h = mtf["15m"], mtf["1h"], mtf["4h"]

        # Indicators
        ind_15m = calculate_all(df_15m)
        ind_1h  = calculate_all(df_1h)
        ind_4h  = calculate_all(df_4h)

        # Skip extreme volatility (news spikes)
        if vol_filter.get_volatility_state(df_15m, ind_15m.atr_14) == "extreme":
            logger.info("%s: extreme volatility — skipping", symbol)
            return

        # Strategy analysis
        pattern   = get_best_pattern(df_15m)
        structure = get_structure_summary(df_4h)
        smc_data  = {
            "order_blocks": detect_order_blocks(df_15m),
            "fvg":          detect_fair_value_gaps(df_15m),
            "sweeps":       detect_liquidity_sweeps(df_15m, []),
            "zones":        get_institutional_zones(
                                detect_order_blocks(df_15m),
                                detect_fair_value_gaps(df_15m)
                            ),
        }

        setups = strategy.evaluate(symbol, mtf, ind_15m, ind_1h, ind_4h,
                                   pattern, structure, smc_data)

        if not setups:
            logger.info("%s $%s | trend:%s | structure:%s",
                        symbol, f"{current_price:,.2f}",
                        ind_4h.trend_ema, structure.get("bias", "?"))
            return

        for setup in setups:
            # Per-direction cooldown key e.g. "BTCUSD_LONG"
            cooldown_key = f"{symbol}_{setup.direction}"
            if _on_cooldown(cooldown_key):
                continue

            # Risk check
            risk = risk_mgr.evaluate_trade(setup.entry, setup.stop_loss, setup.tp1)
            if not risk.can_trade:
                logger.info("%s %s: risk check failed — %s",
                            symbol, setup.direction, risk.reason)
                continue

            # Paper trade
            if config.PAPER_TRADING_MODE:
                paper.open_trade(symbol, setup.direction, setup.entry,
                                 setup.stop_loss, setup.tp1, setup.tp2, risk.lot_size)

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
                logger.info("✅ %s %s | Entry %.2f | SL %.2f | TP %.2f | Lots %.3f",
                            symbol, setup.direction, setup.entry,
                            setup.stop_loss, setup.tp1, risk.lot_size)

    except Exception:
        logger.exception("Error in check_symbol(%s)", symbol)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_checks() -> None:
    # Close paper trades that hit SL/TP
    try:
        prices = {s: (ws_price(s) or market_data.get_current_price(s))
                  for s in config.SYMBOLS}
        prices = {k: v for k, v in prices.items() if v}
        for closed in paper.update(prices):
            telegram.send_paper_closed(
                closed.symbol, closed.direction, closed.pnl, closed.exit_reason)
    except Exception:
        logger.exception("Paper update error")

    for symbol in config.SYMBOLS:
        check_symbol(symbol)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("DeltaSignalBot v2 — signals only when 6/7 conditions pass")

    if "--setup" in sys.argv:
        print("Fill config.py then run: python main.py")
        sys.exit(0)

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

    run_checks()
    schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(run_checks)
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
