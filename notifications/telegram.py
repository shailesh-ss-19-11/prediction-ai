"""
Telegram notifier — sends only what matters: direction, entry, SL, TP, amount.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, token: str, chat_ids):
        self.token    = token
        self.chat_ids = [chat_ids] if isinstance(chat_ids, str) else [c for c in chat_ids if c]

    def _post(self, text: str) -> bool:
        if not self.token or not self.chat_ids:
            logger.warning("Telegram not configured — message not sent:\n%s", text)
            return False

        url     = _API.format(token=self.token)
        success = False
        for chat_id in self.chat_ids:
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            sent = False
            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=10)
                    r.raise_for_status()
                    sent = True
                    success = True
                    logger.debug("Telegram message sent to chat_id=%s", chat_id)
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning("Telegram chat_id=%s attempt %d/3 failed: %s",
                                   chat_id, attempt + 1, e)
                    if attempt < 2:
                        time.sleep(5)
            if not sent:
                logger.error("Telegram message failed for chat_id=%s after 3 attempts", chat_id)
        return success

    def send_signal(self, symbol: str, direction: str,
                    entry: float, sl: float, tp1: float, tp2: float,
                    lot_size: float, risk_dollars: float) -> bool:

        icon   = "📈" if direction == "LONG" else "📉"
        action = "BUY" if direction == "LONG" else "SELL"
        sl_dist = abs(entry - sl)
        rr      = round(abs(tp1 - entry) / sl_dist, 1) if sl_dist else 0

        def fmt(p): return f"{p:,.2f}"

        text = (
            f"{icon} <b>{symbol} — {action}</b>\n\n"
            f"📍 Entry:     <b>${fmt(entry)}</b>\n"
            f"🛑 Stop Loss: <b>${fmt(sl)}</b>\n"
            f"🎯 Target 1:  <b>${fmt(tp1)}</b>\n"
            f"🎯 Target 2:  <b>${fmt(tp2)}</b>\n\n"
            f"💰 Buy <b>{lot_size:.3f} lots</b>  (risk ${risk_dollars:.2f})\n"
            f"📊 R:R  1:{rr}\n\n"
            f"⚠️ Risk max 1% of your balance only."
        )
        return self._post(text)

    def send_paper_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        reason: str,
        entry: float = 0.0,
        exit_price: float = 0.0,
    ) -> bool:
        if reason == "sl":
            header = "🛑 <b>Stop Loss Hit</b>"
            outcome = "LOSS"
        elif reason in ("tp1", "tp2"):
            header = "🎯 <b>Take Profit Hit</b>"
            outcome = "PROFIT"
        else:
            header = "📋 <b>Trade Closed</b>"
            outcome = "PROFIT" if pnl >= 0 else "LOSS"

        pnl_icon = "✅" if pnl >= 0 else "❌"
        tp_label = reason.upper() if reason.startswith("tp") else reason.upper()

        lines = [
            f"{pnl_icon} {header} — <b>{symbol}</b>",
            f"",
            f"Direction:  <b>{direction}</b>",
            f"Exit:       <b>{tp_label}</b>",
        ]
        if entry and exit_price:
            lines.append(f"Entry:      <b>${entry:,.2f}</b>")
            lines.append(f"Exit Price: <b>${exit_price:,.2f}</b>")
        lines.append(f"PnL:        <b>${pnl:+.2f}</b>  ({outcome})")

        logger.info("Paper trade closed: %s %s | reason=%s | PnL=$%.2f",
                    symbol, direction, reason, pnl)
        return self._post("\n".join(lines))

    def send_text(self, text: str) -> bool:
        return self._post(text)
