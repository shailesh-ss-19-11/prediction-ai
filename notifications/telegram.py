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
            logger.warning("Telegram not configured:\n%s", text)
            return False

        url     = _API.format(token=self.token)
        success = False
        for chat_id in self.chat_ids:
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=10)
                    r.raise_for_status()
                    success = True
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning("Telegram %s attempt %d/3 failed: %s", chat_id, attempt + 1, e)
                    if attempt < 2:
                        time.sleep(5)
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

    def send_paper_closed(self, symbol: str, direction: str,
                          pnl: float, reason: str) -> bool:
        icon = "✅" if pnl >= 0 else "❌"
        text = (
            f"{icon} <b>Paper Closed — {symbol}</b>\n"
            f"Direction: {direction}\n"
            f"PnL: <b>${pnl:.2f}</b>  ({reason})"
        )
        return self._post(text)

    def send_text(self, text: str) -> bool:
        return self._post(text)
