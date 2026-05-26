"""
Paper trading simulation engine.

Simulates trade execution using live or historical prices without real money.
Tracks all trades, checks SL/TP hits on price updates, and persists state to JSON.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PaperTrade:
    id: str
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    size: float             # Units / contracts
    status: str             # 'open' | 'closed' | 'cancelled'
    opened_at: str          # ISO datetime string
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Paper trading simulation.

    Tracks positions, simulates SL/TP execution on price updates,
    computes statistics, and persists to a JSON file.
    """

    def __init__(self, initial_balance: float = 1000.0) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive.")

        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._open_trades: Dict[str, PaperTrade] = {}   # id -> PaperTrade
        self._closed_trades: List[PaperTrade] = []

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        size: float,
    ) -> PaperTrade:
        """
        Open a new paper trade.

        Parameters
        ----------
        symbol    : Instrument symbol (e.g. 'BTCUSD').
        direction : 'LONG' or 'SHORT'.
        entry     : Simulated fill price.
        sl        : Stop-loss price.
        tp1       : First take-profit price.
        tp2       : Second take-profit price.
        size      : Number of units / contracts.

        Returns
        -------
        PaperTrade dataclass.
        """
        trade_id = str(uuid.uuid4())[:8]
        trade = PaperTrade(
            id=trade_id,
            symbol=symbol.upper(),
            direction=direction.upper(),
            entry=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            size=size,
            status="open",
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        self._open_trades[trade_id] = trade

        logger.info(
            "[Paper] OPEN %s %s @ %.4f | size=%.4f | SL=%.4f | TP1=%.4f | TP2=%.4f | id=%s",
            trade.direction, trade.symbol, entry, size, sl, tp1, tp2, trade_id,
        )
        return trade

    def update(self, prices: dict) -> List[PaperTrade]:
        """
        Check all open trades against current market prices and close
        those that have hit their SL or TP levels.

        Parameters
        ----------
        prices : dict mapping symbol (str) -> current price (float).

        Returns
        -------
        List of PaperTrade objects that were closed during this update.
        """
        closed_this_update: List[PaperTrade] = []
        to_close: List[tuple] = []   # (trade_id, exit_price, reason)

        for trade_id, trade in self._open_trades.items():
            price = prices.get(trade.symbol) or prices.get(trade.symbol.lower())
            if price is None:
                continue

            direction = trade.direction

            if direction == "LONG":
                # SL hit
                if price <= trade.stop_loss:
                    to_close.append((trade_id, trade.stop_loss, "sl"))
                # TP2 hit
                elif price >= trade.tp2:
                    to_close.append((trade_id, trade.tp2, "tp2"))
                # TP1 hit (partial — close whole position at TP1 for simplicity)
                elif price >= trade.tp1:
                    to_close.append((trade_id, trade.tp1, "tp1"))

            else:  # SHORT
                if price >= trade.stop_loss:
                    to_close.append((trade_id, trade.stop_loss, "sl"))
                elif price <= trade.tp2:
                    to_close.append((trade_id, trade.tp2, "tp2"))
                elif price <= trade.tp1:
                    to_close.append((trade_id, trade.tp1, "tp1"))

        for trade_id, exit_price, reason in to_close:
            closed_trade = self.close_trade(trade_id, exit_price, reason)
            closed_this_update.append(closed_trade)

        return closed_this_update

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> PaperTrade:
        """
        Manually close an open trade.

        Parameters
        ----------
        trade_id   : Trade ID string.
        exit_price : Price at which the trade is closed.
        reason     : Closure reason label.

        Returns
        -------
        The closed PaperTrade.

        Raises
        ------
        KeyError if the trade_id is not found in open trades.
        """
        if trade_id not in self._open_trades:
            raise KeyError(f"Trade {trade_id} not found in open trades.")

        trade = self._open_trades.pop(trade_id)

        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry) * trade.size
        else:
            pnl = (trade.entry - exit_price) * trade.size

        trade.exit_price = exit_price
        trade.pnl        = round(pnl, 6)
        trade.status     = "closed"
        trade.exit_reason = reason
        trade.closed_at  = datetime.now(timezone.utc).isoformat()

        self._balance += pnl
        self._closed_trades.append(trade)

        logger.info(
            "[Paper] CLOSE %s %s @ %.4f | PnL $%.4f | reason=%s | id=%s",
            trade.direction, trade.symbol, exit_price, pnl, reason, trade_id,
        )
        return trade

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return performance statistics for all closed trades.

        Keys: win_rate, total_trades, winning_trades, losing_trades,
              total_pnl, avg_rr, current_balance, best_trade, worst_trade.
        """
        trades = self._closed_trades
        total  = len(trades)

        if total == 0:
            return {
                "win_rate":       0.0,
                "total_trades":   0,
                "winning_trades": 0,
                "losing_trades":  0,
                "total_pnl":      0.0,
                "avg_rr":         0.0,
                "current_balance": round(self._balance, 4),
                "best_trade":     None,
                "worst_trade":    None,
            }

        winners = [t for t in trades if t.pnl > 0]
        losers  = [t for t in trades if t.pnl <= 0]
        pnls    = [t.pnl for t in trades]
        total_pnl = sum(pnls)

        # Average R:R achieved
        rr_values = []
        for t in trades:
            sl_dist = abs(t.entry - t.stop_loss)
            if sl_dist > 0 and t.exit_price is not None:
                achieved = abs(t.exit_price - t.entry)
                rr_values.append(achieved / sl_dist)
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0

        best  = max(trades, key=lambda t: t.pnl)
        worst = min(trades, key=lambda t: t.pnl)

        return {
            "win_rate":        round(len(winners) / total * 100, 2),
            "total_trades":    total,
            "winning_trades":  len(winners),
            "losing_trades":   len(losers),
            "total_pnl":       round(total_pnl, 4),
            "avg_rr":          round(avg_rr, 4),
            "current_balance": round(self._balance, 4),
            "best_trade":      {
                "id": best.id, "symbol": best.symbol,
                "pnl": round(best.pnl, 4), "reason": best.exit_reason,
            },
            "worst_trade":     {
                "id": worst.id, "symbol": worst.symbol,
                "pnl": round(worst.pnl, 4), "reason": worst.exit_reason,
            },
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_trades(self) -> List[PaperTrade]:
        """Return a list of all currently open trades."""
        return list(self._open_trades.values())

    def get_trade_history(self) -> List[PaperTrade]:
        """Return a list of all closed trades."""
        return list(self._closed_trades)

    @property
    def balance(self) -> float:
        return self._balance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, filepath: str = "paper_trades.json") -> None:
        """
        Persist all trades (open and closed) to a JSON file.

        Parameters
        ----------
        filepath : Path to the target JSON file.
        """
        data = {
            "initial_balance": self._initial_balance,
            "current_balance": self._balance,
            "open_trades":     [asdict(t) for t in self._open_trades.values()],
            "closed_trades":   [asdict(t) for t in self._closed_trades],
        }
        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info("[Paper] State saved to %s.", filepath)
        except OSError as exc:
            logger.error("[Paper] Failed to save state: %s", exc)

    def load_from_file(self, filepath: str = "paper_trades.json") -> None:
        """
        Restore trades from a JSON file previously created by save_to_file().

        Parameters
        ----------
        filepath : Path to the source JSON file.
        """
        if not os.path.exists(filepath):
            logger.warning("[Paper] File not found: %s. Nothing loaded.", filepath)
            return

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            self._initial_balance = data.get("initial_balance", self._initial_balance)
            self._balance         = data.get("current_balance", self._balance)

            self._open_trades = {}
            for td in data.get("open_trades", []):
                t = PaperTrade(**{k: v for k, v in td.items()})
                self._open_trades[t.id] = t

            self._closed_trades = [
                PaperTrade(**{k: v for k, v in td.items()})
                for td in data.get("closed_trades", [])
            ]
            logger.info(
                "[Paper] Loaded %d open / %d closed trades from %s.",
                len(self._open_trades), len(self._closed_trades), filepath,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.error("[Paper] Failed to load state from %s: %s", filepath, exc)
