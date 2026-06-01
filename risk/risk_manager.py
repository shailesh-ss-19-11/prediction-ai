"""
Advanced risk management module.

Provides position sizing, dynamic stop-loss calculation, trailing stop logic,
breakeven management, daily loss / drawdown guards, and formatted Telegram output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    lot_size: float
    risk_dollars: float
    sl_distance: float
    sl_pct: float
    max_loss_today: float
    can_trade: bool
    reason: str


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Central risk management engine.

    Parameters
    ----------
    account_balance   : Starting or current account balance in USD.
    max_risk_pct      : Maximum % of balance to risk per trade (default 1.0%).
    max_daily_loss_pct: Maximum cumulative daily loss as % of balance (default 3.0%).
    max_drawdown_pct  : Maximum peak-to-trough drawdown % before all trading halts
                        (default 10.0%).
    """

    def __init__(
        self,
        account_balance: float,
        max_risk_pct: float = 1.0,
        max_daily_loss_pct: float = 3.0,
        max_drawdown_pct: float = 10.0,
    ) -> None:
        if account_balance <= 0:
            raise ValueError("account_balance must be positive.")

        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry: float,
        stop_loss: float,
        balance: Optional[float] = None,
    ) -> float:
        """
        Calculate lot size that risks exactly max_risk_pct % of balance.

        Returns the number of units (contracts / coins) to trade.
        Returns 0.0 if inputs are invalid.
        """
        bal = balance if balance is not None else self.account_balance
        sl_distance = abs(entry - stop_loss)

        if sl_distance == 0 or entry <= 0 or bal <= 0:
            logger.warning(
                "[RiskManager] Invalid inputs for position size: "
                "entry=%.4f, sl=%.4f, balance=%.2f",
                entry, stop_loss, bal,
            )
            return 0.0

        risk_dollars = bal * (self.max_risk_pct / 100.0)
        lot_size = risk_dollars / sl_distance

        logger.debug(
            "[RiskManager] Position size: %.6f units | "
            "risk $%.2f | SL distance %.4f",
            lot_size, risk_dollars, sl_distance,
        )
        return round(lot_size, 6)

    # ------------------------------------------------------------------
    # Dynamic stop-loss (ATR-based)
    # ------------------------------------------------------------------

    def calculate_dynamic_sl(
        self,
        df: pd.DataFrame,
        direction: str,
        atr: float,
        multiplier: float = 1.5,
        min_sl_pct: float = 0.008,
    ) -> float:
        """
        Calculate a stop-loss price that satisfies three constraints:
          1. Beyond the recent structural swing low/high (10-bar lookback).
          2. At least ``multiplier`` × ATR from entry (default 1.5×).
          3. At least ``min_sl_pct`` % from entry (default 0.8% — crypto floor).

        Parameters
        ----------
        df         : OHLCV DataFrame; last row is the signal candle.
        direction  : 'LONG' or 'SHORT'.
        atr        : Current ATR value (14-period recommended).
        multiplier : ATR multiplier for minimum distance (default 1.5×).
        min_sl_pct : Minimum SL distance as fraction of entry (default 0.008).

        Returns
        -------
        Stop-loss price as a float.
        """
        if df is None or df.empty:
            logger.warning("[RiskManager] Empty DataFrame in calculate_dynamic_sl.")
            return 0.0

        entry = float(df.iloc[-1]["close"])
        direction_upper = direction.upper()

        if direction_upper == "LONG":
            swing_low  = float(df["low"].tail(10).min())
            struct_sl  = swing_low  - 0.2 * atr             # buffer below pivot
            atr_sl     = entry - (atr * multiplier)         # 1.5× ATR minimum
            pct_sl     = entry * (1.0 - min_sl_pct)         # 0.8% minimum
            sl = min(struct_sl, atr_sl, pct_sl)             # widest (lowest) wins
        elif direction_upper == "SHORT":
            swing_high = float(df["high"].tail(10).max())
            struct_sl  = swing_high + 0.2 * atr             # buffer above pivot
            atr_sl     = entry + (atr * multiplier)         # 1.5× ATR minimum
            pct_sl     = entry * (1.0 + min_sl_pct)         # 0.8% minimum
            sl = max(struct_sl, atr_sl, pct_sl)             # widest (highest) wins
        else:
            logger.error("[RiskManager] Unknown direction '%s'.", direction)
            return 0.0

        logger.debug(
            "[RiskManager] Dynamic SL=%.4f for %s (entry=%.4f, ATR=%.4f, mult=%.1f)",
            sl, direction_upper, entry, atr, multiplier,
        )
        return round(sl, 6)

    # ------------------------------------------------------------------
    # Trailing stop-loss
    # ------------------------------------------------------------------

    def calculate_trailing_sl(
        self,
        entry: float,
        current_price: float,
        direction: str,
        atr: float,
        initial_sl: float,
    ) -> float:
        """
        Trail stop-loss by 1× ATR once the trade is in 1R profit.

        The stop only moves in the trade's favour — it never widens.

        Parameters
        ----------
        entry         : Trade entry price.
        current_price : Live market price.
        direction     : 'LONG' or 'SHORT'.
        atr           : Current ATR value.
        initial_sl    : The original stop-loss price set at entry.

        Returns
        -------
        Updated stop-loss price (float).
        """
        direction_upper = direction.upper()
        risk = abs(entry - initial_sl)

        if risk == 0 or atr == 0:
            return initial_sl

        if direction_upper == "LONG":
            profit = current_price - entry
            # Activate trailing only once we are in at least 1R profit
            if profit < risk:
                return initial_sl
            # Trail: keep SL 1× ATR below current price, but never retreat
            new_sl = current_price - atr
            return max(new_sl, initial_sl)

        elif direction_upper == "SHORT":
            profit = entry - current_price
            if profit < risk:
                return initial_sl
            new_sl = current_price + atr
            return min(new_sl, initial_sl)

        return initial_sl

    # ------------------------------------------------------------------
    # Breakeven stop
    # ------------------------------------------------------------------

    def calculate_breakeven(
        self,
        entry: float,
        direction: str,
        risk: float,
        buffer_pct: float = 0.001,
    ) -> float:
        """
        Return the breakeven stop price (entry ± small buffer).

        Call this once TP1 is hit to lock in a scratch trade at worst.

        Parameters
        ----------
        entry      : Trade entry price.
        direction  : 'LONG' or 'SHORT'.
        risk       : Dollar risk per unit (entry - initial_sl for LONG).
        buffer_pct : Small buffer above/below entry as % of entry (default 0.1%).

        Returns
        -------
        Breakeven stop price as a float.
        """
        buffer = entry * buffer_pct
        direction_upper = direction.upper()

        if direction_upper == "LONG":
            return round(entry + buffer, 6)
        elif direction_upper == "SHORT":
            return round(entry - buffer, 6)

        logger.error("[RiskManager] Unknown direction '%s' in calculate_breakeven.", direction)
        return entry

    # ------------------------------------------------------------------
    # Daily loss limit guard
    # ------------------------------------------------------------------

    def check_daily_loss_limit(self, daily_pnl: float) -> bool:
        """
        Return False (block trading) if cumulative daily loss exceeds
        max_daily_loss_pct of account balance.

        Parameters
        ----------
        daily_pnl : Realised P&L for today (negative = loss).

        Returns
        -------
        True  → within limit, trading is allowed.
        False → limit breached, no new trades.
        """
        max_loss = self.account_balance * (self.max_daily_loss_pct / 100.0)
        loss = -daily_pnl  # Convert to positive loss value

        if loss >= max_loss:
            logger.warning(
                "[RiskManager] Daily loss limit hit: -$%.2f >= $%.2f limit.",
                loss, max_loss,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Max drawdown guard
    # ------------------------------------------------------------------

    def check_max_drawdown(
        self,
        peak_balance: float,
        current_balance: float,
    ) -> bool:
        """
        Return False if drawdown from peak exceeds max_drawdown_pct.

        Parameters
        ----------
        peak_balance    : Highest recorded account balance (equity).
        current_balance : Current account balance (equity).

        Returns
        -------
        True  → drawdown within limit.
        False → drawdown exceeded, halt trading.
        """
        if peak_balance <= 0:
            return True

        drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100.0

        if drawdown_pct >= self.max_drawdown_pct:
            logger.warning(
                "[RiskManager] Max drawdown breached: %.2f%% >= %.2f%%.",
                drawdown_pct, self.max_drawdown_pct,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Full trade evaluation
    # ------------------------------------------------------------------

    def evaluate_trade(
        self,
        entry: float,
        stop_loss: float,
        tp1: float,
        balance: Optional[float] = None,
        daily_pnl: float = 0.0,
        peak_balance: Optional[float] = None,
        current_balance: Optional[float] = None,
    ) -> RiskResult:
        """
        Run a complete risk check and return a RiskResult.

        Parameters
        ----------
        entry            : Proposed entry price.
        stop_loss        : Initial stop-loss price.
        tp1              : First take-profit target.
        balance          : Override balance (uses self.account_balance if None).
        daily_pnl        : Today's realised P&L (for daily limit check).
        peak_balance     : Equity peak (for drawdown check).
        current_balance  : Current equity (for drawdown check).

        Returns
        -------
        RiskResult dataclass.
        """
        bal = balance if balance is not None else self.account_balance
        pk  = peak_balance if peak_balance is not None else bal
        cur = current_balance if current_balance is not None else bal

        sl_distance = abs(entry - stop_loss)
        sl_pct = (sl_distance / entry) * 100.0 if entry > 0 else 0.0
        risk_dollars = bal * (self.max_risk_pct / 100.0)
        max_loss_today = bal * (self.max_daily_loss_pct / 100.0)

        lot_size = self.calculate_position_size(entry, stop_loss, bal)

        # --- Guard checks ---
        if not self.check_daily_loss_limit(daily_pnl):
            return RiskResult(
                lot_size=0.0,
                risk_dollars=risk_dollars,
                sl_distance=sl_distance,
                sl_pct=sl_pct,
                max_loss_today=max_loss_today,
                can_trade=False,
                reason=f"Daily loss limit reached (>{self.max_daily_loss_pct}% of balance).",
            )

        if not self.check_max_drawdown(pk, cur):
            return RiskResult(
                lot_size=0.0,
                risk_dollars=risk_dollars,
                sl_distance=sl_distance,
                sl_pct=sl_pct,
                max_loss_today=max_loss_today,
                can_trade=False,
                reason=f"Max drawdown exceeded (>{self.max_drawdown_pct}%).",
            )

        if sl_distance == 0:
            return RiskResult(
                lot_size=0.0,
                risk_dollars=risk_dollars,
                sl_distance=0.0,
                sl_pct=0.0,
                max_loss_today=max_loss_today,
                can_trade=False,
                reason="Entry and stop-loss are identical — invalid trade.",
            )

        # Minimum R:R check (at least 1:1)
        reward = abs(tp1 - entry)
        rr = reward / sl_distance if sl_distance > 0 else 0
        if rr < 1.0:
            return RiskResult(
                lot_size=0.0,
                risk_dollars=risk_dollars,
                sl_distance=sl_distance,
                sl_pct=sl_pct,
                max_loss_today=max_loss_today,
                can_trade=False,
                reason=f"R:R ratio {rr:.2f} is below minimum 1.0.",
            )

        return RiskResult(
            lot_size=lot_size,
            risk_dollars=risk_dollars,
            sl_distance=sl_distance,
            sl_pct=sl_pct,
            max_loss_today=max_loss_today,
            can_trade=True,
            reason="All risk checks passed.",
        )

    # ------------------------------------------------------------------
    # Telegram-ready formatted string
    # ------------------------------------------------------------------

    def format_signal_risk(
        self,
        entry: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        balance: Optional[float] = None,
    ) -> str:
        """
        Return a human-readable risk summary suitable for a Telegram message.

        Example output:
            💰 Risk Management
            ├ Entry:        $65,000.00
            ├ Stop Loss:    $64,000.00  (-1.54%)
            ├ TP1:          $67,000.00  (+3.08%)
            ├ TP2:          $69,000.00  (+6.15%)
            ├ R:R (to TP1): 2.00
            ├ Lot Size:     0.001538
            ├ Risk $:       $10.00
            └ Max Loss/Day: $30.00
        """
        bal = balance if balance is not None else self.account_balance
        result = self.evaluate_trade(entry, stop_loss, tp1, balance=bal)

        sl_distance = abs(entry - stop_loss)
        sl_pct = (sl_distance / entry * 100) if entry else 0.0
        tp1_distance = abs(tp1 - entry)
        tp1_pct = (tp1_distance / entry * 100) if entry else 0.0
        tp2_distance = abs(tp2 - entry)
        tp2_pct = (tp2_distance / entry * 100) if entry else 0.0
        rr = tp1_distance / sl_distance if sl_distance > 0 else 0.0

        sl_sign  = "-" if stop_loss < entry else "+"
        tp1_sign = "+" if tp1 > entry else "-"
        tp2_sign = "+" if tp2 > entry else "-"

        lines = [
            "💰 <b>Risk Management</b>",
            f"├ Entry:        ${entry:,.4f}",
            f"├ Stop Loss:    ${stop_loss:,.4f}  ({sl_sign}{sl_pct:.2f}%)",
            f"├ TP1:          ${tp1:,.4f}  ({tp1_sign}{tp1_pct:.2f}%)",
            f"├ TP2:          ${tp2:,.4f}  ({tp2_sign}{tp2_pct:.2f}%)",
            f"├ R:R (to TP1): {rr:.2f}",
            f"├ Lot Size:     {result.lot_size:.6f}",
            f"├ Risk $:       ${result.risk_dollars:.2f}",
            f"└ Max Loss/Day: ${result.max_loss_today:.2f}",
        ]

        if not result.can_trade:
            lines.append(f"\n⛔ <b>BLOCKED:</b> {result.reason}")

        return "\n".join(lines)
