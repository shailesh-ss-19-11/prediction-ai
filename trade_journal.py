"""
Persistent trade journal — appends every signal to trade_records.json.

Each record is written when a trade opens and updated in-place when it closes.
The file is always a valid JSON array — safe to read at any time.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)

JOURNAL_FILE = os.path.join(config.DATA_DIR, "trade_records.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> list:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("[Journal] Failed to load %s: %s", JOURNAL_FILE, exc)
        return []


def _save(records: list) -> None:
    try:
        with open(JOURNAL_FILE, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
        logger.debug("[Journal] Saved %d records to %s", len(records), JOURNAL_FILE)
    except OSError as exc:
        logger.error("[Journal] Failed to save %s: %s", JOURNAL_FILE, exc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_open(
    trade_id: str,
    symbol: str,
    direction: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    lot_size: float,
    risk_dollars: float,
    confidence: float = 0.0,
    rr: float = 0.0,
    reasons: Optional[list] = None,
) -> None:
    """
    Append a new trade record when a signal fires.
    Called immediately after a valid setup is confirmed.
    """
    records = _load()
    record = {
        "id":           trade_id,
        "symbol":       symbol,
        "direction":    direction,
        "entry":        entry,
        "stop_loss":    stop_loss,
        "tp1":          tp1,
        "tp2":          tp2,
        "lot_size":     lot_size,
        "risk_dollars": round(risk_dollars, 4),
        "confidence":   round(confidence, 4),
        "rr":           round(rr, 3),
        "reasons":      reasons or [],
        "opened_at":    _now_iso(),
        "status":       "open",
        # Outcome fields — filled in when trade closes
        "closed_at":    None,
        "exit_price":   None,
        "exit_reason":  None,
        "pnl":          None,
        "sl_hit":       False,
        "tp1_hit":      False,
        "tp2_hit":      False,
    }
    records.append(record)
    _save(records)
    logger.info(
        "[Journal] Recorded OPEN — %s %s | id=%s | entry=%.4f | sl=%.4f | tp1=%.4f | tp2=%.4f",
        symbol, direction, trade_id, entry, stop_loss, tp1, tp2,
    )


def record_close(
    trade_id: str,
    exit_price: float,
    exit_reason: str,
    pnl: Optional[float],
) -> None:
    """
    Update an existing trade record when the trade closes (SL or TP hit).
    If the ID is not found (e.g. trade opened before journal existed), a
    standalone closed record is appended instead.
    """
    records = _load()

    for rec in records:
        if rec.get("id") == trade_id:
            rec["status"]      = "closed"
            rec["closed_at"]   = _now_iso()
            rec["exit_price"]  = round(exit_price, 8)
            rec["exit_reason"] = exit_reason
            rec["pnl"]         = round(pnl, 6) if pnl is not None else None
            rec["sl_hit"]      = exit_reason == "sl"
            rec["tp1_hit"]     = exit_reason == "tp1"
            rec["tp2_hit"]     = exit_reason == "tp2"
            _save(records)
            outcome = "SL" if rec["sl_hit"] else ("TP1" if rec["tp1_hit"] else "TP2")
            logger.info(
                "[Journal] Recorded CLOSE — id=%s | %s hit | exit=%.4f | PnL=$%.4f",
                trade_id, outcome, exit_price, pnl,
            )
            return

    # Trade not in journal (e.g. bot restarted) — append a close-only record
    logger.warning(
        "[Journal] id=%s not found — appending standalone close record", trade_id
    )
    records.append({
        "id":          trade_id,
        "status":      "closed",
        "closed_at":   _now_iso(),
        "exit_price":  round(exit_price, 8),
        "exit_reason": exit_reason,
        "pnl":         round(pnl, 6) if pnl is not None else None,
        "sl_hit":      exit_reason == "sl",
        "tp1_hit":     exit_reason == "tp1",
        "tp2_hit":     exit_reason == "tp2",
    })
    _save(records)


def get_summary() -> dict:
    """
    Return a quick performance summary from the journal.
    Useful for diagnostics / status messages.
    """
    records = _load()
    closed = [r for r in records if r.get("status") == "closed"]

    total    = len(closed)
    wins     = [r for r in closed if (r.get("pnl") or 0) > 0]
    losses   = [r for r in closed if (r.get("pnl") or 0) <= 0]
    total_pnl = sum(r.get("pnl") or 0 for r in closed)
    tp1_count = sum(1 for r in closed if r.get("tp1_hit"))
    tp2_count = sum(1 for r in closed if r.get("tp2_hit"))
    sl_count  = sum(1 for r in closed if r.get("sl_hit"))

    return {
        "total_trades":   total,
        "open_trades":    len(records) - total,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   round(len(wins) / total * 100, 1) if total else 0.0,
        "total_pnl":      round(total_pnl, 4),
        "tp1_hits":       tp1_count,
        "tp2_hits":       tp2_count,
        "sl_hits":        sl_count,
    }
