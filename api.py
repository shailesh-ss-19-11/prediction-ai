"""
REST API server for the prediction-ai bot.
Started as a daemon thread from main.py — shares the live PaperTradingEngine instance.

Endpoints
---------
GET /health                          Bot health and uptime
GET /paper-trades                    All open + closed paper trades
GET /paper-trades/open               Open trades only
GET /paper-trades/closed             Closed trades (filterable)
GET /paper-trades/stats              Performance statistics
GET /paper-trades/<id>               Single trade by ID
GET /paper-trades/download           Download paper_trades.json

GET /trade-records                   All trade journal records (filterable)
GET /trade-records/open              Open journal records only
GET /trade-records/closed            Closed journal records only
GET /trade-records/stats             Journal performance stats
GET /trade-records/<id>              Single record by trade ID
GET /trade-records/download          Download trade_records.json

GET /logs                            List available log files
GET /logs/bot                        Bot log (tail=N lines, default 100)
GET /logs/errors                     Error log (tail=N lines, default 100)
GET /logs/download/<filename>        Download a log file
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file

logger = logging.getLogger(__name__)

TRADE_RECORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_records.json")
PAPER_TRADES_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades.json")
LOGS_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

_start_time = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_trade_records() -> list:
    if not os.path.exists(TRADE_RECORDS_FILE):
        return []
    try:
        with open(TRADE_RECORDS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _tail_file(filepath: str, n: int) -> list[str]:
    """Return the last n lines of a file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return [l.rstrip("\n") for l in lines[-n:]]


def _journal_stats(records: list) -> dict:
    closed = [r for r in records if r.get("status") == "closed"]
    total  = len(closed)
    if total == 0:
        return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_rr": 0.0, "winning": 0, "losing": 0, "breakeven": 0}

    winning   = [r for r in closed if (r.get("pnl") or 0) > 0]
    losing    = [r for r in closed if (r.get("pnl") or 0) < 0]
    breakeven = [r for r in closed if (r.get("pnl") or 0) == 0]
    total_pnl = sum(r.get("pnl") or 0 for r in closed)

    rr_values = []
    for r in closed:
        entry = r.get("entry") or 0
        sl    = r.get("stop_loss") or 0
        exit_ = r.get("exit_price") or 0
        if entry and sl and exit_ and abs(entry - sl) > 0:
            risk = abs(entry - sl)
            rr_values.append(abs(exit_ - entry) / risk)

    return {
        "total_trades": total,
        "winning":      len(winning),
        "losing":       len(losing),
        "breakeven":    len(breakeven),
        "win_rate":     round(len(winning) / total * 100, 2),
        "total_pnl":    round(total_pnl, 4),
        "avg_rr":       round(sum(rr_values) / len(rr_values), 3) if rr_values else 0.0,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(paper_engine):
    app = Flask(__name__)
    app.logger.disabled = True

    # ── root ──────────────────────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "service": "DeltaSignalBot API",
            "status":  "running",
            "endpoints": [
                "/health",
                "/paper-trades",
                "/paper-trades/open",
                "/paper-trades/closed",
                "/paper-trades/stats",
                "/paper-trades/download",
                "/trade-records",
                "/trade-records/open",
                "/trade-records/closed",
                "/trade-records/stats",
                "/trade-records/download",
                "/logs",
                "/logs/bot",
                "/logs/errors",
            ],
        })

    # ── health ────────────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        uptime_s = int(time.time() - _start_time)
        h, rem   = divmod(uptime_s, 3600)
        m, s     = divmod(rem, 60)
        open_trades   = paper_engine.get_open_trades()
        closed_trades = paper_engine.get_trade_history()
        return jsonify({
            "status":        "running",
            "uptime":        f"{h}h {m}m {s}s",
            "uptime_seconds": uptime_s,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "balance":       round(paper_engine.balance, 4),
            "open_trades":   len(open_trades),
            "closed_trades": len(closed_trades),
        })

    # ── paper-trades ──────────────────────────────────────────────────────────

    @app.route("/paper-trades", methods=["GET"])
    def get_paper_trades():
        """All open + closed trades. Optional query: ?symbol=BTCUSD&direction=LONG"""
        symbol    = (request.args.get("symbol")    or "").upper() or None
        direction = (request.args.get("direction") or "").upper() or None

        open_trades   = paper_engine.get_open_trades()
        closed_trades = paper_engine.get_trade_history()

        def _filter(trades):
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            if direction:
                trades = [t for t in trades if t.direction == direction]
            return trades

        return jsonify({
            "initial_balance": paper_engine._initial_balance,
            "current_balance": round(paper_engine.balance, 4),
            "open_trades":     [asdict(t) for t in _filter(open_trades)],
            "closed_trades":   [asdict(t) for t in _filter(closed_trades)],
        })

    @app.route("/paper-trades/open", methods=["GET"])
    def get_open_trades():
        """Open trades only."""
        symbol    = (request.args.get("symbol")    or "").upper() or None
        direction = (request.args.get("direction") or "").upper() or None
        trades    = paper_engine.get_open_trades()
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if direction:
            trades = [t for t in trades if t.direction == direction]
        return jsonify([asdict(t) for t in trades])

    @app.route("/paper-trades/closed", methods=["GET"])
    def get_closed_trades():
        """Closed trades. Optional: ?symbol=&direction=&limit=50"""
        symbol    = (request.args.get("symbol")    or "").upper() or None
        direction = (request.args.get("direction") or "").upper() or None
        limit     = int(request.args.get("limit", 0))

        trades = paper_engine.get_trade_history()
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if direction:
            trades = [t for t in trades if t.direction == direction]
        if limit > 0:
            trades = trades[-limit:]
        return jsonify([asdict(t) for t in trades])

    @app.route("/paper-trades/stats", methods=["GET"])
    def get_paper_stats():
        """Performance statistics from the live engine."""
        return jsonify(paper_engine.get_stats())

    @app.route("/paper-trades/<trade_id>", methods=["GET"])
    def get_paper_trade(trade_id):
        """Single trade by ID (checks open first, then closed)."""
        for t in paper_engine.get_open_trades():
            if t.id == trade_id:
                return jsonify(asdict(t))
        for t in paper_engine.get_trade_history():
            if t.id == trade_id:
                return jsonify(asdict(t))
        return jsonify({"error": f"Trade {trade_id} not found"}), 404

    @app.route("/paper-trades/download", methods=["GET"])
    def download_paper_trades():
        paper_engine.save_to_file(PAPER_TRADES_FILE)
        return send_file(
            PAPER_TRADES_FILE,
            mimetype="application/json",
            as_attachment=True,
            download_name="paper_trades.json",
        )

    # ── trade-records ─────────────────────────────────────────────────────────

    @app.route("/trade-records", methods=["GET"])
    def get_trade_records():
        """All journal records. Optional: ?symbol=&direction=&status=open|closed"""
        symbol    = (request.args.get("symbol")    or "").upper() or None
        direction = (request.args.get("direction") or "").upper() or None
        status    = (request.args.get("status")    or "").lower()  or None
        limit     = int(request.args.get("limit", 0))

        records = _load_trade_records()
        if symbol:
            records = [r for r in records if (r.get("symbol") or "").upper() == symbol]
        if direction:
            records = [r for r in records if (r.get("direction") or "").upper() == direction]
        if status:
            records = [r for r in records if (r.get("status") or "").lower() == status]
        if limit > 0:
            records = records[-limit:]
        return jsonify(records)

    @app.route("/trade-records/open", methods=["GET"])
    def get_open_records():
        """Journal records with status=open."""
        records = [r for r in _load_trade_records() if r.get("status") == "open"]
        return jsonify(records)

    @app.route("/trade-records/closed", methods=["GET"])
    def get_closed_records():
        """Journal records with status=closed. Optional: ?limit=50"""
        limit   = int(request.args.get("limit", 0))
        records = [r for r in _load_trade_records() if r.get("status") == "closed"]
        if limit > 0:
            records = records[-limit:]
        return jsonify(records)

    @app.route("/trade-records/stats", methods=["GET"])
    def get_record_stats():
        """Performance stats computed from trade_records.json."""
        records = _load_trade_records()
        stats   = _journal_stats(records)

        # Per-symbol breakdown
        symbols = {(r.get("symbol") or "").upper() for r in records}
        by_symbol = {}
        for sym in sorted(symbols):
            sym_records = [r for r in records if (r.get("symbol") or "").upper() == sym]
            by_symbol[sym] = _journal_stats(sym_records)

        stats["by_symbol"] = by_symbol
        return jsonify(stats)

    @app.route("/trade-records/<trade_id>", methods=["GET"])
    def get_trade_record(trade_id):
        """Single journal record by trade ID."""
        for r in _load_trade_records():
            if r.get("id") == trade_id:
                return jsonify(r)
        return jsonify({"error": f"Record {trade_id} not found"}), 404

    @app.route("/trade-records/download", methods=["GET"])
    def download_trade_records():
        if not os.path.exists(TRADE_RECORDS_FILE):
            return jsonify([]), 200
        return send_file(
            TRADE_RECORDS_FILE,
            mimetype="application/json",
            as_attachment=True,
            download_name="trade_records.json",
        )

    # ── logs ──────────────────────────────────────────────────────────────────

    @app.route("/logs", methods=["GET"])
    def list_logs():
        """List all log files with size and last-modified time."""
        if not os.path.isdir(LOGS_DIR):
            return jsonify([])
        files = []
        for fname in sorted(os.listdir(LOGS_DIR)):
            fpath = os.path.join(LOGS_DIR, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                files.append({
                    "name":          fname,
                    "size_bytes":    stat.st_size,
                    "size_kb":       round(stat.st_size / 1024, 1),
                    "last_modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
        return jsonify(files)

    @app.route("/logs/bot", methods=["GET"])
    def get_bot_log():
        """Last N lines of bot.log. Query: ?tail=100 (default 100, max 2000)"""
        n     = min(int(request.args.get("tail", 100)), 2000)
        lines = _tail_file(os.path.join(LOGS_DIR, "bot.log"), n)
        return jsonify({"file": "bot.log", "lines": len(lines), "content": lines})

    @app.route("/logs/errors", methods=["GET"])
    def get_error_log():
        """Last N lines of errors.log. Query: ?tail=100"""
        n     = min(int(request.args.get("tail", 100)), 2000)
        lines = _tail_file(os.path.join(LOGS_DIR, "errors.log"), n)
        return jsonify({"file": "errors.log", "lines": len(lines), "content": lines})

    @app.route("/logs/download/<filename>", methods=["GET"])
    def download_log(filename):
        """Download any log file by name (e.g. bot.log, errors.log)."""
        safe = os.path.basename(filename)          # prevent path traversal
        fpath = os.path.join(LOGS_DIR, safe)
        if not os.path.exists(fpath):
            return jsonify({"error": f"{safe} not found"}), 404
        return send_file(
            fpath,
            mimetype="text/plain",
            as_attachment=True,
            download_name=safe,
        )

    return app


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------

def start_api_thread(paper_engine):
    """Start the Flask API in a background daemon thread."""
    app  = create_app(paper_engine)
    port = int(os.environ.get("PORT", 5000))

    def _run():
        logger.info("API server starting on http://0.0.0.0:%d", port)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, name="api-server", daemon=False)
    thread.start()
    return thread
