"""
REST API server for the prediction-ai bot.
Started as a daemon thread from main.py — shares the live PaperTradingEngine instance.
"""

import json
import logging
import os
import threading
from dataclasses import asdict

from flask import Flask, jsonify, send_file

logger = logging.getLogger(__name__)

TRADE_RECORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_records.json")


def create_app(paper_engine):
    app = Flask(__name__)
    app.logger.disabled = True  # suppress Flask's own logger; main.py handles logging

    # ── paper-trades ──────────────────────────────────────────────────────────

    @app.route("/paper-trades", methods=["GET"])
    def get_paper_trades():
        """Return live paper trade state as JSON."""
        return jsonify({
            "initial_balance": paper_engine._initial_balance,
            "current_balance": paper_engine.balance,
            "open_trades":     [asdict(t) for t in paper_engine.get_open_trades()],
            "closed_trades":   [asdict(t) for t in paper_engine.get_trade_history()],
        })

    @app.route("/paper-trades/download", methods=["GET"])
    def download_paper_trades():
        """Save current state to disk and return it as a file download."""
        filepath = "paper_trades.json"
        paper_engine.save_to_file(filepath)
        return send_file(
            os.path.abspath(filepath),
            mimetype="application/json",
            as_attachment=True,
            download_name="paper_trades.json",
        )

    # ── trade-records ─────────────────────────────────────────────────────────

    @app.route("/trade-records", methods=["GET"])
    def get_trade_records():
        """Return trade_records.json content as JSON."""
        if not os.path.exists(TRADE_RECORDS_FILE):
            return jsonify([])
        with open(TRADE_RECORDS_FILE, "r", encoding="utf-8") as fh:
            return jsonify(json.load(fh))

    @app.route("/trade-records/download", methods=["GET"])
    def download_trade_records():
        """Download trade_records.json as a file attachment."""
        if not os.path.exists(TRADE_RECORDS_FILE):
            return jsonify([]), 200
        return send_file(
            TRADE_RECORDS_FILE,
            mimetype="application/json",
            as_attachment=True,
            download_name="trade_records.json",
        )

    return app


def start_api_thread(paper_engine):
    """Start the Flask API in a background daemon thread."""
    app = create_app(paper_engine)
    port = int(os.environ.get("PORT", 5000))

    def _run():
        logger.info("API server starting on port %d", port)
        # use_reloader=False required when running inside a thread
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, name="api-server", daemon=True)
    thread.start()
    return thread
