"""
REST API server for the prediction-ai bot.
Started as a daemon thread from main.py — shares the live PaperTradingEngine instance.
"""

import logging
import os
import threading
from dataclasses import asdict

from flask import Flask, jsonify, send_file

logger = logging.getLogger(__name__)


def create_app(paper_engine):
    app = Flask(__name__)
    app.logger.disabled = True  # suppress Flask's own logger; main.py handles logging

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
