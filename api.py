"""
Simple REST API for the prediction-ai bot.
Run: python api.py
"""

import os
from flask import Flask, jsonify, send_file, abort

app = Flask(__name__)

PAPER_TRADES_FILE = os.path.join(os.path.dirname(__file__), "paper_trades.json")


@app.route("/paper-trades", methods=["GET"])
def get_paper_trades():
    """Return paper_trades.json content as JSON."""
    if not os.path.exists(PAPER_TRADES_FILE):
        abort(404, description="paper_trades.json not found")
    return send_file(PAPER_TRADES_FILE, mimetype="application/json")


@app.route("/paper-trades/download", methods=["GET"])
def download_paper_trades():
    """Download paper_trades.json as a file attachment."""
    if not os.path.exists(PAPER_TRADES_FILE):
        abort(404, description="paper_trades.json not found")
    return send_file(
        PAPER_TRADES_FILE,
        mimetype="application/json",
        as_attachment=True,
        download_name="paper_trades.json",
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
