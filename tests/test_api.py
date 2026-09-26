"""
Tests for api.py — Flask routes, using the test client (no real server).

Run: python -m unittest tests.test_api -v
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api  # noqa: E402
from execution.paper_trading import PaperTradingEngine  # noqa: E402


class ApiTests(unittest.TestCase):
    def setUp(self):
        # Isolate from any real data/ files on disk (trade_records.json,
        # cooldowns.json, etc.) — these tests must not depend on, or be
        # broken by, whatever the bot has actually recorded locally.
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_paths = {
            "TRADE_RECORDS_FILE": api.TRADE_RECORDS_FILE,
            "PAPER_TRADES_FILE": api.PAPER_TRADES_FILE,
            "LIVE_ORDERS_FILE": api.LIVE_ORDERS_FILE,
            "LIVE_POSITIONS_FILE": api.LIVE_POSITIONS_FILE,
            "COOLDOWNS_FILE": api.COOLDOWNS_FILE,
        }
        api.TRADE_RECORDS_FILE = os.path.join(self._tmpdir.name, "trade_records.json")
        api.PAPER_TRADES_FILE = os.path.join(self._tmpdir.name, "paper_trades.json")
        api.LIVE_ORDERS_FILE = os.path.join(self._tmpdir.name, "live_orders.json")
        api.LIVE_POSITIONS_FILE = os.path.join(self._tmpdir.name, "live_positions.json")
        api.COOLDOWNS_FILE = os.path.join(self._tmpdir.name, "cooldowns.json")

        self.engine = PaperTradingEngine(500)
        self.app = api.create_app(self.engine)
        self.client = self.app.test_client()

    def tearDown(self):
        for name, value in self._orig_paths.items():
            setattr(api, name, value)
        self._tmpdir.cleanup()

    def test_dashboard_serves_html(self):
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"DeltaSignalBot", resp.data)
        self.assertIn(b"paper-trades", resp.data)  # fetch() call present

    def test_index_lists_dashboard_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("/dashboard", resp.get_json()["endpoints"])

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["open_trades"], 0)

    def test_paper_trades_empty(self):
        resp = self.client.get("/paper-trades")
        data = resp.get_json()
        self.assertEqual(data["current_balance"], 500)
        self.assertEqual(data["open_trades"], [])
        self.assertEqual(data["closed_trades"], [])

    def test_paper_trades_reflects_open_and_closed(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.engine.open_trade("ETHUSD", "SHORT", 2000, 2050, 1950, 1900, 1.0)
        self.engine.update({"ETHUSD": 1940})  # closes ETH at tp1

        resp = self.client.get("/paper-trades")
        data = resp.get_json()
        self.assertEqual(len(data["open_trades"]), 1)
        self.assertEqual(data["open_trades"][0]["symbol"], "BTCUSD")
        self.assertEqual(len(data["closed_trades"]), 1)
        self.assertEqual(data["closed_trades"][0]["symbol"], "ETHUSD")
        self.assertGreater(data["closed_trades"][0]["pnl"], 0)

    def test_paper_trades_filters_by_symbol(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.engine.open_trade("ETHUSD", "LONG", 2000, 1950, 2100, 2200, 1.0)

        resp = self.client.get("/paper-trades/open?symbol=BTCUSD")
        data = resp.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["symbol"], "BTCUSD")

    def test_paper_trade_by_id_not_found(self):
        resp = self.client.get("/paper-trades/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_paper_stats_shape_matches_dashboard_expectations(self):
        """The dashboard JS reads stats.total_pnl and stats.win_rate directly."""
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.engine.update({"BTCUSD": 111})

        resp = self.client.get("/paper-trades/stats")
        data = resp.get_json()
        self.assertIn("total_pnl", data)
        self.assertIn("win_rate", data)
        self.assertEqual(data["total_pnl"], 10.0)
        self.assertEqual(data["win_rate"], 100.0)

    def test_trade_records_empty_when_no_file(self):
        resp = self.client.get("/trade-records")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [])

    def test_cooldowns_empty_when_no_file(self):
        resp = self.client.get("/cooldowns")
        self.assertEqual(resp.get_json(), {})

    def test_live_orders_empty_when_no_file(self):
        resp = self.client.get("/live-orders")
        self.assertEqual(resp.get_json(), {})


class AccountBalanceTests(unittest.TestCase):
    def test_no_exchange_attached_returns_503(self):
        app = api.create_app(PaperTradingEngine(100))  # exchange=None (default)
        client = app.test_client()
        resp = client.get("/account/balance")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["balances"], {})

    def test_picks_usdt_as_primary(self):
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {"USDT": 5.88, "BTC": 0.0001}
        app = api.create_app(PaperTradingEngine(100), exchange=exchange)
        client = app.test_client()

        resp = client.get("/account/balance")
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["primary_asset"], "USDT")
        self.assertEqual(data["primary_balance"], 5.88)
        self.assertIsNone(data["error"])

    def test_empty_balances_reports_error_hint(self):
        """Mirrors the real 401/IP-whitelist failure: fetch_balance() returns {}."""
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {}
        app = api.create_app(PaperTradingEngine(100), exchange=exchange)
        client = app.test_client()

        resp = client.get("/account/balance")
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(data["primary_asset"])
        self.assertIsNotNone(data["error"])

    def test_exchange_raises_returns_502(self):
        exchange = MagicMock()
        exchange.fetch_balance.side_effect = ConnectionError("network down")
        app = api.create_app(PaperTradingEngine(100), exchange=exchange)
        client = app.test_client()

        resp = client.get("/account/balance")
        self.assertEqual(resp.status_code, 502)
        self.assertIn("network down", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
