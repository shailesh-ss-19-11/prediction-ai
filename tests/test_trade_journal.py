"""
Tests for trade_journal.py. JOURNAL_FILE is monkeypatched to a temp file so
these tests never touch the project's real trade_records.json.

Run: python -m unittest tests.test_trade_journal -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_journal  # noqa: E402


class TradeJournalTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_file = trade_journal.JOURNAL_FILE
        trade_journal.JOURNAL_FILE = os.path.join(self._tmpdir.name, "trade_records.json")

    def tearDown(self):
        trade_journal.JOURNAL_FILE = self._orig_file
        self._tmpdir.cleanup()

    def test_record_open_then_summary_shows_open_trade(self):
        trade_journal.record_open(
            trade_id="abc123", symbol="BTCUSD", direction="LONG",
            entry=100, stop_loss=95, tp1=110, tp2=120,
            lot_size=1.0, risk_dollars=5.0, confidence=0.8, rr=2.0,
            reasons=["test"],
        )
        summary = trade_journal.get_summary()
        self.assertEqual(summary["total_trades"], 0)   # only counts closed
        self.assertEqual(summary["open_trades"], 1)

    def test_record_close_updates_matching_record(self):
        trade_journal.record_open(
            trade_id="abc123", symbol="BTCUSD", direction="LONG",
            entry=100, stop_loss=95, tp1=110, tp2=120,
            lot_size=1.0, risk_dollars=5.0,
        )
        trade_journal.record_close(trade_id="abc123", exit_price=110, exit_reason="tp1", pnl=10.0)

        records = trade_journal._load()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["status"], "closed")
        self.assertTrue(rec["tp1_hit"])
        self.assertFalse(rec["sl_hit"])
        self.assertEqual(rec["pnl"], 10.0)

    def test_record_close_unmatched_id_appends_standalone(self):
        trade_journal.record_close(trade_id="ghost", exit_price=50, exit_reason="sl", pnl=-5.0)
        records = trade_journal._load()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "ghost")
        self.assertEqual(records[0]["status"], "closed")
        self.assertTrue(records[0]["sl_hit"])

    def test_get_summary_computes_win_rate(self):
        trade_journal.record_open(trade_id="w1", symbol="BTCUSD", direction="LONG",
                                   entry=100, stop_loss=95, tp1=110, tp2=120,
                                   lot_size=1.0, risk_dollars=5.0)
        trade_journal.record_close(trade_id="w1", exit_price=110, exit_reason="tp1", pnl=10.0)

        trade_journal.record_open(trade_id="l1", symbol="ETHUSD", direction="LONG",
                                   entry=2000, stop_loss=1950, tp1=2100, tp2=2200,
                                   lot_size=1.0, risk_dollars=5.0)
        trade_journal.record_close(trade_id="l1", exit_price=1950, exit_reason="sl", pnl=-50.0)

        summary = trade_journal.get_summary()
        self.assertEqual(summary["total_trades"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["win_rate_pct"], 50.0)
        self.assertAlmostEqual(summary["total_pnl"], -40.0, places=4)
        self.assertEqual(summary["tp1_hits"], 1)
        self.assertEqual(summary["sl_hits"], 1)

    def test_load_corrupt_file_returns_empty_list(self):
        with open(trade_journal.JOURNAL_FILE, "w") as fh:
            fh.write("{not valid json")
        self.assertEqual(trade_journal._load(), [])

    def test_load_missing_file_returns_empty_list(self):
        self.assertEqual(trade_journal._load(), [])


if __name__ == "__main__":
    unittest.main()
