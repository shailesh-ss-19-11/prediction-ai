"""
Tests for execution.paper_trading.PaperTradingEngine.

Includes a regression test for the "trades never close" bug: a fresh engine
that doesn't load saved state loses track of open trades; save/load round
trip must restore them so main.py's startup load actually works.

Run: python -m unittest tests.test_paper_trading -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.paper_trading import PaperTradingEngine  # noqa: E402


class ConstructorTests(unittest.TestCase):
    def test_rejects_nonpositive_balance(self):
        with self.assertRaises(ValueError):
            PaperTradingEngine(0)
        with self.assertRaises(ValueError):
            PaperTradingEngine(-50)


class OpenTradeTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaperTradingEngine(1000)

    def test_open_trade_is_tracked(self):
        trade = self.engine.open_trade("btcusd", "long", 100, 95, 110, 120, 1.0)
        self.assertEqual(trade.status, "open")
        self.assertEqual(trade.symbol, "BTCUSD")   # uppercased
        self.assertEqual(trade.direction, "LONG")  # uppercased
        self.assertIn(trade.id, [t.id for t in self.engine.get_open_trades()])

    def test_ids_are_unique(self):
        t1 = self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        t2 = self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.assertNotEqual(t1.id, t2.id)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaperTradingEngine(1000)

    def test_long_sl_hit(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.update({"BTCUSD": 94})
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].exit_reason, "sl")
        self.assertAlmostEqual(closed[0].pnl, (95 - 100) * 1.0, places=6)

    def test_long_tp1_hit(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.update({"BTCUSD": 111})
        self.assertEqual(closed[0].exit_reason, "tp1")

    def test_long_tp2_takes_priority_over_tp1(self):
        # price gaps straight past both targets in one update
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.update({"BTCUSD": 130})
        self.assertEqual(closed[0].exit_reason, "tp2")
        self.assertEqual(closed[0].exit_price, 120)  # capped at tp2, not fill price

    def test_short_sl_hit(self):
        self.engine.open_trade("ETHUSD", "SHORT", 2000, 2050, 1950, 1900, 2.0)
        closed = self.engine.update({"ETHUSD": 2060})
        self.assertEqual(closed[0].exit_reason, "sl")
        self.assertAlmostEqual(closed[0].pnl, (2000 - 2050) * 2.0, places=6)

    def test_short_tp_hit_profit_positive(self):
        self.engine.open_trade("ETHUSD", "SHORT", 2000, 2050, 1950, 1900, 2.0)
        closed = self.engine.update({"ETHUSD": 1940})
        self.assertEqual(closed[0].exit_reason, "tp1")
        self.assertGreater(closed[0].pnl, 0)

    def test_no_price_for_symbol_leaves_trade_open(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.update({"ETHUSD": 2000})
        self.assertEqual(closed, [])
        self.assertEqual(len(self.engine.get_open_trades()), 1)

    def test_price_between_sl_and_tp1_stays_open(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.update({"BTCUSD": 102})
        self.assertEqual(closed, [])
        self.assertEqual(len(self.engine.get_open_trades()), 1)

    def test_balance_updates_on_close(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.engine.update({"BTCUSD": 111})
        self.assertAlmostEqual(self.engine.balance, 1000 + (110 - 100) * 1.0, places=6)


class CloseTradeTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaperTradingEngine(1000)

    def test_manual_close(self):
        trade = self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        closed = self.engine.close_trade(trade.id, 105, "manual")
        self.assertEqual(closed.status, "closed")
        self.assertEqual(closed.exit_reason, "manual")
        self.assertNotIn(trade.id, [t.id for t in self.engine.get_open_trades()])

    def test_unknown_id_raises(self):
        with self.assertRaises(KeyError):
            self.engine.close_trade("nonexistent", 100)


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaperTradingEngine(1000)

    def test_empty_stats(self):
        stats = self.engine.get_stats()
        self.assertEqual(stats["total_trades"], 0)
        self.assertEqual(stats["win_rate"], 0.0)
        self.assertIsNone(stats["best_trade"])

    def test_stats_after_mixed_trades(self):
        self.engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        self.engine.update({"BTCUSD": 111})  # win: +10
        self.engine.open_trade("ETHUSD", "LONG", 2000, 1950, 2100, 2200, 1.0)
        self.engine.update({"ETHUSD": 1949})  # loss: -50

        stats = self.engine.get_stats()
        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["winning_trades"], 1)
        self.assertEqual(stats["losing_trades"], 1)
        self.assertEqual(stats["win_rate"], 50.0)
        self.assertAlmostEqual(stats["total_pnl"], -40.0, places=4)
        self.assertEqual(stats["best_trade"]["symbol"], "BTCUSD")
        self.assertEqual(stats["worst_trade"]["symbol"], "ETHUSD")


class PersistenceTests(unittest.TestCase):
    """
    Regression coverage for the startup bug: a process restart that creates
    a fresh PaperTradingEngine without calling load_from_file() orphans any
    trades that were open at the last save — they're never checked against
    price again. save/load must round-trip both open and closed trades.
    """

    def test_save_load_round_trip_preserves_open_and_closed(self):
        engine = PaperTradingEngine(500)
        open_trade = engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)
        engine.open_trade("ETHUSD", "SHORT", 2000, 2050, 1950, 1900, 1.0)
        engine.update({"ETHUSD": 1940})  # closes the ETH trade (tp1)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "paper_trades.json")
            engine.save_to_file(path)

            restored = PaperTradingEngine(1)  # different initial balance —
            restored.load_from_file(path)     # load must override it

        self.assertEqual(restored.balance, engine.balance)
        restored_open_ids = {t.id for t in restored.get_open_trades()}
        self.assertEqual(restored_open_ids, {open_trade.id})
        self.assertEqual(len(restored.get_trade_history()), 1)
        self.assertEqual(restored.get_trade_history()[0].symbol, "ETHUSD")

    def test_restored_open_trade_still_closes_on_price_update(self):
        """The actual bug scenario: after a simulated restart, does the
        reloaded trade still get checked and closed on the next price update?"""
        engine = PaperTradingEngine(500)
        engine.open_trade("BTCUSD", "LONG", 100, 95, 110, 120, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "paper_trades.json")
            engine.save_to_file(path)

            # Simulate a fresh process: brand-new engine instance
            restarted = PaperTradingEngine(500)
            restarted.load_from_file(path)

        closed = restarted.update({"BTCUSD": 94})  # breaches SL
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].exit_reason, "sl")

    def test_load_missing_file_is_a_noop(self):
        engine = PaperTradingEngine(100)
        engine.load_from_file("/nonexistent/path/paper_trades.json")
        self.assertEqual(engine.balance, 100)
        self.assertEqual(engine.get_open_trades(), [])

    def test_load_corrupt_file_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "paper_trades.json")
            with open(path, "w") as fh:
                fh.write("{not valid json")

            engine = PaperTradingEngine(100)
            engine.load_from_file(path)  # must not raise
            self.assertEqual(engine.balance, 100)


if __name__ == "__main__":
    unittest.main()
