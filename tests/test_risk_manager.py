"""
Tests for risk.risk_manager.RiskManager — pure logic, no I/O.
Run: python -m unittest tests.test_risk_manager -v
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager  # noqa: E402


def _df(highs, lows, closes):
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


class ConstructorTests(unittest.TestCase):
    def test_rejects_nonpositive_balance(self):
        with self.assertRaises(ValueError):
            RiskManager(0)
        with self.assertRaises(ValueError):
            RiskManager(-100)


class PositionSizeTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000, max_risk_pct=1.0)

    def test_normal_sizing(self):
        # risk $ = 1000 * 1% = $10; sl_distance = 100 -> lot_size = 0.1
        size = self.rm.calculate_position_size(entry=50000, stop_loss=49900)
        self.assertAlmostEqual(size, 0.1, places=6)

    def test_zero_sl_distance_returns_zero(self):
        self.assertEqual(self.rm.calculate_position_size(50000, 50000), 0.0)

    def test_nonpositive_entry_returns_zero(self):
        self.assertEqual(self.rm.calculate_position_size(0, -10), 0.0)

    def test_balance_override(self):
        size = self.rm.calculate_position_size(50000, 49900, balance=500)
        self.assertAlmostEqual(size, 0.05, places=6)


class DynamicSLTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000)

    def test_empty_df_returns_zero(self):
        self.assertEqual(self.rm.calculate_dynamic_sl(pd.DataFrame(), "LONG", atr=10), 0.0)

    def test_long_picks_widest_stop(self):
        df = _df(highs=[100] * 10, lows=[95, 96, 97, 98, 99, 100, 101, 102, 103, 80], closes=[100] * 10)
        sl = self.rm.calculate_dynamic_sl(df, "LONG", atr=5, multiplier=1.5, min_sl_pct=0.008)
        # struct_sl = 80 - 0.2*5 = 79; atr_sl = 100 - 7.5 = 92.5; pct_sl = 100*0.992 = 99.2
        # LONG takes min() -> struct_sl = 79
        self.assertAlmostEqual(sl, 79.0, places=1)

    def test_short_picks_widest_stop(self):
        df = _df(highs=[100, 101, 102, 103, 104, 105, 106, 107, 108, 130], lows=[95] * 10, closes=[100] * 10)
        sl = self.rm.calculate_dynamic_sl(df, "SHORT", atr=5, multiplier=1.5, min_sl_pct=0.008)
        # struct_sl = 130 + 1 = 131; SHORT takes max() -> 131
        self.assertAlmostEqual(sl, 131.0, places=1)

    def test_unknown_direction_returns_zero(self):
        df = _df([100], [95], [100])
        self.assertEqual(self.rm.calculate_dynamic_sl(df, "SIDEWAYS", atr=5), 0.0)


class TrailingSLTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000)

    def test_long_no_trail_below_1r(self):
        # entry=100, initial_sl=90 (risk=10); price=105 -> profit=5 < risk=10
        sl = self.rm.calculate_trailing_sl(entry=100, current_price=105, direction="LONG", atr=3, initial_sl=90)
        self.assertEqual(sl, 90)

    def test_long_trails_above_1r(self):
        # profit=15 >= risk=10 -> new_sl = 115 - 3 = 112
        sl = self.rm.calculate_trailing_sl(entry=100, current_price=115, direction="LONG", atr=3, initial_sl=90)
        self.assertEqual(sl, 112)

    def test_long_never_retreats(self):
        # new_sl computed below initial_sl should still return initial_sl (max wins)
        sl = self.rm.calculate_trailing_sl(entry=100, current_price=111, direction="LONG", atr=25, initial_sl=95)
        self.assertEqual(sl, 95)

    def test_short_trails_above_1r(self):
        sl = self.rm.calculate_trailing_sl(entry=100, current_price=85, direction="SHORT", atr=3, initial_sl=110)
        self.assertEqual(sl, 88)

    def test_zero_risk_returns_initial(self):
        sl = self.rm.calculate_trailing_sl(entry=100, current_price=105, direction="LONG", atr=3, initial_sl=100)
        self.assertEqual(sl, 100)


class BreakevenTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000)

    def test_long_adds_buffer(self):
        be = self.rm.calculate_breakeven(entry=100, direction="LONG", risk=10, buffer_pct=0.01)
        self.assertAlmostEqual(be, 101.0, places=4)

    def test_short_subtracts_buffer(self):
        be = self.rm.calculate_breakeven(entry=100, direction="SHORT", risk=10, buffer_pct=0.01)
        self.assertAlmostEqual(be, 99.0, places=4)


class DailyLossGuardTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000, max_daily_loss_pct=3.0)

    def test_within_limit_allows(self):
        self.assertTrue(self.rm.check_daily_loss_limit(daily_pnl=-20))

    def test_breach_blocks(self):
        self.assertFalse(self.rm.check_daily_loss_limit(daily_pnl=-30))

    def test_profit_always_allows(self):
        self.assertTrue(self.rm.check_daily_loss_limit(daily_pnl=50))


class DrawdownGuardTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000, max_drawdown_pct=10.0)

    def test_within_limit_allows(self):
        self.assertTrue(self.rm.check_max_drawdown(peak_balance=1000, current_balance=950))

    def test_breach_blocks(self):
        self.assertFalse(self.rm.check_max_drawdown(peak_balance=1000, current_balance=890))

    def test_nonpositive_peak_allows(self):
        self.assertTrue(self.rm.check_max_drawdown(peak_balance=0, current_balance=100))


class EvaluateTradeTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(account_balance=1000, max_risk_pct=1.0,
                               max_daily_loss_pct=3.0, max_drawdown_pct=10.0)

    def test_happy_path(self):
        result = self.rm.evaluate_trade(entry=100, stop_loss=95, tp1=110)
        self.assertTrue(result.can_trade)
        self.assertGreater(result.lot_size, 0)

    def test_blocked_by_daily_loss(self):
        result = self.rm.evaluate_trade(entry=100, stop_loss=95, tp1=110, daily_pnl=-40)
        self.assertFalse(result.can_trade)
        self.assertIn("Daily loss limit", result.reason)

    def test_blocked_by_drawdown(self):
        result = self.rm.evaluate_trade(
            entry=100, stop_loss=95, tp1=110, peak_balance=1000, current_balance=880
        )
        self.assertFalse(result.can_trade)
        self.assertIn("drawdown", result.reason)

    def test_blocked_by_identical_entry_sl(self):
        result = self.rm.evaluate_trade(entry=100, stop_loss=100, tp1=110)
        self.assertFalse(result.can_trade)
        self.assertIn("identical", result.reason)

    def test_blocked_by_low_rr(self):
        # sl_distance=5, reward=abs(102-100)=2 -> rr=0.4 < 1.0
        result = self.rm.evaluate_trade(entry=100, stop_loss=95, tp1=102)
        self.assertFalse(result.can_trade)
        self.assertIn("R:R", result.reason)


if __name__ == "__main__":
    unittest.main()
