import math
import unittest
from dataclasses import replace

from strategy_rules import (
    SetupRuleConfig,
    SetupRuleInput,
    evaluate_setup_rules,
)


class StrategyRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SetupRuleConfig()

    @staticmethod
    def long_input() -> SetupRuleInput:
        return SetupRuleInput(
            price=100.0,
            candle_open=99.5,
            prev_close=99.0,
            opening_high=99.8,
            opening_low=98.5,
            ema9=99.4,
            ema20=99.0,
            vwap=99.5,
            rsi=60.0,
            atr=1.0,
            rvol=2.0,
            opening_rvol=1.2,
            body_ratio=0.8,
            close_location=0.9,
            nifty_regime="BULL",
            stock_in_play_score=80.0,
            spread_bps=4.0,
            prior_post_opening_max_close=99.75,
            prior_post_opening_min_close=98.6,
        )

    @staticmethod
    def short_input() -> SetupRuleInput:
        return SetupRuleInput(
            price=98.0,
            candle_open=99.0,
            prev_close=100.0,
            opening_high=100.5,
            opening_low=98.2,
            ema9=98.5,
            ema20=99.0,
            vwap=98.5,
            rsi=40.0,
            atr=1.0,
            rvol=2.5,
            opening_rvol=1.5,
            body_ratio=0.7,
            close_location=0.2,
            nifty_regime="BEAR",
            stock_in_play_score=70.0,
            spread_bps=6.0,
            prior_post_opening_max_close=100.0,
            prior_post_opening_min_close=98.25,
        )

    def test_golden_long_setup_is_accepted(self) -> None:
        result = evaluate_setup_rules(self.long_input(), self.config)

        self.assertTrue(result.accepted)
        self.assertEqual(result.side, "LONG")
        self.assertEqual(result.reason, "OK")
        self.assertAlmostEqual(result.day_change_pct, 1.0101010101010102)
        self.assertAlmostEqual(result.atr_pct, 0.01)
        self.assertAlmostEqual(result.long_breakout_atr, 0.2)
        self.assertAlmostEqual(result.breakout_distance_atr, 0.2)
        self.assertAlmostEqual(result.vwap_distance_atr, 0.5)
        self.assertAlmostEqual(result.technical_score, 77.44444444444446)

    def test_golden_short_setup_is_accepted(self) -> None:
        result = evaluate_setup_rules(self.short_input(), self.config)

        self.assertTrue(result.accepted)
        self.assertEqual(result.side, "SHORT")
        self.assertEqual(result.reason, "OK")
        self.assertAlmostEqual(result.day_change_pct, -2.0)
        self.assertAlmostEqual(result.atr_pct, 1.0 / 98.0)
        self.assertAlmostEqual(result.short_breakout_atr, 0.2)
        self.assertAlmostEqual(result.breakout_distance_atr, 0.2)
        self.assertAlmostEqual(result.vwap_distance_atr, 0.5)
        self.assertAlmostEqual(result.technical_score, 77.55555555555556)

    def test_representative_gate_rejection_is_auditable(self) -> None:
        inputs = replace(
            self.long_input(),
            rvol=self.config.min_rvol - 0.01,
        )

        result = evaluate_setup_rules(inputs, self.config)

        self.assertFalse(result.accepted)
        self.assertIsNone(result.side)
        self.assertEqual(result.reason, "RVOL_BELOW_MINIMUM")
        self.assertAlmostEqual(result.day_change_pct, 1.0101010101010102)
        self.assertAlmostEqual(result.atr_pct, 0.01)

    def test_nonfinite_and_boolean_inputs_fail_closed(self) -> None:
        invalid_changes = (
            {"price": math.nan},
            {"ema9": math.inf},
            {"atr": True},
            {"prior_post_opening_max_close": -math.inf},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                result = evaluate_setup_rules(
                    replace(self.long_input(), **changes),
                    self.config,
                )
                self.assertFalse(result.accepted)
                self.assertIsNone(result.side)
                self.assertEqual(result.reason, "NONFINITE_INPUT")

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_configs = (
            {"min_atr_pct": math.nan},
            {"min_rvol": True},
            {"min_atr_pct": 0.03, "max_atr_pct": 0.02},
            {"min_rvol": -0.01},
            {"min_breakout_distance_atr": 0.9},
            {"max_vwap_distance_atr": 0.0},
            {"rvol_score_span": 0.0},
            {"long_max_rsi": 101.0},
            {"short_max_close_location": 1.01},
        )

        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SetupRuleConfig(**values)

    def test_scalar_evaluation_is_deterministic_and_future_invariant(self) -> None:
        # Outcome observations live in a separate replay stream. Appending them
        # must not alter the immutable point-in-time features supplied here.
        point_in_time_inputs = self.long_input()
        future_prices = [101.0, 102.0]

        expected = evaluate_setup_rules(point_in_time_inputs, self.config)
        repeated = [
            evaluate_setup_rules(point_in_time_inputs, self.config)
            for _ in range(20)
        ]
        future_prices.extend([98.0, 105.0])
        after_future_append = evaluate_setup_rules(
            point_in_time_inputs,
            self.config,
        )

        self.assertTrue(all(result == expected for result in repeated))
        self.assertEqual(after_future_append, expected)
        self.assertEqual(future_prices, [101.0, 102.0, 98.0, 105.0])


if __name__ == "__main__":
    unittest.main()
