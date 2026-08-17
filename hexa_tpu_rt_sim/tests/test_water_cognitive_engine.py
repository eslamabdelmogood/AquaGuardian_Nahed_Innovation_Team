"""
tests/test_water_cognitive_engine.py
=======================================
Tests for mec/domains/water_cognitive_engine.py, including the
specific reuse claims the module's docstring makes: BatForecaster and
SquidController are the SAME classes as the aviation/structural
domain's, not reimplementations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from mec.bhs_cognitive_engine import BatForecaster, SquidController
from mec.domains.water_cognitive_engine import (
    WaterBHSCognitiveEngine, WaterCandidateAction, WaterHammerHermitCrabEvaluator,
    DEFAULT_WATER_CANDIDATE_ACTIONS,
)


class TestWaterHammerVeto(unittest.TestCase):
    def test_small_flow_change_is_not_vetoed(self):
        hc = WaterHammerHermitCrabEvaluator()
        action = WaterCandidateAction("gentle", flow_rate_change_lps=-5.0,
                                       safety_benefit=0.5, productivity_cost=0.1, power_cost_w=5.0)
        assessment = hc.assess(leak_severity=0.3, candidates=[action])
        self.assertIn("gentle", [a.name for a in assessment.allowed_actions])
        self.assertNotIn("gentle", assessment.vetoed_actions)

    def test_abrupt_flow_change_is_vetoed(self):
        hc = WaterHammerHermitCrabEvaluator()
        action = WaterCandidateAction("slam_shut", flow_rate_change_lps=-100.0,
                                       safety_benefit=1.0, productivity_cost=1.0, power_cost_w=5.0)
        assessment = hc.assess(leak_severity=0.3, candidates=[action])
        self.assertIn("slam_shut", assessment.vetoed_actions)

    def test_veto_is_independent_of_severity(self):
        """The hard constraint should not loosen just because things
        are urgent -- surge pressure risk is a structural limit, not
        a value judgment Squid gets to override."""
        hc = WaterHammerHermitCrabEvaluator()
        action = WaterCandidateAction("slam_shut", flow_rate_change_lps=-100.0,
                                       safety_benefit=1.0, productivity_cost=1.0, power_cost_w=5.0)
        calm = hc.assess(leak_severity=0.0, candidates=[action])
        severe = hc.assess(leak_severity=1.0, candidates=[action])
        self.assertIn("slam_shut", calm.vetoed_actions)
        self.assertIn("slam_shut", severe.vetoed_actions)


class TestWaterBHSCognitiveEngine(unittest.TestCase):
    def test_bat_forecaster_is_the_literal_same_class_as_aviation_domain(self):
        engine = WaterBHSCognitiveEngine()
        self.assertIs(type(engine.bat), BatForecaster)

    def test_squid_controller_is_the_literal_same_class_as_aviation_domain(self):
        engine = WaterBHSCognitiveEngine()
        self.assertIs(type(engine.squid), SquidController)

    def test_emergency_shutdown_never_chosen_despite_highest_raw_safety_benefit(self):
        engine = WaterBHSCognitiveEngine()
        result = engine.evaluate(pressure_history_kpa=[600, 550, 500, 460, 420], leak_severity=0.9)
        self.assertNotEqual(result.control.chosen_action, "emergency_full_shutdown")
        self.assertIn("emergency_full_shutdown", result.control.vetoed_actions)

    def test_declining_pressure_yields_finite_rul_via_deficit_transform(self):
        engine = WaterBHSCognitiveEngine(baseline_pressure_kpa=600.0)
        declining = [600 - i * 3 for i in range(15)]
        result = engine.evaluate(pressure_history_kpa=declining, leak_severity=0.5)
        self.assertIsNotNone(result.rul.remaining_useful_life_windows)
        self.assertGreater(result.rul.trend_slope_kpa_per_window, 0)   # deficit rising

    def test_stable_pressure_yields_no_projected_rul(self):
        engine = WaterBHSCognitiveEngine()
        flat = [600.0] * 10
        result = engine.evaluate(pressure_history_kpa=flat, leak_severity=0.0)
        self.assertIsNone(result.rul.remaining_useful_life_windows)

    def test_control_shifts_toward_stronger_action_as_severity_rises(self):
        engine = WaterBHSCognitiveEngine()
        calm = engine.evaluate(pressure_history_kpa=[600] * 10, leak_severity=0.0)
        severe = engine.evaluate(pressure_history_kpa=[600 - i * 5 for i in range(10)],
                                  leak_severity=0.9)
        self.assertNotEqual(calm.control.chosen_action, severe.control.chosen_action)


if __name__ == "__main__":
    unittest.main()
