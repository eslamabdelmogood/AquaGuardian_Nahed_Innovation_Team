import unittest
from aquaguardian.detector import analyze
from aquaguardian.engine import ClosedLoopEngine
from aquaguardian.scenarios import SCENARIOS


class AquaGuardianTests(unittest.TestCase):
    def test_each_scenario_is_detected(self):
        for name, frame in SCENARIOS.items():
            with self.subTest(name=name):
                self.assertGreaterEqual(analyze(frame)[name], 0.35)

    def test_engine_selects_validated_actions(self):
        engine = ClosedLoopEngine()
        expected = {
            "pipeline_leak": {"reduce_pressure", "isolate_zone"},
            "pump_degradation": {"reduce_speed", "switch_to_backup", "shutdown"},
            "water_contamination": {"divert_to_treatment", "isolate_and_flush"},
            "drought_stress": {"precision_irrigation", "fixed_schedule"},
            "wildfire_risk": {"mist_only", "mist_and_drones"},
        }
        for name, frame in SCENARIOS.items():
            with self.subTest(name=name):
                decision = engine.decide(frame, name)
                self.assertTrue(decision.executed)
                self.assertIn(decision.selected_action, expected[name])
                self.assertGreaterEqual(decision.reliability, 0.9)

    def test_do_nothing_fails_for_critical_cases(self):
        engine = ClosedLoopEngine()
        for name, frame in SCENARIOS.items():
            decision = engine.decide(frame, name)
            do_nothing = next(
                item for item in decision.alternatives if item["action"] == "do_nothing"
            )
            self.assertFalse(do_nothing["passed_all_stress_tests"])


if __name__ == "__main__":
    unittest.main()
