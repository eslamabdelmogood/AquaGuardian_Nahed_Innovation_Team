from pathlib import Path
import unittest

from app.main import all_comparisons, engine_decision, health, list_scenarios


class ApiIntegrationTests(unittest.TestCase):
    def test_health(self):
        self.assertEqual(health()["status"], "ok")

    def test_console_has_scenarios(self):
        self.assertGreaterEqual(len(list_scenarios()), 4)

    def test_engine_decision(self):
        result = engine_decision("pipeline_leak")
        self.assertEqual(result["scenario"], "pipeline_leak")
        self.assertIn("selected_action", result)

    def test_comparison_aggregate(self):
        payload = all_comparisons()
        self.assertEqual(payload["aggregate"]["scenario_count"], 5)
        self.assertIn("aquaguardian_closed_loop", payload["aggregate"]["strategies"])


if __name__ == "__main__":
    unittest.main()


def test_console_contains_same_screen_comparison():
    html = (Path(__file__).parents[1] / "app" / "static" / "index.html").read_text()
    assert "Traditional response vs AquaGuardian" in html
    assert "engine_evidence" in html
    assert "renderComparison" in html


def test_console_scenarios_have_unique_water_engine_mappings():
    from app.main import ENGINE_MAP
    mapped = [ENGINE_MAP[k] for k in ("leak", "pump", "drought", "fire")]
    assert len(mapped) == len(set(mapped))


def test_evidence_metrics_are_backend_derived():
    from app.main import scenario_evidence
    body = scenario_evidence("leak")
    assert body["comparison"]["scenario"] == "pipeline_leak"
    assert body["metrics"]
    labels = {m["label"] for m in body["metrics"]}
    assert "Modeled water-loss reduction" in labels
