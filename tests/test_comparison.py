import unittest

from aquaguardian.comparison import aggregate_comparisons, compare_scenario
from aquaguardian.scenarios import SCENARIOS


class StrategyComparisonTests(unittest.TestCase):
    def test_comparison_contains_three_strategies(self):
        result = compare_scenario("pipeline_leak", SCENARIOS["pipeline_leak"])
        names = {item["strategy"] for item in result["strategies"]}
        self.assertEqual(
            names,
            {"reactive_baseline", "detection_only_ai", "aquaguardian_closed_loop"},
        )

    def test_closed_loop_improves_response_time(self):
        result = compare_scenario("pipeline_leak", SCENARIOS["pipeline_leak"])
        self.assertGreater(
            result["improvements"]["response_time_reduction_pct_vs_reactive"],
            99.0,
        )

    def test_aggregate_summary_is_generated(self):
        comparisons = [compare_scenario(name, frame) for name, frame in SCENARIOS.items()]
        summary = aggregate_comparisons(comparisons)
        self.assertEqual(summary["scenario_count"], 5)
        self.assertGreaterEqual(
            summary["strategies"]["aquaguardian_closed_loop"]["average_stress_pass_rate"],
            summary["strategies"]["reactive_baseline"]["average_stress_pass_rate"],
        )


if __name__ == "__main__":
    unittest.main()
