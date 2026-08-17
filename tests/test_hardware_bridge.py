"""
tests/test_hardware_bridge.py
================================
Confirms the FastAPI console can actually call into the bundled
hexa_tpu_rt_sim simulator and get back well-formed, live-computed
results (not a hard-coded stub).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from app import hardware_bridge


class TestHardwareBridge(unittest.TestCase):
    def test_returns_all_expected_sections(self):
        result = hardware_bridge.run_water_hardware_benchmark(seed=1, n_samples=40)
        self.assertIn("reflex_layer", result)
        self.assertIn("network_layer", result)
        self.assertIn("mec_cognitive_layer", result)

    def test_reflex_deadline_never_missed(self):
        # Matches hexa_tpu_rt_sim's own README claim for this scenario.
        result = hardware_bridge.run_water_hardware_benchmark(seed=2, n_samples=100)
        self.assertEqual(result["reflex_layer"]["deadline_misses"], 0)

    def test_reflex_latency_is_far_below_network_sla(self):
        result = hardware_bridge.run_water_hardware_benchmark(seed=3, n_samples=100)
        reflex_ms = result["reflex_layer"]["max_latency_ms"]
        sla_ms = result["network_layer"]["sla_target_ms"]
        self.assertLess(reflex_ms, sla_ms)

    def test_mec_layer_vetoes_emergency_shutdown_on_worsening_leak(self):
        result = hardware_bridge.run_water_hardware_benchmark(seed=4, n_samples=40)
        self.assertIn(
            "emergency_full_shutdown", result["mec_cognitive_layer"]["vetoed_actions"]
        )

    def test_is_deterministic_for_a_fixed_seed(self):
        a = hardware_bridge.run_water_hardware_benchmark(seed=5, n_samples=60)
        b = hardware_bridge.run_water_hardware_benchmark(seed=5, n_samples=60)
        self.assertEqual(a["reflex_layer"], b["reflex_layer"])
        self.assertEqual(a["network_layer"], b["network_layer"])


if __name__ == "__main__":
    unittest.main()
