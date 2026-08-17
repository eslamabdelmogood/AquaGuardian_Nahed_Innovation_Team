"""
tests/test_middleware_mec.py
=============================
Tests for the Standardization & Middleware Layer (middleware/onem2m.py),
the MEC Compute Layer (mec/*.py), the 5G URLLC network layer
(network/urllc_link.py), and their end-to-end wiring
(integration/edge_to_mec_pipeline.py).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from config import Config
from models.sensor_events import generate_scenario

from middleware.onem2m import (
    MN_CSE, ADN_AE, Operation, ResourceType, ResponseStatusCode, RequestPrimitive,
)
from middleware.onem2m_http import HttpCSEClient
from network.urllc_link import URLLCLink
from mec.bhs_cognitive_engine import (
    BatForecaster, HermitCrabEvaluator, SquidController, BHSCognitiveEngine,
    DEFAULT_CANDIDATE_ACTIONS,
)
from mec.mec_platform import MecPlatform, BHSCognitiveMecApp, MecAppState
from integration.edge_to_mec_pipeline import EdgeToMecPipeline


class TestOneM2MMiddleware(unittest.TestCase):
    def test_ae_registration_creates_resource_and_ae_id(self):
        cse = MN_CSE()
        ae = ADN_AE("edge-01", cse)
        resp = ae.register()
        self.assertEqual(resp.response_status_code, ResponseStatusCode.CREATED)
        self.assertTrue(ae.ae_id.startswith("C"))
        # Both data containers should have been auto-created under the AE.
        retrieved = cse.handle_primitive(RequestPrimitive(
            operation=Operation.RETRIEVE, to="edge-01/sensorData", fr=ae.ae_id))
        self.assertTrue(retrieved.ok)

    def test_duplicate_registration_rejected(self):
        cse = MN_CSE()
        ADN_AE("edge-01", cse).register()
        second = ADN_AE("edge-01", cse)
        resp = second.register()
        self.assertEqual(resp.response_status_code, ResponseStatusCode.ALREADY_EXISTS)

    def test_push_sensor_window_creates_content_instance(self):
        cse = MN_CSE()
        ae = ADN_AE("edge-01", cse)
        ae.register()
        windows = generate_scenario("burst_anomaly", 5, seed=1)
        resp = ae.push_sensor_window(windows[0])
        self.assertEqual(resp.response_status_code, ResponseStatusCode.CREATED)
        self.assertEqual(resp.content["con"]["kind"], "sensorWindow")

        latest = cse.latest_content_instances("edge-01", "sensorData", limit=1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["con"]["index"], windows[0].index)

    def test_push_before_registration_raises(self):
        cse = MN_CSE()
        ae = ADN_AE("edge-01", cse)
        windows = generate_scenario("normal", 1, seed=0)
        with self.assertRaises(Exception):
            ae.push_sensor_window(windows[0])

    def test_reflex_event_lands_in_reflex_container(self):
        cse = MN_CSE()
        ae = ADN_AE("edge-01", cse)
        ae.register()
        resp = ae.push_reflex_event(window_index=3, channels_processed=600, triggered=True)
        self.assertTrue(resp.ok)
        latest = cse.latest_content_instances("edge-01", "reflexEvents", limit=1)
        self.assertEqual(latest[0]["con"]["kind"], "reflexEvent")
        self.assertTrue(latest[0]["con"]["triggered"])


class TestURLLCLink(unittest.TestCase):
    def test_latency_distribution_is_reasonable_and_deterministic(self):
        link_a = URLLCLink(seed=42)
        link_b = URLLCLink(seed=42)
        results_a = [link_a.transmit(i).latency_ms for i in range(50)]
        results_b = [link_b.transmit(i).latency_ms for i in range(50)]
        self.assertEqual(results_a, results_b)      # same seed -> reproducible
        self.assertTrue(all(l > 0 for l in results_a))

    def test_sla_target_enforced_in_stats(self):
        link = URLLCLink(mean_latency_ms=1.0, jitter_sigma_ms=0.1, sla_target_ms=10.0, seed=7)
        for i in range(200):
            link.transmit(i)
        stats = link.stats()
        self.assertEqual(stats["count"], 200)
        self.assertGreater(stats["sla_compliance_rate"], 0.9)

    def test_retransmission_adds_latency(self):
        # Force every transmission to hit a simulated radio-link failure.
        link = URLLCLink(radio_link_failure_prob=1.0, retransmission_penalty_ms=5.0, seed=1)
        result = link.transmit("payload")
        self.assertTrue(result.retransmitted)
        self.assertTrue(result.delivered)


class TestBatForecaster(unittest.TestCase):
    def test_flat_history_yields_no_projected_rul(self):
        bat = BatForecaster(failure_threshold_kpa=24000.0)
        est = bat.forecast([1000.0] * 10)
        self.assertIsNone(est.remaining_useful_life_windows)

    def test_rising_trend_yields_finite_rul(self):
        bat = BatForecaster(failure_threshold_kpa=1000.0)
        history = [100.0 * i for i in range(10)]  # clearly rising
        est = bat.forecast(history)
        self.assertGreater(est.trend_slope_kpa_per_window, 0)
        self.assertIsNotNone(est.remaining_useful_life_windows)
        self.assertGreaterEqual(est.r_squared, 0.99)   # perfectly linear data

    def test_already_past_threshold_yields_zero_rul(self):
        bat = BatForecaster(failure_threshold_kpa=500.0)
        est = bat.forecast([100.0, 200.0, 300.0, 600.0])
        self.assertEqual(est.remaining_useful_life_windows, 0.0)


class TestHermitCrabEvaluator(unittest.TestCase):
    def test_high_severity_lowers_stability_score(self):
        hc = HermitCrabEvaluator()
        calm = hc.assess(mean_temp_c=25.0, anomaly_severity=0.0, candidates=[])
        severe = hc.assess(mean_temp_c=25.0, anomaly_severity=1.0, candidates=[])
        self.assertGreater(calm.stability_score, severe.stability_score)

    def test_veto_blocks_actions_near_natural_frequency(self):
        hc = HermitCrabEvaluator(resonance_guard_band_hz=2.0)
        # Force a known natural frequency, then place one candidate
        # exactly inside the guard band and one clearly outside it.
        from mec.bhs_cognitive_engine import CandidateAction
        assessment = hc.assess(mean_temp_c=25.0, anomaly_severity=0.0, candidates=[
            CandidateAction("inside_band", actuation_freq_hz=40.5,
                             safety_benefit=0.9, productivity_cost=0.1, power_cost_w=5.0),
            CandidateAction("outside_band", actuation_freq_hz=5.0,
                             safety_benefit=0.9, productivity_cost=0.1, power_cost_w=5.0),
        ])
        self.assertIn("inside_band", assessment.vetoed_actions)
        allowed_names = [c.name for c in assessment.allowed_actions]
        self.assertIn("outside_band", allowed_names)
        self.assertNotIn("inside_band", allowed_names)


class TestSquidController(unittest.TestCase):
    def test_never_selects_vetoed_action(self):
        sq = SquidController()
        hc = HermitCrabEvaluator(resonance_guard_band_hz=100.0)  # veto everything
        assessment = hc.assess(25.0, 0.5, DEFAULT_CANDIDATE_ACTIONS)
        self.assertEqual(len(assessment.allowed_actions), 0)
        decision = sq.decide(DEFAULT_CANDIDATE_ACTIONS, assessment, anomaly_severity=0.5)
        self.assertIsNone(decision.chosen_action)

    def test_weights_shift_toward_safety_with_severity(self):
        sq = SquidController()
        calm = sq.reweight(0.0)
        severe = sq.reweight(1.0)
        self.assertGreater(severe["safety"], calm["safety"])
        self.assertLess(severe["productivity"], calm["productivity"])


class TestBHSCognitiveEngineAndMecPlatform(unittest.TestCase):
    def test_engine_end_to_end_evaluate(self):
        engine = BHSCognitiveEngine()
        result = engine.evaluate(strain_history_kpa=[1000, 1200, 1500, 1900, 2400],
                                  mean_temp_c=30.0, anomaly_severity=0.4)
        self.assertIsNotNone(result.rul)
        self.assertIsNotNone(result.stability)
        self.assertIsNotNone(result.control)

    def test_mec_app_lifecycle_and_service_registry(self):
        platform = MecPlatform()
        app = BHSCognitiveMecApp(platform).deploy()
        self.assertEqual(app.state, MecAppState.ACTIVE)
        self.assertIsNotNone(platform.discover(BHSCognitiveMecApp.SERVICE_BAT_FORECASTER))
        self.assertIsNotNone(platform.discover(BHSCognitiveMecApp.SERVICE_HERMIT_CRAB_EVALUATOR))
        self.assertIsNotNone(platform.discover(BHSCognitiveMecApp.SERVICE_SQUID_CONTROLLER))
        app.terminate()
        self.assertIsNone(platform.discover(BHSCognitiveMecApp.SERVICE_SQUID_CONTROLLER))

    def test_process_before_deploy_raises(self):
        platform = MecPlatform()
        app = BHSCognitiveMecApp(platform)
        with self.assertRaises(RuntimeError):
            app.process(0, [1000.0], 25.0, 0.0)


class TestEdgeToMecPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_runs_and_reports_all_layers(self):
        cfg = Config(NUM_WORKERS=4)
        pipeline = EdgeToMecPipeline(cfg=cfg, seed=0)
        report = pipeline.run(scenario="burst_anomaly", num_windows=20, seed=0, num_workers=4)

        self.assertEqual(len(report.telemetry), 20)
        # oneM2M: AE registration (1) + 2 containers + 20 sensor + 20 reflex
        self.assertEqual(report.onem2m_request_log_len, 1 + 2 + 20 + 20)
        self.assertEqual(report.urllc_stats["count"], 20)
        # The TPU-side report must still be produced by the *existing*,
        # unmodified simulator -- confirming this integration is additive.
        self.assertGreater(report.tpu_report.tasks_completed, 0)

    def test_reflex_and_sensor_content_instances_reach_the_cse(self):
        pipeline = EdgeToMecPipeline(cfg=Config(NUM_WORKERS=2), seed=1)
        pipeline.run(scenario="normal", num_windows=5, seed=1, num_workers=2)
        sensor_cis = pipeline.cse.latest_content_instances("hexa-tpu-edge-01", "sensorData", limit=5)
        reflex_cis = pipeline.cse.latest_content_instances("hexa-tpu-edge-01", "reflexEvents", limit=5)
        self.assertEqual(len(sensor_cis), 5)
        self.assertEqual(len(reflex_cis), 5)


CSE_HTTP_URL = "http://127.0.0.1:8080"


def _real_cse_reachable() -> bool:
    try:
        return HttpCSEClient(base_url=CSE_HTTP_URL).ping()
    except Exception:
        return False


@unittest.skipUnless(_real_cse_reachable(),
                      f"no live oneM2M CSE reachable at {CSE_HTTP_URL} -- start one with "
                      f"'pip install acmecse && python -m acmecse --headless --no-coap "
                      f"--no-mqtt --no-ws --no-remote-cse --http-port 8080' to run this class")
class TestRealCSEHttpTransport(unittest.TestCase):
    """Exercises middleware/onem2m_http.py against an actual running
    oneM2M CSE over real HTTP -- the interoperability surface the
    ESTIMED hackathon evaluates, as opposed to the in-memory
    simulation in TestOneM2MMiddleware above. Skipped automatically
    when no CSE is reachable (e.g. in CI without acmecse installed)."""

    def setUp(self):
        self.cse = HttpCSEClient(base_url=CSE_HTTP_URL)

    def test_ae_registers_and_gets_a_real_ae_id(self):
        import time
        ae = ADN_AE(f"test-edge-{int(time.time()*1000)}", self.cse)
        resp = ae.register()
        self.assertEqual(resp.response_status_code, ResponseStatusCode.CREATED)
        self.assertTrue(ae.ae_id)

    def test_push_sensor_window_round_trips_over_http(self):
        import time
        ae = ADN_AE(f"test-edge-{int(time.time()*1000)}", self.cse)
        ae.register()
        windows = generate_scenario("normal", 3, seed=2)
        resp = ae.push_sensor_window(windows[0])
        self.assertEqual(resp.response_status_code, ResponseStatusCode.CREATED)
        self.assertEqual(resp.content["con"]["kind"], "sensorWindow")

        latest = self.cse.latest_content_instances(ae.app_name, "sensorData", limit=1)
        self.assertEqual(latest[0]["con"]["index"], windows[0].index)

    def test_pipeline_runs_end_to_end_against_real_cse(self):
        import time
        pipeline = EdgeToMecPipeline(
            cfg=Config(NUM_WORKERS=4),
            edge_node_name=f"test-edge-{int(time.time()*1000)}",
            seed=3, cse_http_url=CSE_HTTP_URL,
        )
        report = pipeline.run(scenario="critical_event", num_windows=5, seed=3, num_workers=4)
        self.assertEqual(len(report.telemetry), 5)
        self.assertGreater(report.tpu_report.tasks_completed, 0)


if __name__ == "__main__":
    unittest.main()
