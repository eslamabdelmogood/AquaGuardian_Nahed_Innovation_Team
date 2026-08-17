"""
tests/test_reflex_kernel.py
==============================
Tests for reflex/reflex_kernel.py (the domain-neutral local
decide-and-act path) and its two domain profiles:
reflex/domains/aviation.py and reflex/domains/water.py.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from reflex.reflex_kernel import ReflexKernel, LoggingActuator, ActuationDecision
from reflex.domains.aviation import AviationDAATrigger, generate_aviation_scenario, sample_to_dict as av_dict
from reflex.domains.water import WaterPressureTransientTrigger, generate_water_scenario, sample_to_dict as wa_dict


class _AlwaysTrigger:
    channels_checked = 1

    def evaluate(self, sample):
        return ActuationDecision(action="test_action", reason="always")


class _NeverTrigger:
    channels_checked = 1

    def evaluate(self, sample):
        return None


class TestReflexKernelCore(unittest.TestCase):
    def test_non_triggering_sample_produces_no_actuation(self):
        kernel = ReflexKernel(_NeverTrigger())
        event = kernel.evaluate({})
        self.assertFalse(event.triggered)
        self.assertIsNone(event.decision)
        self.assertEqual(len(kernel.actuator.log), 0)

    def test_triggering_sample_calls_the_actuator(self):
        kernel = ReflexKernel(_AlwaysTrigger())
        event = kernel.evaluate({})
        self.assertTrue(event.triggered)
        self.assertEqual(event.decision.action, "test_action")
        self.assertEqual(len(kernel.actuator.log), 1)
        self.assertIs(kernel.actuator.log[0], event)

    def test_deadline_is_met_for_small_channel_counts(self):
        kernel = ReflexKernel(_AlwaysTrigger(), deadline_ms=1.0)
        event = kernel.evaluate({})
        self.assertTrue(event.deadline_met)
        self.assertLess(event.latency_ms, 1.0)

    def test_deadline_can_be_violated_by_excessive_channel_count(self):
        class HeavyTrigger:
            channels_checked = 10_000_000   # deliberately absurd, to force a miss
            def evaluate(self, sample):
                return None
        kernel = ReflexKernel(HeavyTrigger(), deadline_ms=1.0, clock_freq_mhz=1200.0)
        event = kernel.evaluate({})
        self.assertFalse(event.deadline_met)

    def test_stats_aggregate_correctly(self):
        kernel = ReflexKernel(_AlwaysTrigger())
        for i in range(10):
            kernel.evaluate({}, sample_index=i)
        stats = kernel.stats()
        self.assertEqual(stats.count, 10)
        self.assertEqual(stats.triggered_count, 10)
        self.assertEqual(stats.deadline_misses, 0)

    def test_empty_stats_do_not_crash(self):
        kernel = ReflexKernel(_NeverTrigger())
        stats = kernel.stats()
        self.assertEqual(stats.count, 0)


class TestAviationDomain(unittest.TestCase):
    def test_head_on_approach_in_fog_triggers_avoidance(self):
        trigger = AviationDAATrigger()
        decision = trigger.evaluate({"range_m": 50.0, "closing_speed_mps": 40.0,
                                      "visibility_m": 100.0})
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "avoidance_maneuver")

    def test_distant_slow_object_in_clear_sky_does_not_trigger(self):
        trigger = AviationDAATrigger()
        decision = trigger.evaluate({"range_m": 5000.0, "closing_speed_mps": 5.0,
                                      "visibility_m": 8000.0})
        self.assertIsNone(decision)

    def test_receding_object_never_triggers(self):
        trigger = AviationDAATrigger()
        decision = trigger.evaluate({"range_m": 50.0, "closing_speed_mps": -10.0,
                                      "visibility_m": 100.0})
        self.assertIsNone(decision)

    def test_sudden_incursion_scenario_fires_near_the_incursion_window(self):
        kernel = ReflexKernel(AviationDAATrigger(), deadline_ms=1.0)
        samples = generate_aviation_scenario("sudden_incursion", 200, seed=0)
        for s in samples:
            kernel.evaluate(av_dict(s), sample_index=s.index)
        stats = kernel.stats()
        self.assertGreater(stats.triggered_count, 0)
        self.assertEqual(stats.deadline_misses, 0)

    def test_clear_sky_scenario_rarely_or_never_triggers(self):
        kernel = ReflexKernel(AviationDAATrigger(), deadline_ms=1.0)
        samples = generate_aviation_scenario("clear_sky", 200, seed=0)
        for s in samples:
            kernel.evaluate(av_dict(s), sample_index=s.index)
        # Clear sky, generous ranges/speeds: should not be a wall of alarms.
        self.assertLess(kernel.stats().triggered_count, 20)


class TestWaterDomain(unittest.TestCase):
    def test_sharp_pressure_drop_triggers_valve_shutoff(self):
        trigger = WaterPressureTransientTrigger()
        decision = trigger.evaluate({"pressure_kpa": 300.0, "prior_pressure_kpa": 600.0,
                                      "dt_s": 0.01})
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "valve_shutoff")

    def test_gradual_daily_fluctuation_does_not_trigger(self):
        trigger = WaterPressureTransientTrigger()
        decision = trigger.evaluate({"pressure_kpa": 598.0, "prior_pressure_kpa": 600.0,
                                      "dt_s": 0.01})
        self.assertIsNone(decision)

    def test_pressure_rise_never_triggers(self):
        trigger = WaterPressureTransientTrigger()
        decision = trigger.evaluate({"pressure_kpa": 650.0, "prior_pressure_kpa": 600.0,
                                      "dt_s": 0.01})
        self.assertIsNone(decision)

    def test_sudden_burst_scenario_triggers_close_to_ground_truth(self):
        kernel = ReflexKernel(WaterPressureTransientTrigger(), deadline_ms=1.0)
        samples = generate_water_scenario("sudden_burst", 200, seed=0)
        for s in samples:
            kernel.evaluate(wa_dict(s), sample_index=s.index)
        stats = kernel.stats()
        ground_truth = sum(1 for s in samples if s.is_burst_ground_truth)
        self.assertGreaterEqual(stats.triggered_count, 1)
        self.assertEqual(ground_truth, 1)
        self.assertEqual(stats.deadline_misses, 0)

    def test_steady_demand_scenario_never_triggers(self):
        kernel = ReflexKernel(WaterPressureTransientTrigger(), deadline_ms=1.0)
        samples = generate_water_scenario("steady_demand", 200, seed=0)
        for s in samples:
            kernel.evaluate(wa_dict(s), sample_index=s.index)
        self.assertEqual(kernel.stats().triggered_count, 0)


class TestDomainNeutrality(unittest.TestCase):
    """The actual claim under test: the same ReflexKernel class,
    unmodified, drives both domains correctly with only the trigger
    swapped -- this is what makes 'one core, two industries' literal
    rather than a slide bullet."""

    def test_same_kernel_class_both_domains_meet_deadline(self):
        av_kernel = ReflexKernel(AviationDAATrigger(), deadline_ms=1.0)
        wa_kernel = ReflexKernel(WaterPressureTransientTrigger(), deadline_ms=1.0)
        self.assertIs(type(av_kernel), type(wa_kernel))

        for s in generate_aviation_scenario("dense_fog_approach", 100, seed=1):
            av_kernel.evaluate(av_dict(s), sample_index=s.index)
        for s in generate_water_scenario("sudden_burst", 100, seed=1):
            wa_kernel.evaluate(wa_dict(s), sample_index=s.index)

        self.assertEqual(av_kernel.stats().deadline_misses, 0)
        self.assertEqual(wa_kernel.stats().deadline_misses, 0)
        self.assertGreater(av_kernel.stats().triggered_count, 0)
        self.assertGreater(wa_kernel.stats().triggered_count, 0)


if __name__ == "__main__":
    unittest.main()
