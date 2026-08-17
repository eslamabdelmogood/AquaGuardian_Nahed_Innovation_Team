"""
reflex/domains/water.py
==========================
Domain profile #2: Smart water infrastructure -- main transmission
pipelines, valves, pumping stations. Local reflex mission: detect a
sudden pressure-drop transient (pipe burst / water-hammer event) and
shut the local safety valve before flooding, before the reading even
leaves the edge.

Same pattern as reflex/domains/aviation.py: only the sample shape,
trigger rule, and scenario generator are domain-specific here -- the
kernel, deadline enforcement, and actuation plumbing
(reflex/reflex_kernel.py) are the shared, domain-neutral core.

Trigger model: rate-of-change of local pressure over a short rolling
window. A genuine pipe burst produces a sharp, fast pressure drop
(the transient signature water-hammer analysis is built around);
gradual demand-driven pressure changes (e.g. daily consumption
curves) do not. This is a simplified rate-of-change heuristic for
demonstrating the reflex path, not a certified water-hammer/transient
analysis; see the class docstring for the exact (named, adjustable)
assumptions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from reflex.reflex_kernel import ActuationDecision, ReflexTrigger


@dataclass
class WaterPressureSample:
    index: int
    scenario: str
    pressure_kpa: float
    prior_pressure_kpa: float
    dt_s: float                  # time between this sample and the prior one
    is_burst_ground_truth: bool


class WaterPressureTransientTrigger:
    """Water-hammer / pipe-burst reflex trigger: fires when the local
    pressure drop rate exceeds a threshold within one short sampling
    interval -- the fast-transient signature of a sudden burst, as
    opposed to slow, demand-driven pressure changes."""

    channels_checked = 2   # current pressure reading, prior pressure reading

    def __init__(self, drop_rate_threshold_kpa_per_s: float = 400.0,
                 minimum_absolute_drop_kpa: float = 15.0):
        self.drop_rate_threshold_kpa_per_s = drop_rate_threshold_kpa_per_s
        self.minimum_absolute_drop_kpa = minimum_absolute_drop_kpa

    def evaluate(self, sample: Dict[str, Any]) -> Optional[ActuationDecision]:
        pressure = sample["pressure_kpa"]
        prior = sample["prior_pressure_kpa"]
        dt_s = max(sample["dt_s"], 1e-3)

        drop = prior - pressure
        if drop < self.minimum_absolute_drop_kpa:
            return None   # noise floor -- not a meaningful drop at all

        drop_rate = drop / dt_s
        if drop_rate >= self.drop_rate_threshold_kpa_per_s:
            severity = min(1.0, drop_rate / (self.drop_rate_threshold_kpa_per_s * 3.0))
            return ActuationDecision(
                action="valve_shutoff",
                reason=f"pressure drop rate {drop_rate:.1f} kPa/s >= threshold "
                       f"{self.drop_rate_threshold_kpa_per_s:.1f} kPa/s "
                       f"(dropped {drop:.1f} kPa in {dt_s*1000:.1f}ms)",
                severity=severity,
                metadata={"drop_rate_kpa_per_s": drop_rate, "drop_kpa": drop},
            )
        return None


# --------------------------------------------------------------------
# Scenario generator
# --------------------------------------------------------------------

SCENARIOS = ["steady_demand", "daily_cycle", "gradual_leak", "sudden_burst"]


def generate_water_scenario(scenario: str, n: int, seed: int = 0,
                             dt_s: float = 0.01) -> List[WaterPressureSample]:
    """Named, adjustable assumption set for demo purposes -- not field
    SCADA data. dt_s defaults to 10ms between samples (a plausible
    edge sampling interval for this kind of transient detection)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown water scenario '{scenario}', expected one of {SCENARIOS}")

    rng = random.Random(seed)
    samples: List[WaterPressureSample] = []
    base_pressure = 600.0   # kPa, typical main-line operating pressure
    pressure = base_pressure
    burst_index = int(n * 0.5) if scenario == "sudden_burst" else -1

    for i in range(n):
        prior = pressure
        is_burst = False

        if scenario == "steady_demand":
            pressure = base_pressure + rng.uniform(-3, 3)

        elif scenario == "daily_cycle":
            import math
            pressure = base_pressure + 40 * math.sin(i / max(1, n) * math.pi) + rng.uniform(-3, 3)

        elif scenario == "gradual_leak":
            # Slow decline over the run -- a growing micro-leak, not a burst.
            pressure = base_pressure - (i / max(1, n)) * 80 + rng.uniform(-3, 3)

        else:  # sudden_burst
            if i == burst_index:
                pressure = prior - rng.uniform(250, 400)   # sharp single-step drop
                is_burst = True
            elif burst_index <= i < burst_index + 5:
                pressure = prior + rng.uniform(-5, 5)       # stays low post-burst
            else:
                pressure = base_pressure + rng.uniform(-3, 3)

        samples.append(WaterPressureSample(
            index=i, scenario=scenario, pressure_kpa=pressure, prior_pressure_kpa=prior,
            dt_s=dt_s, is_burst_ground_truth=is_burst,
        ))

    return samples


def sample_to_dict(s: WaterPressureSample) -> Dict[str, Any]:
    return {"pressure_kpa": s.pressure_kpa, "prior_pressure_kpa": s.prior_pressure_kpa,
            "dt_s": s.dt_s}
