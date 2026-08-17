"""
reflex/domains/aviation.py
=============================
Domain profile #1: Aircraft / UAV / Urban Air Mobility Detect-and-Avoid
(DAA) in degraded (fog/low-visibility) conditions.

Plugs into the domain-neutral reflex/reflex_kernel.py: this module
only supplies (a) what a "local sample" looks like for this domain,
(b) the trigger rule (`AviationDAATrigger`), and (c) a scenario
generator to exercise it -- the kernel, deadline enforcement, and
actuation plumbing are entirely shared with the water-infrastructure
profile (reflex/domains/water.py).

Trigger model: closing-range time-to-collision (TTC) against the
nearest tracked object, degraded by a visibility/fog attenuation
factor the way onboard optical/LiDAR ranging actually degrades in
fog (SNR drops, effective max range shrinks, and -- modeled here --
the required reaction margin should widen because sensor confidence
drops too). This is a simplified TTC heuristic for demonstrating a
sub-1ms local decide-and-act path, not a certified DAA algorithm; see
the class docstring for the exact (named, adjustable) assumptions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from reflex.reflex_kernel import ActuationDecision, ReflexTrigger


@dataclass
class AviationProximitySample:
    index: int
    scenario: str
    range_m: float                # distance to nearest tracked object
    closing_speed_mps: float       # positive = closing (approaching)
    visibility_m: float            # effective sensor visibility (fog/weather derated)
    is_near_miss: bool


class AviationDAATrigger:
    """Detect-and-Avoid trigger for degraded-visibility conditions.

    Fires when projected time-to-collision (range / closing_speed)
    drops below a visibility-scaled reaction-margin threshold: in
    heavy fog, effective visibility is short, so even a moderate
    closing speed leaves little TTC margin once an object *is*
    resolved -- the margin threshold widens as visibility shrinks to
    reflect reduced sensor confidence, not just reduced geometry.
    """

    channels_checked = 3   # range, closing_speed, visibility -- the three local readings this trigger reads

    def __init__(self, base_ttc_margin_s: float = 2.0, fog_margin_scale: float = 3.0,
                 reference_visibility_m: float = 1500.0):
        self.base_ttc_margin_s = base_ttc_margin_s
        self.fog_margin_scale = fog_margin_scale
        self.reference_visibility_m = reference_visibility_m

    def evaluate(self, sample: Dict[str, Any]) -> Optional[ActuationDecision]:
        range_m = sample["range_m"]
        closing_speed = sample["closing_speed_mps"]
        visibility_m = sample["visibility_m"]

        if closing_speed <= 0.0:
            return None   # not approaching -- nothing to react to

        ttc_s = range_m / closing_speed

        # Required margin widens as visibility degrades (fog): a
        # sensor that can barely resolve objects needs more reaction
        # time once it does, not less.
        fog_factor = max(1.0, self.reference_visibility_m / max(visibility_m, 1.0))
        required_margin_s = self.base_ttc_margin_s * min(fog_factor, self.fog_margin_scale)

        if ttc_s < required_margin_s:
            severity = min(1.0, required_margin_s / max(ttc_s, 1e-3) / self.fog_margin_scale)
            return ActuationDecision(
                action="avoidance_maneuver",
                reason=f"TTC {ttc_s:.2f}s < required margin {required_margin_s:.2f}s "
                       f"(visibility {visibility_m:.0f}m)",
                severity=severity,
                metadata={"ttc_s": ttc_s, "required_margin_s": required_margin_s,
                          "visibility_m": visibility_m},
            )
        return None


# --------------------------------------------------------------------
# Scenario generator
# --------------------------------------------------------------------

SCENARIOS = ["clear_sky", "light_haze", "dense_fog_approach", "sudden_incursion"]


def generate_aviation_scenario(scenario: str, n: int, seed: int = 0) -> List[AviationProximitySample]:
    """Named, adjustable assumption set for demo purposes -- not
    flight-test data. Each scenario is a distinct visibility/traffic
    profile a DAA reflex path should behave sensibly under."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown aviation scenario '{scenario}', expected one of {SCENARIOS}")

    rng = random.Random(seed)
    samples: List[AviationProximitySample] = []

    for i in range(n):
        if scenario == "clear_sky":
            visibility = rng.uniform(4000, 10000)
            range_m = rng.uniform(500, 3000)
            closing_speed = rng.uniform(0, 40)
            near_miss = False

        elif scenario == "light_haze":
            visibility = rng.uniform(1200, 3000)
            range_m = rng.uniform(300, 2000)
            closing_speed = rng.uniform(0, 60)
            near_miss = False

        elif scenario == "dense_fog_approach":
            visibility = rng.uniform(80, 400)
            # Range trends down across the run as an object closes in fog.
            progress = i / max(1, n - 1)
            range_m = max(15.0, 600.0 * (1.0 - progress) + rng.uniform(-20, 20))
            closing_speed = rng.uniform(25, 70)
            near_miss = progress > 0.7

        else:  # sudden_incursion
            visibility = rng.uniform(1500, 5000)
            if 0.4 * n <= i <= 0.42 * n:
                range_m = rng.uniform(20, 60)
                closing_speed = rng.uniform(60, 110)
                near_miss = True
            else:
                range_m = rng.uniform(800, 3000)
                closing_speed = rng.uniform(0, 30)
                near_miss = False

        samples.append(AviationProximitySample(
            index=i, scenario=scenario, range_m=range_m, closing_speed_mps=closing_speed,
            visibility_m=visibility, is_near_miss=near_miss,
        ))

    return samples


def sample_to_dict(s: AviationProximitySample) -> Dict[str, Any]:
    return {"range_m": s.range_m, "closing_speed_mps": s.closing_speed_mps,
            "visibility_m": s.visibility_m}
