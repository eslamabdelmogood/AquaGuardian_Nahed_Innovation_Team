"""
mec/domains/water_cognitive_engine.py
========================================
Water-infrastructure MEC cognitive engine. Demonstrates the platform's
domain-neutral claim at the MEC layer, the same way
reflex/domains/{aviation,water}.py demonstrates it at the reflex
layer: swap only what physically has to differ, reuse everything else
as literally the same objects.

Aviation's MEC layer needs no new code at all: mec/bhs_cognitive_engine.py
was already written as structural/airframe monitoring (resonance,
remaining-useful-life, safety/productivity/power balancing under
vibration and thermal stress) -- that IS the aviation domain's
cognitive engine, unmodified, since BDO-SKIN's original framing is
literally an aircraft-wing optical skin. This module is what the water
vertical actually needs on top of that:

  - Bat Forecaster: REUSED UNMODIFIED (`mec.bhs_cognitive_engine.BatForecaster`).
    Structural RUL forecasts a RISING trend toward a failure ceiling;
    a growing pipeline leak shows up as a DECLINING baseline pressure
    trend instead. Rather than write a second forecaster, this module
    transforms the input into a "cumulative pressure deficit" series
    (baseline - reading, which rises as a leak grows) and feeds that,
    unmodified, into the same least-squares matrix-solve forecaster --
    same object, same math, different data framing.
  - Hermit Crab Evaluator: genuinely different physics, so genuinely
    new code (`WaterHammerHermitCrabEvaluator` below): vetoes any
    candidate flow-rate change whose estimated Joukowsky surge
    pressure exceeds a hard structural limit, instead of vetoing
    actions whose actuation frequency falls in a resonance band. Same
    *role* (hard veto on Squid's candidate set, independent of how
    valuable the action looks), different underlying model.
  - Squid Controller: REUSED UNMODIFIED
    (`mec.bhs_cognitive_engine.SquidController`). Its
    safety/productivity/power weighted scoring is duck-typed against
    whatever `CandidateAction`-shaped objects it's given
    (`.name`/`.safety_benefit`/`.productivity_cost`/`.power_cost_w`) --
    `WaterCandidateAction` below matches that shape without importing
    or subclassing the aviation `CandidateAction` class, so the same
    controller object works for pump/valve scheduling exactly as it
    does for structural-actuator scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from mec.bhs_cognitive_engine import BatForecaster, SquidController, RULEstimate, ControlDecision


# --------------------------------------------------------------------
# Water-domain candidate actions and stability assessment
# --------------------------------------------------------------------

@dataclass
class WaterCandidateAction:
    """Same four attributes SquidController.decide() actually reads
    (name/safety_benefit/productivity_cost/power_cost_w) as
    mec.bhs_cognitive_engine.CandidateAction -- deliberately NOT a
    subclass of it, to make the duck-typed reuse explicit rather than
    implicit. `flow_rate_change_lps` replaces `actuation_freq_hz` as
    the domain-specific field WaterHammerHermitCrabEvaluator reads to
    assess risk."""
    name: str
    flow_rate_change_lps: float      # negative = closing/reducing flow, positive = opening/increasing
    safety_benefit: float
    productivity_cost: float
    power_cost_w: float


@dataclass
class WaterStabilityAssessment:
    stability_score: float
    worst_case_surge_kpa: float
    vetoed_actions: List[str]
    allowed_actions: List[WaterCandidateAction]


# --------------------------------------------------------------------
# Hermit Crab: water-hammer veto physics
# --------------------------------------------------------------------

class WaterHammerHermitCrabEvaluator:
    """Water-domain Hermit Crab Evaluator: assesses overall hydraulic
    stability and applies veto logic against destructive water-hammer
    transients, instead of destructive harmonic vibration.

    Physics: the Joukowsky equation estimates the pressure surge from
    a sudden flow velocity change, dP = rho * a * dV (fluid density *
    pressure-wave speed * velocity change) -- the standard first-order
    model for why slamming a valve shut (rather than closing it in
    stages) can rupture a pipe even at normal operating pressure. Any
    candidate action whose estimated surge exceeds
    `surge_pressure_veto_threshold_kpa` is vetoed outright, the same
    hard-constraint role the structural evaluator's resonance guard-
    band plays -- Squid never even sees it as an option, regardless of
    how attractive its safety score looks.

    All constants below are named, adjustable assumptions (typical
    water-transmission-main figures), not a re-derivation of any
    specific pipeline's real hydraulic properties -- same convention
    as the structural evaluator's resonance-frequency model.
    """

    def __init__(self, fluid_density_kg_m3: float = 1000.0,
                 pressure_wave_speed_m_s: float = 1200.0,
                 pipe_cross_section_m2: float = 0.2,
                 surge_pressure_veto_threshold_kpa: float = 500.0,
                 stability_severity_weight: float = 0.9):
        self.fluid_density_kg_m3 = fluid_density_kg_m3
        self.pressure_wave_speed_m_s = pressure_wave_speed_m_s
        self.pipe_cross_section_m2 = pipe_cross_section_m2
        self.surge_pressure_veto_threshold_kpa = surge_pressure_veto_threshold_kpa
        self.stability_severity_weight = stability_severity_weight

    def _estimate_surge_kpa(self, flow_rate_change_lps: float) -> float:
        delta_v_m_s = (flow_rate_change_lps / 1000.0) / self.pipe_cross_section_m2  # L/s -> m^3/s -> m/s
        surge_pa = self.fluid_density_kg_m3 * self.pressure_wave_speed_m_s * abs(delta_v_m_s)
        return surge_pa / 1000.0  # Pa -> kPa

    def assess(self, leak_severity: float,
               candidates: Sequence[WaterCandidateAction]) -> WaterStabilityAssessment:
        stability = max(0.0, 1.0 - self.stability_severity_weight * leak_severity)

        vetoed, allowed = [], []
        worst_case = 0.0
        for c in candidates:
            surge_kpa = self._estimate_surge_kpa(c.flow_rate_change_lps)
            worst_case = max(worst_case, surge_kpa)
            if surge_kpa >= self.surge_pressure_veto_threshold_kpa:
                vetoed.append(c.name)
            else:
                allowed.append(c)

        return WaterStabilityAssessment(
            stability_score=stability, worst_case_surge_kpa=worst_case,
            vetoed_actions=vetoed, allowed_actions=allowed,
        )


# --------------------------------------------------------------------
# Default candidate action set
# --------------------------------------------------------------------

DEFAULT_WATER_CANDIDATE_ACTIONS: List[WaterCandidateAction] = [
    WaterCandidateAction("hold_flow", flow_rate_change_lps=0.0,
                          safety_benefit=0.2, productivity_cost=0.0, power_cost_w=2.0),
    WaterCandidateAction("gradual_flow_reduction", flow_rate_change_lps=-5.0,
                          safety_benefit=0.6, productivity_cost=0.2, power_cost_w=8.0),
    WaterCandidateAction("reroute_via_secondary_main", flow_rate_change_lps=-15.0,
                          safety_benefit=0.7, productivity_cost=0.3, power_cost_w=15.0),
    WaterCandidateAction("staged_valve_shutdown", flow_rate_change_lps=-40.0,
                          safety_benefit=0.85, productivity_cost=0.6, power_cost_w=25.0),
    # Deliberately included as a stress case, the same way
    # reflex/domains/water.py and mec/bhs_cognitive_engine.py each
    # include one deliberately-vetoable option: an abrupt full
    # shutdown is the *most tempting* action by raw safety_benefit,
    # and the one the veto logic must actually block regardless.
    WaterCandidateAction("emergency_full_shutdown", flow_rate_change_lps=-100.0,
                          safety_benefit=1.0, productivity_cost=1.0, power_cost_w=5.0),
]


# --------------------------------------------------------------------
# Combined engine
# --------------------------------------------------------------------

@dataclass
class WaterCognitiveResult:
    rul: RULEstimate
    stability: WaterStabilityAssessment
    control: ControlDecision


class WaterBHSCognitiveEngine:
    """Water-infrastructure MEC application: Bat Forecaster (reused
    unmodified) + Water-Hammer Hermit Crab (new physics) + Squid
    Controller (reused unmodified). Deployable behind
    mec/mec_platform.py's MecApp lifecycle exactly like
    mec.bhs_cognitive_engine.BHSCognitiveEngine is for aviation."""

    def __init__(self,
                 bat: Optional[BatForecaster] = None,
                 hermit_crab: Optional[WaterHammerHermitCrabEvaluator] = None,
                 squid: Optional[SquidController] = None,
                 candidate_actions: Optional[Sequence[WaterCandidateAction]] = None,
                 baseline_pressure_kpa: float = 600.0):
        # failure_threshold_kpa here means "maximum tolerable
        # cumulative pressure deficit before rupture/dry-pipe risk" --
        # see evaluate()'s baseline-minus-reading transform below.
        self.bat = bat or BatForecaster(failure_threshold_kpa=150.0, history_len=15)
        self.hermit_crab = hermit_crab or WaterHammerHermitCrabEvaluator()
        self.squid = squid or SquidController()
        self.candidate_actions = list(candidate_actions or DEFAULT_WATER_CANDIDATE_ACTIONS)
        self.baseline_pressure_kpa = baseline_pressure_kpa

    def evaluate(self, pressure_history_kpa: Sequence[float],
                 leak_severity: float) -> WaterCognitiveResult:
        # Transform declining pressure into a rising "deficit" series
        # so the unmodified BatForecaster (which projects a RISING
        # trend toward a ceiling) can be reused as-is for a pipeline's
        # DECLINING trend toward a floor.
        deficit_history = [self.baseline_pressure_kpa - p for p in pressure_history_kpa]
        rul = self.bat.forecast(deficit_history)

        stability = self.hermit_crab.assess(leak_severity, self.candidate_actions)
        control = self.squid.decide(self.candidate_actions, stability, leak_severity)
        return WaterCognitiveResult(rul=rul, stability=stability, control=control)
