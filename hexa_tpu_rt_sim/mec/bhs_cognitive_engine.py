"""
mec/bhs_cognitive_engine.py
============================
MEC Application-level "BHS Cognitive Engine" -- the compute logic that
runs as an ETSI-MEC application (see mec/mec_platform.py) once oneM2M
contentInstances arrive at the MEC host. It is deliberately separate
from models/bdo_skin.py's MAC-cost stimulus generator: bdo_skin.py
answers "how expensive is this workload for the TPU to execute",
whereas this module answers "what does the workload actually
*compute*" -- real (if simplified) numerical logic for the three
named roles, using matrix operations (numpy), not placeholder MAC
counts.

Three cooperating components, matching the requested architecture:

  BatForecaster       -- Remaining Useful Life (RUL) + stress-trend
                          prediction via least-squares trend fitting
                          (matrix operations) over a per-cell strain
                          history buffer.
  HermitCrabEvaluator  -- overall mechanical-stability assessment +
                           veto logic: blocks any candidate actuation
                           action whose frequency sits inside a
                           destructive-resonance band around the
                           panel's current estimated natural frequency.
  SquidController      -- multi-objective trade-off balancing across
                           safety / productivity / power, scored over
                           whatever candidate actions Hermit Crab did
                           NOT veto.

None of the numeric constants below are re-derived from BDO-SKIN's own
report (a physics/control simulation, not a control-algorithm spec);
they are named, adjustable assumptions calibrated to be architecturally
plausible, in the same spirit as models/bdo_skin.py's MAC-cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------
# Bat Forecaster
# --------------------------------------------------------------------

@dataclass
class RULEstimate:
    trend_slope_kpa_per_window: float
    trend_intercept_kpa: float
    projected_windows_to_threshold: Optional[float]   # None = not currently trending toward failure
    remaining_useful_life_windows: Optional[float]
    r_squared: float


class BatForecaster:
    """Computes Remaining Useful Life (RUL) and stress trends via
    matrix operations (ordinary least squares, solved through the
    normal-equation matrix form) over historical strain-window data --
    the same role a bat's biosonar plays in the source metaphor:
    forward-looking, cheap, continuous echo-ranging against an
    approaching threshold rather than a one-shot classification."""

    def __init__(self, failure_threshold_kpa: float = 24000.0, history_len: int = 15):
        self.failure_threshold_kpa = failure_threshold_kpa
        self.history_len = history_len

    def _ols_fit(self, y: Sequence[float]) -> tuple:
        """y[i] = slope * i + intercept, via the normal-equation matrix
        solve X^T X beta = X^T y (explicit matrix operations, not
        numpy.polyfit's black box) so the "matrix operations" the
        architecture calls for are literal here."""
        n = len(y)
        x = np.arange(n, dtype=float)
        X = np.vstack([x, np.ones(n)]).T          # design matrix, shape (n, 2)
        y_arr = np.asarray(y, dtype=float)
        XtX = X.T @ X
        Xty = X.T @ y_arr
        beta = np.linalg.solve(XtX, Xty)           # [slope, intercept]
        slope, intercept = float(beta[0]), float(beta[1])

        y_hat = X @ beta
        ss_res = float(np.sum((y_arr - y_hat) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        return slope, intercept, r_squared

    def forecast(self, strain_history_kpa: Sequence[float]) -> RULEstimate:
        if len(strain_history_kpa) < 2:
            return RULEstimate(0.0, strain_history_kpa[-1] if strain_history_kpa else 0.0,
                                None, None, 0.0)

        window = list(strain_history_kpa)[-self.history_len:]
        slope, intercept, r2 = self._ols_fit(window)
        current = window[-1]
        n = len(window)

        if slope <= 1e-9 or current >= self.failure_threshold_kpa:
            # Not trending toward failure (or already past threshold):
            # RUL is undefined/zero rather than a meaningless projection.
            windows_to_threshold = 0.0 if current >= self.failure_threshold_kpa else None
            rul = 0.0 if current >= self.failure_threshold_kpa else None
        else:
            # Solve slope * t + current = threshold for t (windows from now).
            windows_to_threshold = (self.failure_threshold_kpa - current) / slope
            rul = windows_to_threshold

        return RULEstimate(
            trend_slope_kpa_per_window=slope,
            trend_intercept_kpa=intercept,
            projected_windows_to_threshold=windows_to_threshold,
            remaining_useful_life_windows=rul,
            r_squared=r2,
        )


# --------------------------------------------------------------------
# Hermit Crab Evaluator
# --------------------------------------------------------------------

@dataclass
class CandidateAction:
    name: str
    actuation_freq_hz: float
    safety_benefit: float      # 0..1, higher = better for arresting the fault
    productivity_cost: float   # 0..1, higher = more disruptive to normal operation
    power_cost_w: float


@dataclass
class StabilityAssessment:
    stability_score: float               # 0..1, 1 = fully stable
    estimated_natural_freq_hz: float
    vetoed_actions: List[str]
    allowed_actions: List[CandidateAction]


class HermitCrabEvaluator:
    """Assesses overall mechanical stability from the current sensor
    state and applies veto logic to the Squid Controller's candidate
    action set: like a hermit crab testing a shell before committing
    to it, every candidate action is checked against a
    destructive-harmonic-vibration risk model before it is allowed
    through, regardless of how attractive its safety/productivity
    score looks."""

    def __init__(self, resonance_guard_band_hz: float = 2.0,
                 stability_severity_weight: float = 0.9):
        self.resonance_guard_band_hz = resonance_guard_band_hz
        self.stability_severity_weight = stability_severity_weight

    def _estimate_natural_freq_hz(self, mean_temp_c: float, anomaly_severity: float) -> float:
        """A structural panel's natural frequency drops as damage
        (here, anomaly_severity) accumulates and as thermal expansion
        reduces effective stiffness -- a standard qualitative
        structural-health-monitoring relationship. Base frequency and
        sensitivity coefficients are named, adjustable assumptions
        (this module does not claim BDO-SKIN's own panel's real modal
        properties)."""
        base_freq_hz = 40.0
        thermal_derate = max(0.0, mean_temp_c - 25.0) * 0.05
        damage_derate = anomaly_severity * 10.0
        return max(5.0, base_freq_hz - thermal_derate - damage_derate)

    def assess(self, mean_temp_c: float, anomaly_severity: float,
               candidates: Sequence[CandidateAction]) -> StabilityAssessment:
        natural_freq = self._estimate_natural_freq_hz(mean_temp_c, anomaly_severity)
        stability = max(0.0, 1.0 - self.stability_severity_weight * anomaly_severity)

        vetoed, allowed = [], []
        for c in candidates:
            in_resonance_band = abs(c.actuation_freq_hz - natural_freq) <= self.resonance_guard_band_hz
            if in_resonance_band:
                vetoed.append(c.name)
            else:
                allowed.append(c)

        return StabilityAssessment(
            stability_score=stability,
            estimated_natural_freq_hz=natural_freq,
            vetoed_actions=vetoed,
            allowed_actions=allowed,
        )


# --------------------------------------------------------------------
# Squid Controller
# --------------------------------------------------------------------

@dataclass
class ControlDecision:
    chosen_action: Optional[str]
    scores: Dict[str, float]
    weights: Dict[str, float]
    vetoed_actions: List[str]


class SquidController:
    """Dynamically balances real-time operational trade-offs between
    safety, productivity, and power consumption -- a squid's
    distributed, many-armed coordination as the metaphor for a
    controller that must reweight competing objectives on every cycle
    rather than optimizing a single fixed goal. Operates only over the
    action set HermitCrabEvaluator has NOT vetoed: the veto is a hard
    constraint, not one more term in the weighted score."""

    def __init__(self, w_safety: float = 0.35, w_productivity: float = 0.45,
                 w_power: float = 0.20, power_cost_scale_w: float = 50.0):
        # Baseline (zero-severity) weights deliberately favor
        # productivity over safety-at-all-costs: with nothing wrong,
        # a rational controller shouldn't prefer full_stop over
        # hold_position just because full_stop's raw safety_benefit is
        # higher. reweight() is what shifts this balance as severity
        # rises -- the baseline itself should reflect routine
        # efficiency-seeking operation, not a standing bias toward
        # maximal caution.
        total = w_safety + w_productivity + w_power
        self.w_safety = w_safety / total
        self.w_productivity = w_productivity / total
        self.w_power = w_power / total
        self.power_cost_scale_w = power_cost_scale_w

    def reweight(self, anomaly_severity: float) -> Dict[str, float]:
        """As severity climbs, safety's weight is pushed up and
        productivity/power's are pushed down -- a smooth, bounded
        reweighting (not a hard mode switch) between routine
        efficiency-seeking behavior and emergency-response behavior.
        The boost is intentionally steep enough that a genuinely
        severe (severity -> 1.0) event dominates the score, matching
        "safety overrides productivity/power once things are bad
        enough" without hand-coding that as a discrete rule."""
        safety_boost = anomaly_severity * 0.9
        w_safety = self.w_safety + safety_boost
        w_productivity = max(0.02, self.w_productivity - safety_boost * 0.7)
        w_power = max(0.02, self.w_power - safety_boost * 0.5)
        total = w_safety + w_productivity + w_power
        return {"safety": w_safety / total, "productivity": w_productivity / total,
                "power": w_power / total}

    def decide(self, candidates: Sequence[CandidateAction],
               assessment: StabilityAssessment, anomaly_severity: float) -> ControlDecision:
        weights = self.reweight(anomaly_severity)
        scores: Dict[str, float] = {}
        for c in assessment.allowed_actions:
            power_penalty = min(1.0, c.power_cost_w / self.power_cost_scale_w)
            score = (weights["safety"] * c.safety_benefit
                     - weights["productivity"] * c.productivity_cost
                     - weights["power"] * power_penalty)
            scores[c.name] = score

        chosen = max(scores, key=scores.get) if scores else None
        return ControlDecision(chosen_action=chosen, scores=scores, weights=weights,
                                vetoed_actions=assessment.vetoed_actions)


# --------------------------------------------------------------------
# Combined engine
# --------------------------------------------------------------------

DEFAULT_CANDIDATE_ACTIONS: List[CandidateAction] = [
    CandidateAction("hold_position", actuation_freq_hz=0.0,
                     safety_benefit=0.2, productivity_cost=0.0, power_cost_w=2.0),
    CandidateAction("active_damping_low", actuation_freq_hz=8.0,
                     safety_benefit=0.5, productivity_cost=0.2, power_cost_w=12.0),
    CandidateAction("active_damping_high", actuation_freq_hz=18.0,
                     safety_benefit=0.75, productivity_cost=0.4, power_cost_w=30.0),
    # Sits inside the natural-frequency band the panel sweeps through
    # as damage/thermal derating progress (see
    # HermitCrabEvaluator._estimate_natural_freq_hz) -- included
    # deliberately so the veto path is actually exercised sometimes,
    # not just theoretically possible.
    CandidateAction("active_damping_resonant_risk", actuation_freq_hz=33.0,
                     safety_benefit=0.85, productivity_cost=0.4, power_cost_w=20.0),
    CandidateAction("load_shed", actuation_freq_hz=2.0,
                     safety_benefit=0.9, productivity_cost=0.8, power_cost_w=5.0),
    CandidateAction("full_stop", actuation_freq_hz=0.0,
                     safety_benefit=1.0, productivity_cost=1.0, power_cost_w=1.0),
]


@dataclass
class CognitiveResult:
    rul: RULEstimate
    stability: StabilityAssessment
    control: ControlDecision


class BHSCognitiveEngine:
    """The MEC-application-level cognitive engine: Bat Forecaster +
    Hermit Crab Evaluator + Squid Controller, composed into a single
    per-window evaluation. Deployed as one MecApp by mec/mec_platform.py."""

    def __init__(self,
                 bat: Optional[BatForecaster] = None,
                 hermit_crab: Optional[HermitCrabEvaluator] = None,
                 squid: Optional[SquidController] = None,
                 candidate_actions: Optional[Sequence[CandidateAction]] = None):
        self.bat = bat or BatForecaster()
        self.hermit_crab = hermit_crab or HermitCrabEvaluator()
        self.squid = squid or SquidController()
        self.candidate_actions = list(candidate_actions or DEFAULT_CANDIDATE_ACTIONS)

    def evaluate(self, strain_history_kpa: Sequence[float], mean_temp_c: float,
                 anomaly_severity: float) -> CognitiveResult:
        rul = self.bat.forecast(strain_history_kpa)
        stability = self.hermit_crab.assess(mean_temp_c, anomaly_severity, self.candidate_actions)
        control = self.squid.decide(self.candidate_actions, stability, anomaly_severity)
        return CognitiveResult(rul=rul, stability=stability, control=control)
