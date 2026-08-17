from __future__ import annotations
from .models import CandidateAction, SimulationResult


ACTIONS = {
    "pipeline_leak": [
        CandidateAction("do_nothing", {}),
        CandidateAction("reduce_pressure", {"pressure_reduction_pct": 25}),
        CandidateAction("isolate_zone", {"valve_delay_s": 2}),
    ],
    "pump_degradation": [
        CandidateAction("do_nothing", {}),
        CandidateAction("reduce_speed", {"speed_reduction_pct": 30}),
        CandidateAction("switch_to_backup", {}),
        CandidateAction("shutdown", {}),
    ],
    "water_contamination": [
        CandidateAction("do_nothing", {}),
        CandidateAction("divert_to_treatment", {}),
        CandidateAction("isolate_and_flush", {}),
    ],
    "drought_stress": [
        CandidateAction("do_nothing", {}),
        CandidateAction("fixed_schedule", {"duration_min": 45}),
        CandidateAction("precision_irrigation", {"zone_fraction": 0.35}),
    ],
    "wildfire_risk": [
        CandidateAction("do_nothing", {}),
        CandidateAction("mist_only", {"duration_min": 8}),
        CandidateAction("mist_and_drones", {"drone_count": 2}),
    ],
}


def validate(result: SimulationResult) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if result.safety_risk > 0.35:
        reasons.append("safety risk above 0.35")
    if result.contamination_exposure > 20:
        reasons.append("contamination exposure above limit")
    if result.fire_risk > 0.40:
        reasons.append("fire risk above limit")
    if result.crop_stress > 0.65:
        reasons.append("crop stress above limit")
    if result.water_loss_l > 400:
        reasons.append("water loss above limit")
    return not reasons, reasons


def score(result: SimulationResult) -> float:
    # Lower cost is better. Safety and exposure receive dominant penalties.
    cost = (
        result.water_loss_l / 500.0
        + result.energy_kwh / 4.0
        + result.service_disruption_min / 60.0
        + 4.0 * result.safety_risk
        + result.crop_stress
        + result.contamination_exposure / 25.0
        + 2.0 * result.fire_risk
    )
    return round(100.0 / (1.0 + cost), 4)
