from __future__ import annotations

from dataclasses import dataclass

from .detector import analyze
from .models import CandidateAction, SensorFrame
from .policy import ACTIONS


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    action: CandidateAction
    detection_delay_s: float
    explanation: str


def reactive_decision(frame: SensorFrame, scenario: str) -> StrategyDecision:
    """Model a conventional alarm-and-response workflow.

    The action is selected only after an assumed operator/alarm delay. The
    strongest emergency action is used, but no candidate simulation is run.
    """
    preferred = {
        "pipeline_leak": "isolate_zone",
        "pump_degradation": "shutdown",
        "water_contamination": "isolate_and_flush",
        "drought_stress": "fixed_schedule",
        "wildfire_risk": "mist_and_drones",
    }[scenario]
    action = next(item for item in ACTIONS[scenario] if item.name == preferred)
    return StrategyDecision(
        strategy="reactive_baseline",
        action=action,
        detection_delay_s=300.0,
        explanation="Fixed emergency response after a modeled five-minute alarm/operator delay.",
    )


def detection_only_decision(frame: SensorFrame, scenario: str) -> StrategyDecision:
    """Model an AI detector that triggers a fixed rule without a digital twin."""
    risk = analyze(frame)[scenario]
    preferred = {
        "pipeline_leak": "reduce_pressure",
        "pump_degradation": "reduce_speed",
        "water_contamination": "divert_to_treatment",
        "drought_stress": "fixed_schedule",
        "wildfire_risk": "mist_only",
    }[scenario]
    action_name = preferred if risk >= 0.35 else "do_nothing"
    action = next(item for item in ACTIONS[scenario] if item.name == action_name)
    return StrategyDecision(
        strategy="detection_only_ai",
        action=action,
        detection_delay_s=2.0,
        explanation="Immediate fixed rule selected from detector output; no candidate simulation or stress testing.",
    )
