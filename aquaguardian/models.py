from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class SensorFrame:
    flow_lps: float
    pressure_bar: float
    pump_vibration_mm_s: float
    turbidity_ntu: float
    soil_moisture_pct: float
    plant_stress: float
    temperature_c: float
    humidity_pct: float


@dataclass(frozen=True)
class CandidateAction:
    name: str
    parameters: Dict[str, float]


@dataclass
class SimulationResult:
    action: str
    water_loss_l: float
    energy_kwh: float
    safety_risk: float
    service_disruption_min: float
    crop_stress: float
    contamination_exposure: float
    fire_risk: float
    passed: bool = False
    score: float = 0.0
    reasons: List[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    scenario: str
    detected_risk: float
    selected_action: str
    confidence: float
    reliability: float
    iterations: int
    executed: bool
    evidence: List[str]
    alternatives: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)
