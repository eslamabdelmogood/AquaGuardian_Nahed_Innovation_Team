from __future__ import annotations
import random
from .models import CandidateAction, SensorFrame, SimulationResult


def simulate(
    scenario: str,
    frame: SensorFrame,
    action: CandidateAction,
    *,
    severity: float = 1.0,
    sensor_noise: float = 0.0,
    communication_loss: bool = False,
    seed: int = 0,
) -> SimulationResult:
    rng = random.Random(seed)
    noise = rng.uniform(-sensor_noise, sensor_noise)

    water_loss = 0.0
    energy = 0.5
    safety = 0.1
    disruption = 0.0
    crop_stress = max(0.0, frame.plant_stress)
    exposure = 0.0
    fire_risk = max(0.0, frame.plant_stress * 0.7)

    if scenario == "pipeline_leak":
        baseline = 900.0 * severity
        if action.name == "isolate_zone":
            water_loss = baseline * 0.08
            disruption = 18.0
        elif action.name == "reduce_pressure":
            water_loss = baseline * 0.35
            disruption = 4.0
            energy = 0.35
        else:
            water_loss = baseline
            safety = 0.55

    elif scenario == "pump_degradation":
        if action.name == "switch_to_backup":
            safety = 0.08
            disruption = 2.0
            energy = 1.15
        elif action.name == "reduce_speed":
            safety = 0.22
            disruption = 5.0
            energy = 0.55
        elif action.name == "shutdown":
            safety = 0.05
            disruption = 30.0
            energy = 0.1
        else:
            safety = 0.82 * severity
            disruption = 8.0

    elif scenario == "water_contamination":
        base_exposure = 100.0 * severity
        if action.name == "isolate_and_flush":
            exposure = base_exposure * 0.04
            water_loss = 120.0
            disruption = 25.0
        elif action.name == "divert_to_treatment":
            exposure = base_exposure * 0.12
            energy = 1.4
            disruption = 8.0
        else:
            exposure = base_exposure
            safety = 0.95

    elif scenario == "drought_stress":
        base_crop = min(1.0, frame.plant_stress * severity)
        if action.name == "precision_irrigation":
            crop_stress = base_crop * 0.30
            water_loss = 95.0 * severity
            energy = 0.75
            fire_risk = base_crop * 0.25
        elif action.name == "fixed_schedule":
            crop_stress = base_crop * 0.52
            water_loss = 260.0 * severity
            energy = 1.15
            fire_risk = base_crop * 0.48
        else:
            crop_stress = base_crop
            water_loss = 15.0
            fire_risk = min(1.0, base_crop * 0.88)

    elif scenario == "wildfire_risk":
        base_fire = min(1.0, (0.55 * frame.plant_stress + 0.45 * max(0.0, (40.0-frame.humidity_pct)/40.0)) * severity)
        crop_stress = min(1.0, frame.plant_stress * severity)
        if action.name == "mist_and_drones":
            crop_stress = crop_stress * 0.55
            fire_risk = base_fire * 0.10
            water_loss = 180.0 * severity
            energy = 1.35
            disruption = 1.0
        elif action.name == "mist_only":
            crop_stress = crop_stress * 0.72
            fire_risk = base_fire * 0.48
            water_loss = 145.0 * severity
            energy = 0.95
        else:
            fire_risk = base_fire
            safety = min(1.0, 0.45 + 0.35 * severity)

    if communication_loss and action.name in {
        "divert_to_treatment",
        "switch_to_backup",
        "isolate_zone",
    }:
        # Edge-safe fallback remains possible but incurs delay.
        disruption += 4.0
        safety += 0.06

    safety = max(0.0, min(1.0, safety + noise))
    fire_risk = max(0.0, min(1.0, fire_risk + noise / 2.0))

    return SimulationResult(
        action=action.name,
        water_loss_l=round(max(0.0, water_loss), 3),
        energy_kwh=round(max(0.0, energy), 3),
        safety_risk=round(safety, 4),
        service_disruption_min=round(disruption, 3),
        crop_stress=round(max(0.0, crop_stress), 4),
        contamination_exposure=round(max(0.0, exposure), 3),
        fire_risk=round(fire_risk, 4),
    )
