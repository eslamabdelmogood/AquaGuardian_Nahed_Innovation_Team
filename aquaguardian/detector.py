from __future__ import annotations
from .models import SensorFrame


def analyze(frame: SensorFrame) -> dict:
    leak = max(
        0.0,
        min(
            1.0,
            0.55 * max(0.0, (12.0 - frame.pressure_bar) / 4.0)
            + 0.45 * max(0.0, (35.0 - frame.flow_lps) / 20.0),
        ),
    )
    pump = max(0.0, min(1.0, (frame.pump_vibration_mm_s - 4.0) / 8.0))
    contamination = max(0.0, min(1.0, (frame.turbidity_ntu - 1.0) / 9.0))
    drought = max(
        0.0,
        min(
            1.0,
            0.45 * max(0.0, (35.0 - frame.soil_moisture_pct) / 30.0)
            + 0.35 * frame.plant_stress
            + 0.20 * max(0.0, (frame.temperature_c - 30.0) / 18.0),
        ),
    )
    wildfire = max(
        0.0,
        min(
            1.0,
            0.35 * frame.plant_stress
            + 0.35 * max(0.0, (frame.temperature_c - 32.0) / 18.0)
            + 0.30 * max(0.0, (35.0 - frame.humidity_pct) / 30.0),
        ),
    )
    return {
        "pipeline_leak": round(leak, 4),
        "pump_degradation": round(pump, 4),
        "water_contamination": round(contamination, 4),
        "drought_stress": round(drought, 4),
        "wildfire_risk": round(wildfire, 4),
    }
