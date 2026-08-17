from __future__ import annotations
from .models import SensorFrame


SCENARIOS = {
    "pipeline_leak": SensorFrame(
        flow_lps=18.0,
        pressure_bar=7.8,
        pump_vibration_mm_s=4.2,
        turbidity_ntu=0.7,
        soil_moisture_pct=44.0,
        plant_stress=0.20,
        temperature_c=29.0,
        humidity_pct=48.0,
    ),
    "pump_degradation": SensorFrame(
        flow_lps=39.0,
        pressure_bar=11.5,
        pump_vibration_mm_s=10.5,
        turbidity_ntu=0.8,
        soil_moisture_pct=42.0,
        plant_stress=0.18,
        temperature_c=31.0,
        humidity_pct=45.0,
    ),
    "water_contamination": SensorFrame(
        flow_lps=37.0,
        pressure_bar=11.8,
        pump_vibration_mm_s=4.8,
        turbidity_ntu=8.2,
        soil_moisture_pct=41.0,
        plant_stress=0.22,
        temperature_c=30.0,
        humidity_pct=46.0,
    ),
    "drought_stress": SensorFrame(
        flow_lps=35.0,
        pressure_bar=11.9,
        pump_vibration_mm_s=4.1,
        turbidity_ntu=0.9,
        soil_moisture_pct=13.0,
        plant_stress=0.84,
        temperature_c=44.0,
        humidity_pct=17.0,
    ),
    "wildfire_risk": SensorFrame(
        flow_lps=34.0,
        pressure_bar=11.7,
        pump_vibration_mm_s=4.0,
        turbidity_ntu=0.9,
        soil_moisture_pct=16.0,
        plant_stress=0.78,
        temperature_c=46.0,
        humidity_pct=14.0,
    ),
}
