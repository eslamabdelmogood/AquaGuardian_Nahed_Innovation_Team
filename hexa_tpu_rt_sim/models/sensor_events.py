"""
models/sensor_events.py
========================
Sensor/event-level model for the BDO-SKIN integration, deliberately
kept independent of HEXA-TPU-RT specifics (no Task, no MAC counts here
-- that translation happens in models/bdo_skin.py). This module only
answers "what is the sensor grid seeing, window by window, under each
scenario" -- the same separation of concerns as a real system, where
the sensing/physics layer doesn't know anything about the chip that
will eventually process it.

Source: Black_Dragon_Optical_Skin_Research_Report.docx (BDO-SKIN),
Section 2.2 -- "600 individual gratings spaced every 2 cells along
fiber lines spaced every 2 rows" over a 60x40 finite-difference mesh.
That gives a 30 (rows) x 20 (cols) grating grid = 600 channels, which
is what this module models. The rest of BDO-SKIN's own physics
(finite-difference heat/stress/fatigue-damage simulation) is NOT
reproduced here -- this generates a plausible sensor-signal envelope
consistent with the report's four fault narratives, not a re-derivation
of its physics engine. Treat the numeric severity curves below as a
stimulus generator for stressing HEXA-TPU-RT's scheduling and memory
pipeline, not as a validated structural model.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

FBG_GRID_ROWS = 20
FBG_GRID_COLS = 30
NUM_FBG_CHANNELS = FBG_GRID_ROWS * FBG_GRID_COLS  # 600, matches the report


@dataclass
class SensorWindow:
    """One reporting window's worth of sensor state across all 600
    channels, summarized (not per-channel, to keep this module cheap
    to generate) into the fields the workload generator needs."""
    index: int
    mean_temp_c: float             # baseline ~25C ambient
    mean_strain_kpa: float
    anomaly_active: bool
    anomaly_severity: float        # 0..1, normalized proximity-to-failure
    anomaly_cell: Optional[Tuple[int, int]]   # (row, col) in the 20x30 grating grid
    is_critical: bool              # True => emergency/reflex path triggers
    scenario: str


def _baseline_noise(rng, mean_temp=25.0, mean_strain=0.0):
    return (mean_temp + rng.gauss(0, 0.15), mean_strain + rng.gauss(0, 150.0))


def generate_normal(num_windows: int, seed: int = 0) -> List[SensorWindow]:
    """Quiet baseline -- sensor noise only, no anomaly ever. Used both
    as its own scenario and as the lead-in period for the other three."""
    rng = random.Random(seed)
    windows = []
    for i in range(num_windows):
        temp, strain = _baseline_noise(rng)
        windows.append(SensorWindow(i, temp, strain, False, 0.0, None, False, "normal"))
    return windows


def generate_gradual_anomaly(num_windows: int, onset_window: int = 40,
                              ramp_windows: int = 80, seed: int = 1,
                              anomaly_cell: Tuple[int, int] = (13, 21)) -> List[SensorWindow]:
    """Slow-ramping localized fault (report Scenario A/B narrative:
    thermal fault or crack growth) -- severity climbs linearly from
    onset, crossing the critical threshold only once it's had time to
    develop. Tests whether the Bat-style forecast layer gets enough
    lead time to matter before things become urgent."""
    rng = random.Random(seed)
    windows = []
    for i in range(num_windows):
        temp, strain = _baseline_noise(rng)
        if i < onset_window:
            windows.append(SensorWindow(i, temp, strain, False, 0.0, None, False, "gradual_anomaly"))
            continue
        progress = min(1.0, (i - onset_window) / ramp_windows)
        severity = progress ** 1.3  # slightly convex -- accelerating late-stage growth
        temp += severity * 45.0
        strain += severity * 8000.0
        critical = severity >= 0.85
        windows.append(SensorWindow(i, temp, strain, True, severity, anomaly_cell,
                                     critical, "gradual_anomaly"))
    return windows


def generate_burst_anomaly(num_windows: int, onset_window: int = 60, seed: int = 2,
                            anomaly_cell: Tuple[int, int] = (10, 15),
                            burst_duration: int = 6) -> List[SensorWindow]:
    """Sudden localized spike (report Scenario C narrative: mechanical
    overload) -- near-instant jump to high severity, sustained briefly,
    then partial decay. This is the scenario built specifically to
    stress-test the sudden-burst-of-critical-work question."""
    rng = random.Random(seed)
    windows = []
    for i in range(num_windows):
        temp, strain = _baseline_noise(rng)
        if i < onset_window:
            windows.append(SensorWindow(i, temp, strain, False, 0.0, None, False, "burst_anomaly"))
            continue
        steps_since_onset = i - onset_window
        if steps_since_onset < burst_duration:
            severity = 0.95  # near-instant jump, sustained through the burst
        else:
            decay_steps = steps_since_onset - burst_duration
            severity = max(0.35, 0.95 - decay_steps * 0.03)  # settles to an elevated plateau
        temp += severity * 20.0        # overload is mostly mechanical, modest thermal signature
        strain += severity * 22000.0   # but a sharp stress spike
        critical = steps_since_onset < burst_duration + 4  # critical during and just after the spike
        windows.append(SensorWindow(i, temp, strain, True, severity, anomaly_cell,
                                     critical, "burst_anomaly"))
    return windows


def generate_critical_event(num_windows: int, onset_window: int = 50, seed: int = 3,
                             anomaly_cells: Tuple[Tuple[int, int], ...] = ((13, 11), (14, 12)),
                             sustained_duration: int = 40) -> List[SensorWindow]:
    """Worst case (report Scenario D narrative: combined heat + vibration
    + stress) -- multiple overlapping fault signatures, sustained at
    high severity for an extended period rather than a brief spike.
    This is the scenario that tests sustained (not just momentary)
    emergency-path load."""
    rng = random.Random(seed)
    windows = []
    for i in range(num_windows):
        temp, strain = _baseline_noise(rng)
        if i < onset_window:
            windows.append(SensorWindow(i, temp, strain, False, 0.0, None, False, "critical_event"))
            continue
        steps_since_onset = i - onset_window
        if steps_since_onset < sustained_duration:
            severity = min(1.0, 0.7 + 0.01 * steps_since_onset)  # ramps up and stays high
        else:
            severity = max(0.4, 1.0 - (steps_since_onset - sustained_duration) * 0.02)
        temp += severity * 55.0
        strain += severity * 26000.0
        critical = severity >= 0.75
        # Alternate which of the multiple simultaneous anomaly cells is
        # "peak" this window, modeling multi-site fault attribution.
        cell = anomaly_cells[i % len(anomaly_cells)]
        windows.append(SensorWindow(i, temp, strain, True, severity, cell,
                                     critical, "critical_event"))
    return windows


SCENARIOS = {
    "normal": generate_normal,
    "gradual_anomaly": generate_gradual_anomaly,
    "burst_anomaly": generate_burst_anomaly,
    "critical_event": generate_critical_event,
}


def generate_scenario(scenario: str, num_windows: int, seed: int = 0) -> List[SensorWindow]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}.")
    return SCENARIOS[scenario](num_windows, seed=seed) if scenario == "normal" \
        else SCENARIOS[scenario](num_windows, seed=seed)


def critical_window_indices(windows: List[SensorWindow]) -> List[int]:
    return [w.index for w in windows if w.is_critical]
