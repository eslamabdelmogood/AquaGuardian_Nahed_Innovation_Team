from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Callable

from .baselines import detection_only_decision, reactive_decision
from .digital_twin import simulate
from .engine import ClosedLoopEngine
from .models import CandidateAction, SensorFrame
from .policy import validate

STRESS_PROFILES = [
    {"name": "nominal", "severity": 1.0, "sensor_noise": 0.00, "communication_loss": False},
    {"name": "moderate", "severity": 1.20, "sensor_noise": 0.03, "communication_loss": False},
    {"name": "severe_offline", "severity": 1.45, "sensor_noise": 0.06, "communication_loss": True},
    {"name": "extreme", "severity": 1.70, "sensor_noise": 0.10, "communication_loss": True},
]


def _evaluate_action(
    scenario: str,
    frame: SensorFrame,
    action: CandidateAction,
    *,
    strategy: str,
    detection_delay_s: float,
) -> dict:
    runs = []
    for index, profile in enumerate(STRESS_PROFILES):
        result = simulate(
            scenario,
            frame,
            action,
            severity=profile["severity"],
            sensor_noise=profile["sensor_noise"],
            communication_loss=profile["communication_loss"],
            seed=7000 + index,
        )

        # A delayed reactive response allows additional loss/exposure/risk to
        # accumulate before the selected action takes effect.
        delay_factor = detection_delay_s / 300.0
        result.water_loss_l = round(result.water_loss_l + 260.0 * delay_factor, 3)
        result.contamination_exposure = round(
            result.contamination_exposure + 22.0 * delay_factor, 3
        )
        result.crop_stress = round(min(1.0, result.crop_stress + 0.10 * delay_factor), 4)
        result.fire_risk = round(min(1.0, result.fire_risk + 0.12 * delay_factor), 4)
        result.service_disruption_min = round(
            result.service_disruption_min + detection_delay_s / 60.0, 3
        )

        passed, reasons = validate(result)
        result.passed = passed
        result.reasons = reasons
        run = result.to_dict()
        run["profile"] = profile["name"]
        runs.append(run)

    return {
        "strategy": strategy,
        "selected_action": action.name,
        "detection_delay_s": detection_delay_s,
        "stress_pass_rate": round(sum(run["passed"] for run in runs) / len(runs), 4),
        "mean_water_loss_l": round(mean(run["water_loss_l"] for run in runs), 3),
        "mean_energy_kwh": round(mean(run["energy_kwh"] for run in runs), 3),
        "mean_service_disruption_min": round(
            mean(run["service_disruption_min"] for run in runs), 3
        ),
        "mean_safety_risk": round(mean(run["safety_risk"] for run in runs), 4),
        "mean_contamination_exposure": round(
            mean(run["contamination_exposure"] for run in runs), 3
        ),
        "mean_crop_stress": round(mean(run["crop_stress"] for run in runs), 4),
        "mean_fire_risk": round(mean(run["fire_risk"] for run in runs), 4),
        "stress_runs": runs,
    }


def compare_scenario(scenario: str, frame: SensorFrame) -> dict:
    reactive = reactive_decision(frame, scenario)
    detection = detection_only_decision(frame, scenario)
    closed_loop = ClosedLoopEngine().decide(frame, scenario)
    closed_action = CandidateAction(closed_loop.selected_action, {})

    results = [
        _evaluate_action(
            scenario,
            frame,
            reactive.action,
            strategy=reactive.strategy,
            detection_delay_s=reactive.detection_delay_s,
        ),
        _evaluate_action(
            scenario,
            frame,
            detection.action,
            strategy=detection.strategy,
            detection_delay_s=detection.detection_delay_s,
        ),
        _evaluate_action(
            scenario,
            frame,
            closed_action,
            strategy="aquaguardian_closed_loop",
            detection_delay_s=0.25,
        ),
    ]

    by_name = {item["strategy"]: item for item in results}
    baseline = by_name["reactive_baseline"]
    aqua = by_name["aquaguardian_closed_loop"]
    improvements = {
        "water_loss_reduction_pct_vs_reactive": round(
            100.0 * (baseline["mean_water_loss_l"] - aqua["mean_water_loss_l"])
            / max(baseline["mean_water_loss_l"], 1e-9),
            2,
        ),
        "response_time_reduction_pct_vs_reactive": round(
            100.0 * (baseline["detection_delay_s"] - aqua["detection_delay_s"])
            / baseline["detection_delay_s"],
            2,
        ),
        "stress_pass_rate_gain_vs_reactive": round(
            aqua["stress_pass_rate"] - baseline["stress_pass_rate"], 4
        ),
        "safety_risk_reduction_pct_vs_reactive": round(
            100.0 * (baseline["mean_safety_risk"] - aqua["mean_safety_risk"])
            / max(baseline["mean_safety_risk"], 1e-9),
            2,
        ),
    }

    return {
        "scenario": scenario,
        "model_note": (
            "All values are deterministic PoC simulation outputs, not field-trial claims. "
            "The reactive baseline includes a modeled five-minute response delay."
        ),
        "strategies": results,
        "improvements": improvements,
    }


def aggregate_comparisons(comparisons: list[dict]) -> dict:
    strategy_names = [
        "reactive_baseline",
        "detection_only_ai",
        "aquaguardian_closed_loop",
    ]
    summary = {}
    for strategy in strategy_names:
        rows = [
            next(item for item in comparison["strategies"] if item["strategy"] == strategy)
            for comparison in comparisons
        ]
        summary[strategy] = {
            "average_stress_pass_rate": round(mean(row["stress_pass_rate"] for row in rows), 4),
            "average_water_loss_l": round(mean(row["mean_water_loss_l"] for row in rows), 3),
            "average_safety_risk": round(mean(row["mean_safety_risk"] for row in rows), 4),
            "average_service_disruption_min": round(
                mean(row["mean_service_disruption_min"] for row in rows), 3
            ),
            "average_response_delay_s": round(mean(row["detection_delay_s"] for row in rows), 3),
        }
    return {
        "model_note": "Comparative outputs are generated by the included deterministic PoC simulator.",
        "scenario_count": len(comparisons),
        "stress_profiles_per_strategy": len(STRESS_PROFILES),
        "strategies": summary,
    }
