from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from aquaguardian.comparison import aggregate_comparisons, compare_scenario
from aquaguardian.engine import ClosedLoopEngine
from aquaguardian.scenarios import SCENARIOS as ENGINE_SCENARIOS
from . import hardware_bridge
from .data import SCENARIOS as CONSOLE_SCENARIOS, STAGES

app = FastAPI(
    title="AquaGuardian AI — Engineering Validation Console",
    version="3.0.0-gpiw",
    description="Closed-loop simulation, stress testing, comparison and validation PoC.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "AQUAGUARDIAN_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)

STEP_DELAY = 0.65
FAIL_PAUSE = 0.8

# Maps the visual console scenarios to the deterministic engineering engine.
ENGINE_MAP = {
    "leak": "pipeline_leak",
    "pump": "pump_degradation",
    "drought": "drought_stress",
    "fire": "wildfire_risk",
}



def build_console_metrics(comparison: dict | None, decision: dict | None) -> list[dict]:
    """Create UI metrics only from engine outputs; never from presentation constants."""
    if not comparison or not decision:
        return []
    by_name = {row["strategy"]: row for row in comparison["strategies"]}
    reactive = by_name["reactive_baseline"]
    aqua = by_name["aquaguardian_closed_loop"]
    return [
        {"label": "Modeled water-loss reduction", "value": comparison["improvements"]["water_loss_reduction_pct_vs_reactive"], "unit": "%", "max": 100},
        {"label": "Stress-test pass rate", "value": round(aqua["stress_pass_rate"] * 100, 1), "unit": "%", "max": 100},
        {"label": "Decision reliability", "value": round(decision["reliability"] * 100, 1), "unit": "%", "max": 100},
        {"label": "Reactive delay avoided", "value": round(reactive["detection_delay_s"] - aqua["detection_delay_s"], 2), "unit": "s", "max": reactive["detection_delay_s"]},
    ]

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aquaguardian-integrated",
        "engine_scenarios": len(ENGINE_SCENARIOS),
        "console_scenarios": len(CONSOLE_SCENARIOS),
        "simulation_claim": "deterministic PoC; not field-trial data",
    }


@app.get("/api/hardware/reflex")
def hardware_reflex_benchmark(seed: int = 0) -> dict:
    """
    Runs hexa_tpu_rt_sim's water-domain reflex kernel, URLLC network
    model, and MEC cognitive engine live and returns the results.
    This is a real (if simplified/simulated) computation each call,
    not a cached or hand-written constant.
    """
    try:
        return hardware_bridge.run_water_hardware_benchmark(seed=seed)
    except Exception as exc:  # pragma: no cover - defensive, see module docstring
        raise HTTPException(
            status_code=503,
            detail=f"hexa_tpu_rt_sim hardware simulation unavailable: {exc}",
        )


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    return [
        {
            "id": scenario["id"],
            "kind": scenario["kind"],
            "tag": scenario["tag"],
            "name": scenario["name"],
            "loc": scenario["loc"],
            "twinFocus": scenario["twinFocus"],
            "engineScenario": ENGINE_MAP.get(scenario["id"]),
        }
        for scenario in CONSOLE_SCENARIOS.values()
    ]


@app.get("/api/engine/scenarios")
def list_engine_scenarios() -> dict:
    return {name: frame.to_dict() for name, frame in ENGINE_SCENARIOS.items()}


@app.get("/api/engine/decision/{scenario_id}")
def engine_decision(scenario_id: str) -> dict:
    if scenario_id not in ENGINE_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown engine scenario: {scenario_id}")
    decision = ClosedLoopEngine().decide(ENGINE_SCENARIOS[scenario_id], scenario_id)
    return decision.to_dict()


@app.get("/api/comparison/{scenario_id}")
def scenario_comparison(scenario_id: str) -> dict:
    if scenario_id not in ENGINE_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown engine scenario: {scenario_id}")
    return compare_scenario(scenario_id, ENGINE_SCENARIOS[scenario_id])


@app.get("/api/comparison")
def all_comparisons() -> dict:
    comparisons = [compare_scenario(name, frame) for name, frame in ENGINE_SCENARIOS.items()]
    return {
        "comparisons": comparisons,
        "aggregate": aggregate_comparisons(comparisons),
    }


@app.get("/api/evidence/{scenario_id}")
def scenario_evidence(scenario_id: str) -> dict:
    scenario = CONSOLE_SCENARIOS.get(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Unknown console scenario: {scenario_id}")
    engine_id = ENGINE_MAP.get(scenario_id)
    if not engine_id:
        return {"consoleScenario": scenario_id, "engineScenario": None, "status": "narrative_only", "model_note": "No water-engine mapping for this heritage scenario."}
    decision = ClosedLoopEngine().decide(ENGINE_SCENARIOS[engine_id], engine_id).to_dict()
    comparison = compare_scenario(engine_id, ENGINE_SCENARIOS[engine_id])
    return {
        "consoleScenario": scenario_id,
        "engineScenario": engine_id,
        "decision": decision,
        "comparison": comparison,
        "metrics": build_console_metrics(comparison, decision),
        "model_note": "Deterministic PoC simulation evidence; not field-trial data.",
    }


@app.get("/api/run/{scenario_id}")
async def run_scenario(scenario_id: str):
    """Stream the visible closed-loop cycle and attach real engine evidence."""
    scenario = CONSOLE_SCENARIOS.get(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Unknown console scenario: {scenario_id}")

    engine_id = ENGINE_MAP.get(scenario_id)
    engine_result = None
    comparison = None
    if engine_id:
        engine_result = ClosedLoopEngine().decide(ENGINE_SCENARIOS[engine_id], engine_id).to_dict()
        comparison = compare_scenario(engine_id, ENGINE_SCENARIOS[engine_id])

    async def gen():
        yield sse("engine_evidence", {
            "engineScenario": engine_id,
            "decision": engine_result,
            "comparison": comparison,
            "note": "Narrative console telemetry is illustrative; engineering metrics are generated by the deterministic PoC engine.",
        })

        for stage in STAGES:
            sid = stage["id"]
            detail = scenario["stages"][sid]
            yield sse("stage_start", {"stage": sid, **detail})
            await asyncio.sleep(STEP_DELAY)

            if sid == "stress_test" and scenario["stressFails"]:
                yield sse("stress_fail", {"stage": sid})
                await asyncio.sleep(FAIL_PAUSE)
                yield sse("reoptimize_start", {"stage": "optimize", **scenario["stages"]["optimize"]})
                await asyncio.sleep(STEP_DELAY)
                yield sse("reoptimize_done", {"stage": "optimize"})
                yield sse("stress_retry_start", {"stage": "stress_test", **scenario["stages"]["stress_retry"]})
                await asyncio.sleep(STEP_DELAY)

            yield sse("stage_done", {"stage": sid})

        yield sse("complete", {
            "metrics": build_console_metrics(comparison, engine_result),
            "engine": engine_result,
            "comparison": comparison,
            "model_note": "Simulation outputs are PoC results, not field-trial claims.",
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
