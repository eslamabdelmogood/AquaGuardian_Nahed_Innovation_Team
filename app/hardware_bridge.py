"""
app/hardware_bridge.py
=======================
Bridges the FastAPI console to `hexa_tpu_rt_sim`, an independently
built architecture simulator that models the edge hardware layer
underneath AquaGuardian's "sense locally, decide before the network
round trip" claim: a domain-neutral reflex kernel (sub-1ms deadline),
a 5G URLLC edge->MEC link model, and a bio-inspired MEC cognitive
engine (Bat Forecaster / Squid Controller / Hermit Crab veto) reused
unmodified between the project's aviation and water domain profiles.

Every number this module returns is computed fresh, on request, by
actually running that simulator's water-domain code -- it is not a
constant copied out of hexa_tpu_rt_sim's README. If the simulator
package is missing or its API changes, `run_water_hardware_benchmark`
raises, and the caller (app/main.py) turns that into a clear "hardware
simulation unavailable" response rather than a silent fabricated
number.
"""
from __future__ import annotations

import os
import sys
from statistics import mean

_HEXA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hexa_tpu_rt_sim")
if _HEXA_ROOT not in sys.path:
    sys.path.insert(0, _HEXA_ROOT)


def run_water_hardware_benchmark(seed: int = 0, n_samples: int = 200) -> dict:
    from reflex.reflex_kernel import ReflexKernel
    from reflex.domains.water import (
        WaterPressureTransientTrigger,
        generate_water_scenario,
        sample_to_dict,
    )
    from network.urllc_link import URLLCLink
    from mec.domains.water_cognitive_engine import WaterBHSCognitiveEngine

    # --- Reflex layer: does the sub-1ms local decision actually hold,
    # on a real pipe-burst transient, for this run? ---
    trigger = WaterPressureTransientTrigger()
    kernel = ReflexKernel(trigger, deadline_ms=1.0)
    samples = generate_water_scenario("sudden_burst", n_samples, seed=seed)
    for s in samples:
        kernel.evaluate(sample_to_dict(s), sample_index=s.index)
    reflex_stats = kernel.stats()
    ground_truth_bursts = sum(1 for s in samples if s.is_burst_ground_truth)
    detected_bursts = sum(
        1 for e in kernel.events if e.triggered and samples[e.sample_index].is_burst_ground_truth
    )

    # --- Network layer: what does the oneM2M/URLLC edge->MEC hop cost,
    # measured from this same run, not a spec figure? ---
    link = URLLCLink(seed=seed)
    for _ in range(n_samples):
        link.transmit({"type": "water_pressure_sample"})
    latencies = [t.latency_ms for t in link.transmissions]
    latencies_sorted = sorted(latencies)
    p99_index = max(0, int(len(latencies_sorted) * 0.99) - 1)

    # --- MEC cognitive layer: on a worsening leak, what does the
    # reused Bat/Squid/Hermit-Crab stack actually decide? ---
    engine = WaterBHSCognitiveEngine()
    declining = [600.0 - i * 3.0 for i in range(15)]
    cognitive = engine.evaluate(pressure_history_kpa=declining, leak_severity=0.8)

    # --- oneM2M layer: does the edge node's reflex event actually reach
    # a standard-shaped resource tree, round-trip, via real CRUD
    # primitives (CREATE/RETRIEVE), not just get logged locally? ---
    from middleware.onem2m import MN_CSE, ADN_AE, RequestPrimitive, Operation

    cse = MN_CSE(cse_id="mn-cse-aquaguardian-demo")
    ae = ADN_AE("aquaguardian-water-edge-01", cse, app_type="water-pressure-edge")
    register_resp = ae.register()

    pushed = 0
    for e in kernel.events:
        if e.triggered:
            resp = ae.push_reflex_event(
                window_index=e.sample_index,
                channels_processed=1,
                triggered=True,
                detail={"scenario": "sudden_burst", "trigger": "water_pressure_transient"},
            )
            if resp.ok:
                pushed += 1

    latest = cse.latest_content_instances(ae.app_name, ADN_AE.REFLEX_CONTAINER, limit=3)
    retrieve_resp = cse.handle_primitive(RequestPrimitive(
        operation=Operation.RETRIEVE, to=f"{ae.app_name}/{ADN_AE.REFLEX_CONTAINER}",
        fr=ae.ae_id,
    ))

    return {
        "source": "hexa_tpu_rt_sim (bundled, executed live)",
        "reflex_layer": {
            "domain": "water — pressure transient / water-hammer trigger",
            "samples": reflex_stats.count,
            "deadline_ms": kernel.deadline_ms,
            "max_latency_ms": round(reflex_stats.max_latency_ms, 6),
            "mean_latency_ms": round(reflex_stats.mean_latency_ms, 6),
            "deadline_margin_ms": round(reflex_stats.max_latency_margin_ms, 6),
            "deadline_misses": reflex_stats.deadline_misses,
            "triggered_count": reflex_stats.triggered_count,
            "ground_truth_bursts": ground_truth_bursts,
            "detected_bursts": detected_bursts,
        },
        "network_layer": {
            "domain": "edge -> MEC, modeled as 5G NR URLLC",
            "samples": len(latencies),
            "mean_latency_ms": round(mean(latencies), 4),
            "p99_latency_ms": round(latencies_sorted[p99_index], 4),
            "sla_target_ms": link.sla_target_ms,
            "sla_met_rate": round(sum(1 for t in link.transmissions if t.sla_met) / len(link.transmissions), 4),
        },
        "reflex_vs_network_ratio": round(
            (link.sla_target_ms) / max(reflex_stats.max_latency_ms, 1e-9), 1
        ),
        "mec_cognitive_layer": {
            "domain": "worsening leak, severity=0.8 (Bat Forecaster + Squid Controller + Water-Hammer Hermit Crab)",
            "remaining_useful_life_windows": cognitive.rul.remaining_useful_life_windows,
            "trend_slope_kpa_per_window": round(cognitive.rul.trend_slope_kpa_per_window, 4),
            "vetoed_actions": cognitive.stability.vetoed_actions,
            "chosen_action": cognitive.control.chosen_action,
            "bat_forecaster_class": type(engine.bat).__module__ + "." + type(engine.bat).__name__,
            "squid_controller_class": type(engine.squid).__module__ + "." + type(engine.squid).__name__,
            "note": "Same BatForecaster/SquidController classes as the project's aviation domain — not reimplementations.",
        },
        "onem2m_layer": {
            "domain": "standardization/middleware, water edge node (TS-0001/TS-0004 shapes)",
            "ae_registration_ok": register_resp.ok,
            "ae_registration_status_code": int(register_resp.response_status_code),
            "assigned_ae_id": ae.ae_id,
            "reflex_events_pushed_via_create": pushed,
            "resource_tree_path": f"{ae.app_name}/{ADN_AE.REFLEX_CONTAINER}",
            "retrieve_ok": retrieve_resp.ok,
            "retrieve_status_code": int(retrieve_resp.response_status_code),
            "latest_content_instances": latest,
            "cse_request_log_length": len(cse.request_log),
            "note": "MN_CSE.handle_primitive and ADN_AE.register/push_reflex_event run for real against an in-memory oneM2M resource tree: CREATE for AE + container registration, CREATE for each pushed reflex event, and an explicit RETRIEVE primitive (logged, status above) against the container. 'latest_content_instances' itself uses a convenience accessor (documented in hexa_tpu_rt_sim as the RETRIEVE-with-filterCriteria equivalent) rather than a second logged primitive. No real socket/wire protocol — this is the architecture-layer simulation hexa_tpu_rt_sim's own README describes — but the resource tree, addressing, and CRUD semantics follow the standard.",
        },
        "model_note": (
            "hexa_tpu_rt_sim is a separate, independently developed architecture simulator "
            "bundled in this repo (see /hexa_tpu_rt_sim). Its own README states these are "
            "cycle-accurate deadline checks and a statistical network model, not measurements "
            "from real embedded hardware or a real radio."
        ),
    }
