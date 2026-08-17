"""
integration/edge_to_mec_pipeline.py
====================================
Wires the three new layers together with the existing simulator,
end to end, per window:

    SensorWindow (models/sensor_events.py)
        |
        v
    ADN-AE.push_sensor_window()      -- middleware/onem2m.py
    ADN-AE.push_reflex_event()          (Standardization & Middleware
        |                                Layer: oneM2M ADN-AE -> MN-CSE)
        v
    URLLCLink.transmit()              -- network/urllc_link.py
        |                                (5G URLLC, sub-10ms target)
        v
    BHSCognitiveMecApp.process()      -- mec/mec_platform.py +
        |                                mec/bhs_cognitive_engine.py
        |                                (MEC Compute Layer: Bat/Hermit
        |                                 Crab/Squid, ETSI ISG MEC app)
        v
    models/bdo_skin.py's existing (layer_name, [Task,...]) generator
        |                                (UNCHANGED -- this integration
        |                                 adds a standards-compliant
        |                                 front end, it does not touch
        |                                 the TPU-side execution model)
        v
    HexaTPUSimulator.run()            -- simulator.py (existing)

This module does not change models/bdo_skin.py's own MAC-cost
generation, and does not change anything under Scheduler/DMA/AXI/
Cache/Worker/PowerModel: those already-validated components are reused
exactly as-is (same principle the BDO-SKIN integration itself
documents in README.md -- "a new workload/stimulus feeding the
existing engine, not a second simulator"). What this module adds is
the standardized transport and edge-decision layer *in front of* that
existing pipeline, plus telemetry (oneM2M audit trail, URLLC
latency/SLA stats, MEC cognitive-engine decisions) that has no
equivalent in the TPU-only path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import Config
from simulator import HexaTPUSimulator
from models.bdo_skin import build_bdo_skin_workload
from models.sensor_events import generate_scenario, SensorWindow

from middleware.onem2m import MN_CSE, ADN_AE
from middleware.onem2m_http import HttpCSEClient
from network.urllc_link import URLLCLink
from mec.mec_platform import MecPlatform, BHSCognitiveMecApp
from mec.bhs_cognitive_engine import CognitiveResult


REFLEX_TRIGGER_SEVERITY = 0.5   # local edge-reflex threshold, independent of is_critical


@dataclass
class WindowTelemetry:
    window_index: int
    onem2m_sensor_rsc: int
    onem2m_reflex_rsc: int
    urllc_latency_ms: float
    urllc_sla_met: bool
    urllc_retransmitted: bool
    mec_compute_ms: float
    cognitive: CognitiveResult


@dataclass
class PipelineReport:
    scenario: str
    num_windows: int
    telemetry: List[WindowTelemetry]
    onem2m_request_log_len: int
    urllc_stats: dict
    tpu_report: Any                 # simulator.py's SimulationReport, unmodified
    tpu_layers: list
    critical_window_indices: List[int]

    def urllc_sla_violations(self) -> List[int]:
        return [t.window_index for t in self.telemetry if not t.urllc_sla_met]

    def mec_vetoes(self) -> Dict[int, List[str]]:
        return {t.window_index: t.cognitive.stability.vetoed_actions
                for t in self.telemetry if t.cognitive.stability.vetoed_actions}


class EdgeToMecPipeline:
    """One instance = one ADN-AE edge node, its MN-CSE, its URLLC
    link, and its MEC-hosted BHS Cognitive Engine, all wired to a
    single HEXA-TPU-RT simulator run."""

    def __init__(self, cfg: Optional[Config] = None,
                 edge_node_name: str = "hexa-tpu-edge-01",
                 urllc_link: Optional[URLLCLink] = None,
                 seed: int = 0,
                 cse_http_url: Optional[str] = None,
                 cse_id: str = "id-in"):
        """
        cse_http_url: if given (e.g. "http://127.0.0.1:8080"), the
        pipeline registers against a REAL, running oneM2M CSE
        (ACME/tinyIoT/Mobius) over HTTP via
        middleware/onem2m_http.py's HttpCSEClient -- the interoperable
        mode the ESTIMED hackathon evaluates. If omitted, falls back
        to middleware/onem2m.py's in-memory MN_CSE simulation, useful
        for fast offline unit testing without a CSE process running.
        """
        self.cfg = cfg or Config()

        if cse_http_url:
            self.cse = HttpCSEClient(base_url=cse_http_url, cse_id=cse_id)
            if not self.cse.ping():
                raise ConnectionError(
                    f"No oneM2M CSE reachable at {cse_http_url}. Start one first, e.g.:\n"
                    f"  pip install acmecse\n"
                    f"  python -m acmecse --headless --no-coap --no-mqtt --no-ws "
                    f"--no-remote-cse --http-port 8080 -dir <runtime-dir>"
                )
        else:
            self.cse = MN_CSE()

        self.ae = ADN_AE(edge_node_name, self.cse)
        self.ae.register()

        self.link = urllc_link or URLLCLink(seed=seed)

        self.platform = MecPlatform()
        self.mec_app = BHSCognitiveMecApp(self.platform).deploy()

        self._strain_history: List[float] = []

    def _run_window_through_standards_stack(self, w: SensorWindow) -> WindowTelemetry:
        # 1) Standardization & Middleware Layer: encapsulate + register
        #    with the MN-CSE via the ADN-AE.
        sensor_resp = self.ae.push_sensor_window(w)

        reflex_triggered = w.anomaly_active and w.anomaly_severity >= REFLEX_TRIGGER_SEVERITY
        reflex_resp = self.ae.push_reflex_event(
            window_index=w.index, channels_processed=600, triggered=reflex_triggered,
            detail={"anomaly_severity": w.anomaly_severity, "is_critical": w.is_critical},
        )

        # 2) Network layer: transmit the encapsulated sensor envelope
        #    to the MEC server over the 5G URLLC link.
        tx = self.link.transmit(sensor_resp.content)

        # 3) MEC Compute Layer: Bat/Hermit-Crab/Squid cognitive
        #    evaluation of the delivered window.
        self._strain_history.append(w.mean_strain_kpa)
        record = self.mec_app.process(
            window_index=w.index,
            strain_history_kpa=list(self._strain_history),
            mean_temp_c=w.mean_temp_c,
            anomaly_severity=w.anomaly_severity,
        )

        return WindowTelemetry(
            window_index=w.index,
            onem2m_sensor_rsc=int(sensor_resp.response_status_code),
            onem2m_reflex_rsc=int(reflex_resp.response_status_code),
            urllc_latency_ms=tx.latency_ms,
            urllc_sla_met=tx.sla_met,
            urllc_retransmitted=tx.retransmitted,
            mec_compute_ms=record.compute_time_ms,
            cognitive=record.result,
        )

    def run(self, scenario: str = "burst_anomaly", num_windows: int = 150,
            seed: int = 0, num_workers: Optional[int] = None) -> PipelineReport:
        num_workers = num_workers or self.cfg.NUM_WORKERS
        windows = generate_scenario(scenario, num_windows, seed=seed)

        telemetry = [self._run_window_through_standards_stack(w) for w in windows]

        # The TPU-side execution path is untouched: the existing
        # bdo_skin.py generator still produces the (layer_name, [Task])
        # workload the Scheduler/DMA/AXI/Cache/Worker/PowerModel chain
        # consumes -- this integration's contribution is everything
        # that happened above (standardized transport + real edge
        # decisioning), not a replacement for the TPU compute model.
        layers, meta = build_bdo_skin_workload(num_workers=num_workers, scenario=scenario,
                                                num_windows=num_windows, seed=seed)
        sim = HexaTPUSimulator(self.cfg)
        tpu_report = sim.run(layers)

        return PipelineReport(
            scenario=scenario, num_windows=num_windows, telemetry=telemetry,
            onem2m_request_log_len=len(self.cse.request_log),
            urllc_stats=self.link.stats(),
            tpu_report=tpu_report, tpu_layers=layers,
            critical_window_indices=meta["critical_window_indices"],
        )
