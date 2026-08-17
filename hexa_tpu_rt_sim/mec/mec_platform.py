"""
mec/mec_platform.py
====================
MEC Compute Layer, modeled on the ETSI ISG MEC architecture (GS MEC
003 Framework and Reference Architecture): a MEC host running a MEC
platform that provides a service registry, plus one or more MEC
applications registered against it. This module implements the
platform-level plumbing; mec/bhs_cognitive_engine.py implements the
one application-level workload this project needs (BHSCognitiveEngine).

Mapping onto GS MEC 003's terms:

  MecHost      -- the physical/virtual MEC compute node at the network
                  edge (GS MEC 003 section 6.2.2's "Mobile Edge Host").
  MecPlatform  -- hosts the Mp1 service registry MEC apps use to
                  discover/advertise services (GS MEC 003 section
                  6.2.4). Simplified here to registration + lookup by
                  service name, since this project has exactly one
                  producer/consumer pair (edge -> BHS engine).
  MecApp       -- a MEC application instance (GS MEC 003 section
                  6.2.3) with the lifecycle states the standard
                  defines: INSTANTIATED -> ACTIVE -> TERMINATED.

Not modeled (deliberately out of scope for a single-app simulator):
Mp2 traffic-rules/DNS APIs, multi-host MEC-to-MEC orchestration
(Mm3), and the ETSI NFV MANO stack GS MEC 003 assumes underneath --
those are deployment/orchestration concerns, not the
compute-and-decide path this integration exercises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from mec.bhs_cognitive_engine import BHSCognitiveEngine, CognitiveResult


class MecAppState(Enum):
    INSTANTIATED = "instantiated"
    ACTIVE = "active"
    TERMINATED = "terminated"


@dataclass
class MecServiceRecord:
    """One Mp1 service-registry entry (GS MEC 003 section 6.2.4 /
    MEC 011 service registration)."""
    service_name: str
    app_instance_id: str
    version: str = "v1"
    state: str = "ACTIVE"


class MecApp:
    """Base MEC application: standard instantiate/activate/terminate
    lifecycle plus Mp1 service (de)registration against a MecPlatform."""

    def __init__(self, app_name: str, platform: "MecPlatform"):
        self.app_name = app_name
        self.app_instance_id = f"mec-app-{app_name}-{id(self):x}"
        self.platform = platform
        self.state = MecAppState.INSTANTIATED

    def activate(self, provides: Optional[List[str]] = None):
        self.state = MecAppState.ACTIVE
        for svc in (provides or []):
            self.platform.register_service(svc, self.app_instance_id)

    def terminate(self):
        self.platform.deregister_app_services(self.app_instance_id)
        self.state = MecAppState.TERMINATED


class MecPlatform:
    """Mp1 service registry + a small dispatch surface for this
    project's single deployed application."""

    def __init__(self, platform_id: str = "mec-platform-01"):
        self.platform_id = platform_id
        self._services: Dict[str, MecServiceRecord] = {}
        self._apps: Dict[str, MecApp] = {}

    def register_service(self, service_name: str, app_instance_id: str):
        self._services[service_name] = MecServiceRecord(service_name, app_instance_id)

    def deregister_app_services(self, app_instance_id: str):
        stale = [name for name, rec in self._services.items()
                 if rec.app_instance_id == app_instance_id]
        for name in stale:
            del self._services[name]

    def discover(self, service_name: str) -> Optional[MecServiceRecord]:
        return self._services.get(service_name)

    def deploy(self, app: MecApp, provides: Optional[List[str]] = None):
        self._apps[app.app_instance_id] = app
        app.activate(provides=provides)
        return app


@dataclass
class MecInvocationRecord:
    window_index: int
    receive_time_s: float
    compute_time_ms: float
    result: CognitiveResult


class BHSCognitiveMecApp(MecApp):
    """MEC Application Level deployment of the BHS Cognitive Engine.
    Registers three Mp1 services -- one per cognitive role, matching
    the requested architecture's own component names -- even though
    all three currently run in-process within one app instance (real
    ETSI MEC deployments commonly co-locate tightly-coupled services
    inside one app while still advertising them separately for
    discovery/versioning purposes)."""

    SERVICE_BAT_FORECASTER = "bat-forecaster"
    SERVICE_HERMIT_CRAB_EVALUATOR = "hermit-crab-evaluator"
    SERVICE_SQUID_CONTROLLER = "squid-controller"

    def __init__(self, platform: "MecPlatform", engine: Optional[BHSCognitiveEngine] = None):
        super().__init__("bhs-cognitive-engine", platform)
        self.engine = engine or BHSCognitiveEngine()
        self.invocations: List[MecInvocationRecord] = []

    def deploy(self):
        self.platform.deploy(self, provides=[
            self.SERVICE_BAT_FORECASTER,
            self.SERVICE_HERMIT_CRAB_EVALUATOR,
            self.SERVICE_SQUID_CONTROLLER,
        ])
        return self

    def process(self, window_index: int, strain_history_kpa: List[float],
                mean_temp_c: float, anomaly_severity: float,
                compute_time_ms_fn: Optional[Callable[[], float]] = None
                ) -> MecInvocationRecord:
        """Runs one sensor window through the deployed cognitive
        engine. `compute_time_ms_fn`, if given, lets a caller measure
        wall-clock compute cost (e.g. for a benchmark); otherwise a
        nominal fixed budget is reported (this platform-layer module
        does not itself model MEC-host compute contention -- that is
        the existing TPU-side simulator's job once the resulting
        decision is turned back into scheduled work)."""
        if self.state != MecAppState.ACTIVE:
            raise RuntimeError(f"MEC app '{self.app_name}' is not ACTIVE "
                                f"(state={self.state.value}); deploy() it first")

        t0 = time.perf_counter()
        result = self.engine.evaluate(strain_history_kpa, mean_temp_c, anomaly_severity)
        compute_ms = compute_time_ms_fn() if compute_time_ms_fn else \
            (time.perf_counter() - t0) * 1000.0

        record = MecInvocationRecord(window_index=window_index, receive_time_s=time.time(),
                                      compute_time_ms=compute_ms, result=result)
        self.invocations.append(record)
        return record
