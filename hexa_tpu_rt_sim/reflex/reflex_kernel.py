"""
reflex/reflex_kernel.py
=========================
The "Instantaneous Local Reflex" layer: a domain-neutral, bounded-
latency decide-and-act path that runs at the edge, structurally
independent of the oneM2M -> URLLC -> MEC pipeline (middleware/,
network/, mec/). This is the piece that makes the platform's central
pitch literal rather than aspirational: "the same core protects an
aircraft in the air and an underground water network" only holds up
if the local reflex actually decides and acts on its own hard
deadline, for *either* domain, without waiting on a network round
trip -- not just compute something that happens to get scheduled on
the TPU alongside everything else.

Architecturally this models dedicated, always-on comparator-class
hardware (the "Micro-TPU / Spiking Reflex Kernel" in the pitch) --
NOT a task submitted to the main TPU's Scheduler/Worker pipeline
(scheduler.py/worker.py), which has queueing/contention latency by
design and is exactly what a hard real-time reflex path cannot
tolerate waiting on. That's why this module has no dependency on
scheduler.py, worker.py, or the MEC layer: the whole point is a
latency budget those shared, contended resources cannot guarantee.

What is modeled, as a named/adjustable assumption (same convention as
config.py and mec/bhs_cognitive_engine.py):

  - A fixed per-channel evaluation cost (`cycles_per_channel`) plus a
    fixed sampling/comparator overhead (`overhead_cycles`), converted
    to wall-clock time via a configurable reflex-kernel clock
    (independent from the main TPU's CLOCK_FREQ_MHZ in config.py,
    since this is deliberately separate, simpler silicon).
  - Only a small, LOCAL number of channels are read per decision (a
    handful of nearest-neighbor sensors), not the full FBG array --
    consistent with "involuntary reflex" scope: this path answers
    "is something dangerous happening right here, right now", not
    "what is the full structural/leak state of the whole system"
    (that broader picture is exactly what the MEC's BHS Cognitive
    Engine is for, on its own slower, standards-mediated path).

What is NOT modeled: real ADC/sensor sampling jitter, interrupt
latency, or actual embedded-hardware timing variance -- this is a
cycle-count deadline check in the same spirit as the rest of the
simulator's cycle-accurate (not wall-clock) methodology, not a claim
about a specific physical implementation's measured latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


# --------------------------------------------------------------------
# Trigger / actuation contracts -- the domain-neutral interface each
# vertical (aviation, water, or any future domain) plugs into.
# --------------------------------------------------------------------

@dataclass
class ActuationDecision:
    """What a ReflexTrigger decides to do, if anything. Domain-
    specific triggers populate `action` and `reason` with their own
    vocabulary (e.g. "avoidance_maneuver" / "valve_shutoff") -- the
    kernel itself is agnostic to what the words mean."""
    action: str
    reason: str
    severity: float = 1.0             # 0..1, for actuators/logs that want a magnitude
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflexTrigger(Protocol):
    """A domain's local danger-detection rule. Must be cheap and
    branch-free-ish by construction -- `channels_checked` is what the
    kernel bills against the cycle budget, so a trigger that reads a
    handful of local channels stays fast by design, not by promise."""

    channels_checked: int

    def evaluate(self, sample: Dict[str, Any]) -> Optional[ActuationDecision]:
        ...


class Actuator(Protocol):
    def actuate(self, event: "ReflexEvent") -> None:
        ...


class LoggingActuator:
    """Default actuator: records every fired event. Stands in for a
    real avoidance-maneuver controller or valve solenoid driver --
    swap in a domain-specific Actuator implementation to actually
    drive hardware; the kernel doesn't care which."""

    def __init__(self):
        self.log: List["ReflexEvent"] = []

    def actuate(self, event: "ReflexEvent") -> None:
        self.log.append(event)


# --------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------

@dataclass
class ReflexEvent:
    sample_index: int
    triggered: bool
    decision: Optional[ActuationDecision]
    cycles_used: int
    latency_ms: float
    deadline_ms: float
    deadline_met: bool
    wallclock_eval_ms: float          # actual Python-side eval time, for reference only


@dataclass
class ReflexKernelStats:
    count: int
    triggered_count: int
    deadline_misses: int
    max_latency_ms: float
    mean_latency_ms: float
    max_latency_margin_ms: float      # deadline_ms - worst-case latency_ms

    def summary(self) -> str:
        return (f"{self.count} samples, {self.triggered_count} triggered, "
                f"{self.deadline_misses} deadline misses, "
                f"max latency {self.max_latency_ms * 1000:.3f}\u00b5s "
                f"(deadline {self.max_latency_margin_ms:+.4f}ms margin)")


class ReflexKernel:
    """Domain-neutral bounded-latency local decide-and-act path.
    One instance = one dedicated reflex circuit for one edge node.
    Call `evaluate(sample)` on every new local sensor reading, in the
    same loop that would otherwise go straight to
    ADN_AE.push_sensor_window() -- this call must happen BEFORE that
    handoff, not after, since the entire point is deciding faster than
    the oneM2M/URLLC/MEC path could ever respond."""

    def __init__(self, trigger: ReflexTrigger, actuator: Optional[Actuator] = None,
                 deadline_ms: float = 1.0,
                 clock_freq_mhz: float = 1200.0,
                 overhead_cycles: int = 12,
                 cycles_per_channel: int = 12):
        """
        deadline_ms: the hard bound this reflex path must meet
            (1ms default, matching both pitches' "<1ms" requirement).
        clock_freq_mhz: the dedicated reflex-kernel clock -- separate
            from config.py's CLOCK_FREQ_MHZ, since the pitch
            describes distinct "Micro-TPU / Spiking Reflex Kernel"
            silicon, not a task on the main systolic array.
        overhead_cycles: fixed cost of latching a sample + arming the
            comparator, independent of how many channels are read.
        cycles_per_channel: cost of one threshold/rate-of-change
            comparison. Both are named, adjustable assumptions -- see
            module docstring.
        """
        self.trigger = trigger
        self.actuator = actuator or LoggingActuator()
        self.deadline_ms = deadline_ms
        self.clock_freq_mhz = clock_freq_mhz
        self.overhead_cycles = overhead_cycles
        self.cycles_per_channel = cycles_per_channel

        self.events: List[ReflexEvent] = []

    def _cycles_to_ms(self, cycles: int) -> float:
        ns_per_cycle = 1000.0 / self.clock_freq_mhz   # MHz -> ns/cycle
        return (cycles * ns_per_cycle) / 1e6

    def evaluate(self, sample: Dict[str, Any], sample_index: int = 0) -> ReflexEvent:
        t0 = time.perf_counter()
        decision = self.trigger.evaluate(sample)
        wallclock_ms = (time.perf_counter() - t0) * 1000.0

        cycles = self.overhead_cycles + self.trigger.channels_checked * self.cycles_per_channel
        latency_ms = self._cycles_to_ms(cycles)
        deadline_met = latency_ms <= self.deadline_ms

        event = ReflexEvent(
            sample_index=sample_index, triggered=decision is not None, decision=decision,
            cycles_used=cycles, latency_ms=latency_ms, deadline_ms=self.deadline_ms,
            deadline_met=deadline_met, wallclock_eval_ms=wallclock_ms,
        )
        self.events.append(event)

        if decision is not None:
            self.actuator.actuate(event)

        return event

    def stats(self) -> ReflexKernelStats:
        if not self.events:
            return ReflexKernelStats(0, 0, 0, 0.0, 0.0, self.deadline_ms)
        lats = [e.latency_ms for e in self.events]
        misses = sum(1 for e in self.events if not e.deadline_met)
        triggered = sum(1 for e in self.events if e.triggered)
        max_lat = max(lats)
        return ReflexKernelStats(
            count=len(self.events), triggered_count=triggered, deadline_misses=misses,
            max_latency_ms=max_lat, mean_latency_ms=sum(lats) / len(lats),
            max_latency_margin_ms=self.deadline_ms - max_lat,
        )
