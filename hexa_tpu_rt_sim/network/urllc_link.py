"""
network/urllc_link.py
======================
Network protocol layer connecting the ADN-AE edge hardware to the MEC
server, modeled as a 5G New Radio URLLC (Ultra-Reliable Low-Latency
Communication) link -- 3GPP TS 22.261's URLLC service class, whose
headline targets this module is calibrated against are 1 ms user-plane
latency and >=99.999% reliability for the air-interface hop alone.

This is a statistical latency/reliability model, not a PHY/MAC-layer
radio simulator: no resource-block scheduling, no HARQ state machine,
no channel model. What it does model, explicitly and as named,
tunable assumptions (same convention as config.py):

  - Base one-way latency: log-normal around a configurable mean, since
    real URLLC latency distributions are right-skewed (occasional
    retransmission-driven tail) rather than symmetric.
  - HARQ-style retransmission on a (rare) simulated radio-link failure,
    which adds one full round-trip before the packet is delivered --
    the mechanism 5G NR actually uses to hit >=99.999% *end-to-end*
    reliability despite an imperfect single-shot PHY.
  - A hard SLA check against the project's target: sub-10 ms
    transmission latency end-to-end (edge -> MEC), per this
    integration's stated requirement -- looser than the 1 ms
    air-interface-only 3GPP URLLC target because it also has to cover
    the MN-CSE hop this project's data actually transits before
    reaching the MEC platform.

Every numeric default below is a named, adjustable assumption for a
sub-6GHz 5G NR URLLC deployment (typical reported field figures), not
a spec value re-derived from a 3GPP conformance test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TransmissionResult:
    payload: Any
    latency_ms: float
    retransmitted: bool
    delivered: bool
    sla_met: bool                 # latency_ms <= link.sla_target_ms and delivered


class URLLCLink:
    """A single edge<->MEC 5G URLLC link. One instance models one
    ADN-AE's radio connection to its serving MEC host."""

    def __init__(self,
                 mean_latency_ms: float = 2.5,
                 jitter_sigma_ms: float = 0.6,
                 radio_link_failure_prob: float = 0.001,
                 retransmission_penalty_ms: float = 4.0,
                 sla_target_ms: float = 10.0,
                 seed: Optional[int] = None):
        self.mean_latency_ms = mean_latency_ms
        self.jitter_sigma_ms = jitter_sigma_ms
        self.radio_link_failure_prob = radio_link_failure_prob
        self.retransmission_penalty_ms = retransmission_penalty_ms
        self.sla_target_ms = sla_target_ms
        self._rng = random.Random(seed)

        self.transmissions: List[TransmissionResult] = []

    def _sample_base_latency_ms(self) -> float:
        # Log-normal: mean_latency_ms is the distribution's approximate
        # median-ish center; jitter_sigma_ms controls the (right) skew.
        mu = 0.0
        sigma = max(1e-6, self.jitter_sigma_ms / max(self.mean_latency_ms, 1e-6))
        multiplier = self._rng.lognormvariate(mu, sigma)
        return max(0.05, self.mean_latency_ms * multiplier)

    def transmit(self, payload: Any) -> TransmissionResult:
        """Simulates sending one payload (a oneM2M contentInstance
        envelope, typically) edge -> MEC. Returns the delivered
        payload plus latency/reliability telemetry."""
        latency = self._sample_base_latency_ms()
        retransmitted = False
        delivered = True

        if self._rng.random() < self.radio_link_failure_prob:
            # One HARQ retransmission round -- modeled as delivered
            # after paying the extra round-trip, matching how URLLC
            # reaches its reliability target rather than silently
            # dropping the packet.
            retransmitted = True
            latency += self.retransmission_penalty_ms + self._sample_base_latency_ms()

        result = TransmissionResult(
            payload=payload, latency_ms=latency, retransmitted=retransmitted,
            delivered=delivered, sla_met=delivered and latency <= self.sla_target_ms,
        )
        self.transmissions.append(result)
        return result

    # -- reporting -------------------------------------------------------
    def stats(self) -> dict:
        if not self.transmissions:
            return {"count": 0}
        lats = [t.latency_ms for t in self.transmissions]
        sla_hits = sum(1 for t in self.transmissions if t.sla_met)
        retx = sum(1 for t in self.transmissions if t.retransmitted)
        return {
            "count": len(self.transmissions),
            "mean_latency_ms": sum(lats) / len(lats),
            "max_latency_ms": max(lats),
            "min_latency_ms": min(lats),
            "p99_latency_ms": sorted(lats)[int(0.99 * (len(lats) - 1))],
            "sla_target_ms": self.sla_target_ms,
            "sla_compliance_rate": sla_hits / len(self.transmissions),
            "retransmission_rate": retx / len(self.transmissions),
        }
