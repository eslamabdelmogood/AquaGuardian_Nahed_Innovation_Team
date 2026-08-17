"""
axi.py
======
Models the High-Speed Parallel Data Interconnect between Hot SRAM and
the Micro-TPU worker array as a shared, arbitrated, bandwidth-limited
bus (AXI4-style: fixed bytes/cycle, one arbiter decision per cycle).

IMPORTANT: the spec's own "AXI4" label is only used for the *command*
bus between the Master and the Memory Controller (control-plane,
low-bandwidth). It does NOT specify the bandwidth of the *data* path
feeding the 10 worker cores. This module fills that gap with an
explicit, tunable assumption -- see config.AXI_WIDTH_BYTES.

This is a per-cycle recurring arbiter, not a one-shot DMA transfer
model: every active worker re-requests its bytes-per-cycle need every
cycle, and the arbiter decides who gets served (fully or partially)
out of the shared width. A worker that doesn't get its full request
this cycle is data-starved and cannot make compute progress -- that's
the real, load-bearing mechanism for detecting the bottleneck you
flagged.
"""


class AXIBus:
    def __init__(self, config):
        self.cfg = config
        self.width = config.AXI_WIDTH_BYTES
        self.arbitration = config.AXI_ARBITRATION
        self._rr_pointer = 0

        # Stats
        self.total_bytes_transferred = 0
        self.total_bytes_demanded = 0
        self.contended_cycles = 0       # cycles where demand > width
        self.starved_requests = 0       # individual (worker, cycle) under-served events
        self.cycles_with_any_demand = 0

    def arbitrate(self, requests: list) -> dict:
        """requests: list of (requester_id, bytes_wanted, priority).
        Lower priority value = served first under 'priority' arbitration.
        Returns {requester_id: bytes_granted} for this cycle only."""
        if not requests:
            return {}

        self.cycles_with_any_demand += 1
        total_demand = sum(r[1] for r in requests)
        self.total_bytes_demanded += total_demand
        if total_demand > self.width:
            self.contended_cycles += 1

        if self.arbitration == "priority":
            order = sorted(requests, key=lambda r: r[2])
        else:  # round_robin: rotate who gets served first each cycle
            ids_present = sorted(r[0] for r in requests)
            n = len(ids_present)
            start = self._rr_pointer % n
            rotated_ids = ids_present[start:] + ids_present[:start]
            self._rr_pointer = (self._rr_pointer + 1) % max(n, 1)
            by_id = {r[0]: r for r in requests}
            order = [by_id[i] for i in rotated_ids]

        granted = {}
        remaining = self.width
        for rid, want, _prio in order:
            give = min(want, remaining)
            granted[rid] = give
            remaining -= give
            self.total_bytes_transferred += give
            if give < want:
                self.starved_requests += 1

        return granted

    def utilization(self) -> float:
        """Fraction of total available bandwidth actually used, over
        cycles where at least one worker wanted data."""
        max_possible = self.cycles_with_any_demand * self.width
        if max_possible == 0:
            return 0.0
        return self.total_bytes_transferred / max_possible

    def contention_rate(self) -> float:
        if self.cycles_with_any_demand == 0:
            return 0.0
        return self.contended_cycles / self.cycles_with_any_demand
