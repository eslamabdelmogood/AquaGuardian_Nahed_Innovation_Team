"""
systolic.py
===========
Models one Micro-TPU's 8x8 systolic Multiply-Accumulate array, including
the Event-Driven Sparsity Gate (zero-value rows/columns are clock-gated
and skipped).

The model works at the granularity of "MAC-cycles required for a task"
rather than simulating individual PEs, which is the right resolution
for an architectural (not RTL-accurate) simulator. It is intentionally
simple so it can be swapped for a more detailed model later without
touching the rest of the codebase.
"""

import math


class SystolicArray:
    def __init__(self, config):
        self.cfg = config
        self.rows = config.SYSTOLIC_ROWS
        self.cols = config.SYSTOLIC_COLS
        self.macs_per_cycle = self.rows * self.cols
        self.total_mac_cycles_active = 0     # cycles that actually ran (real progress only)
        self.total_macs_executed = 0         # booked incrementally, per real cycle -- see record_active_cycle
        self.total_macs_gated = 0

    def plan_task(self, task) -> int:
        """Compute the cycle count and per-cycle MAC rates for `task`,
        WITHOUT booking any energy/throughput accounting yet. That
        accounting only happens as real cycles actually elapse (see
        record_active_cycle) -- otherwise a task that gets preempted
        before finishing would still be counted as if it had fully
        executed, which double-counts energy for work that was aborted
        and inflates throughput with output that was never produced.
        This bug existed through Phase 2; Phase 3's power model is what
        surfaced it (a livelock scenario reported ~0.68 TOPS and full
        MAC energy despite completing zero tasks)."""
        total_macs = task.mac_count
        sparsity = task.sparsity if self.cfg.SPARSITY_GATING_ENABLED else 0.0
        gated_macs = int(total_macs * sparsity)
        effective_macs = total_macs - gated_macs

        cycles = max(1, math.ceil(effective_macs / self.macs_per_cycle))

        task.effective_macs = effective_macs
        task.gated_macs = gated_macs
        task.effective_macs_per_cycle = effective_macs / cycles
        task.gated_macs_per_cycle = gated_macs / cycles

        return cycles

    def record_active_cycle(self, task):
        """Call exactly once per cycle in which this array made real
        (AXI-fed, non-starved) compute progress on `task`. Books energy/
        throughput accounting incrementally, so a task preempted after
        K of its N required cycles is only charged for K/N of its work
        -- matching what the hardware actually did before being cut off."""
        self.total_mac_cycles_active += 1
        self.total_macs_executed += task.effective_macs_per_cycle
        self.total_macs_gated += task.gated_macs_per_cycle

    def utilization(self) -> float:
        """Fraction of executed-MAC slots that were doing useful work,
        vs the theoretical max the cycles spent could have delivered."""
        theoretical_max = self.total_mac_cycles_active * self.macs_per_cycle
        if theoretical_max == 0:
            return 0.0
        return self.total_macs_executed / theoretical_max
