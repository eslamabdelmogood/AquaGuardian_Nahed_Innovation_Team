"""
worker.py
=========
Models a single Micro-TPU worker core: its systolic array plus the
state machine (IDLE / BUSY / WAITING_MEMORY) that the top-level
simulator samples every cycle to build the trace and the final report.
"""

from enum import Enum
from systolic import SystolicArray


class WorkerState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING_MEMORY = "waiting_memory"
    PREEMPTED = "preempted"


class Task:
    """One unit of scheduled work: a matmul/conv tile assigned to a
    single worker. `mac_count` and `sparsity` come from the model
    (see models/cnn.py).

    Phase 1.5 additions:
      weight_block_id -- cache key (see cache.py); tasks from the same
                          layer share a block, modeling weight reuse.
      priority         -- lower = more critical; used by the "priority"
                           and "edf" scheduler policies.
      cache_hit        -- resolved at assignment time; if True the task
                           streams unconstrained (already in cache), if
                           False it is AXI-bandwidth-gated every cycle.
    """

    _next_id = 0

    def __init__(self, layer_name: str, mac_count: int, sparsity: float,
                 issue_cycle: int = None, weight_block_id=None, priority: int = 0):
        self.id = Task._next_id
        Task._next_id += 1
        self.layer_name = layer_name
        self.mac_count = mac_count
        self.sparsity = sparsity
        self.weight_block_id = weight_block_id if weight_block_id is not None else layer_name
        self.priority = priority
        self.cache_hit = False
        self.is_critical = False            # set by workload generators that model criticality tiers
        # Per-task deadline slack override. None = use cfg.DEADLINE_SLACK_FACTOR
        # (the global default). Set by workload generators that need a
        # tighter hard-RT deadline for specific tasks regardless of the
        # global setting -- e.g. BDO-SKIN's emergency/reflex path, which
        # has a genuine sub-second response requirement independent of
        # whatever slack the rest of the workload uses.
        self.deadline_slack_override = None

        self.issue_cycle = issue_cycle      # cycle it was dispatched
        self.deadline_cycle = None          # set by scheduler
        self.cycles_required = None         # set by worker on start (ideal, memory-blind)
        self.expected_cycles = None         # set by worker on start (memory-aware estimate, if enabled)
        self.cycles_remaining = None
        self.effective_macs = None          # set by SystolicArray.plan_task
        self.gated_macs = None
        self.effective_macs_per_cycle = None
        self.gated_macs_per_cycle = None
        self.start_cycle = None
        self.finish_cycle = None
        self.missed_deadline = False
        self.starved_cycles = 0             # cycles lost waiting on AXI bandwidth


class MicroTPU:
    def __init__(self, worker_id: int, config):
        self.id = worker_id
        self.cfg = config
        self.systolic = SystolicArray(config)
        self.state = WorkerState.IDLE
        self.current_task: Task = None

        # bookkeeping
        self.busy_cycles = 0
        self.idle_cycles = 0
        self.waiting_cycles = 0
        self.axi_starved_cycles = 0    # tracked independently of task completion,
                                        # so it's correct even under total livelock
        self.tasks_completed = 0
        self.deadline_misses = 0

    @property
    def is_free(self) -> bool:
        return self.state == WorkerState.IDLE and self.current_task is None

    def assign(self, task: Task, current_cycle: int, cache_hit: bool = False,
               concurrent_axi_requesters: int = 1):
        """Scheduler hands this worker a new task. `cache_hit` is
        resolved by the scheduler against the shared WeightCache before
        calling this -- if True, this task never contends for AXI
        bandwidth (its weights are already on-chip).

        `concurrent_axi_requesters` is the scheduler's estimate of how
        many workers (including this one) will be contending for AXI
        bandwidth at the moment this task starts. Used only if
        cfg.DEADLINE_MEMORY_AWARE is True -- see estimate_deadline().
        """
        task.cache_hit = cache_hit
        ideal_cycles = self.systolic.plan_task(task)
        task.cycles_required = ideal_cycles
        task.cycles_remaining = ideal_cycles
        task.start_cycle = current_cycle

        expected_cycles = self.estimate_deadline_cycles(
            ideal_cycles, cache_hit, concurrent_axi_requesters
        )
        task.expected_cycles = expected_cycles
        slack = (task.deadline_slack_override if task.deadline_slack_override is not None
                 else self.cfg.DEADLINE_SLACK_FACTOR)
        task.deadline_cycle = current_cycle + int(expected_cycles * slack)

        self.current_task = task
        self.state = WorkerState.BUSY

    def estimate_deadline_cycles(self, ideal_cycles: int, cache_hit: bool,
                                  concurrent_axi_requesters: int) -> float:
        """Phase 1.5's deadline was `ideal_cycles` alone -- purely a
        function of MAC count, blind to memory contention. That's the
        mechanism behind the livelock cliff documented in the Phase 1.5
        README: below a bandwidth threshold, EVERY task blows its
        (unrealistically tight) deadline; above it, NONE do.

        When DEADLINE_MEMORY_AWARE is on, this instead derates the
        worker's achievable bytes/cycle by its estimated fair share of
        the AXI bus (width / concurrent requesters) and stretches the
        deadline by however much slower that makes the task, so a
        task competing with 9 others for a narrow bus gets a
        proportionally longer deadline instead of an impossible one.
        This is what should turn the cliff into a slope, if it works.
        """
        if cache_hit or not self.cfg.DEADLINE_MEMORY_AWARE:
            return float(ideal_cycles)

        fair_share_bw = self.cfg.AXI_WIDTH_BYTES / max(1, concurrent_axi_requesters)
        if fair_share_bw >= self.cfg.AXI_BYTES_PER_MAC_CYCLE:
            stretch_factor = 1.0  # bus has enough bandwidth for everyone
        else:
            stretch_factor = self.cfg.AXI_BYTES_PER_MAC_CYCLE / fair_share_bw
        return ideal_cycles * stretch_factor

    def wait_on_memory(self):
        """Called by the simulator when the hot bank this worker needs
        isn't ready yet (only possible transiently right after a bank
        switch is requested but not yet complete)."""
        self.state = WorkerState.WAITING_MEMORY

    def step(self, current_cycle: int, fed: bool = True):
        """Advance this worker by one cycle. `fed` is False when this
        worker's AXI request for weight-stream bytes was not fully
        granted this cycle -- in that case the array is data-starved
        and cannot advance, even though it's nominally BUSY. Returns
        the task if it just finished this cycle, else None."""
        finished_task = None

        if self.state == WorkerState.BUSY and self.current_task is not None:
            if fed:
                self.busy_cycles += 1
                self.current_task.cycles_remaining -= 1
                self.systolic.record_active_cycle(self.current_task)
            else:
                # Data-starved: array is powered and assigned, but has
                # nothing to multiply this cycle. Counted separately
                # from a true WAITING_MEMORY state (that's for bank
                # switch stalls) so the two causes stay distinguishable
                # in the report.
                self.waiting_cycles += 1
                self.axi_starved_cycles += 1
                self.current_task.starved_cycles += 1

            if self.current_task.cycles_remaining <= 0:
                self.current_task.finish_cycle = current_cycle
                if current_cycle > self.current_task.deadline_cycle:
                    self.current_task.missed_deadline = True
                    self.deadline_misses += 1
                finished_task = self.current_task
                self.tasks_completed += 1
                self.current_task = None
                self.state = WorkerState.IDLE

        elif self.state == WorkerState.WAITING_MEMORY:
            self.waiting_cycles += 1
            self.state = WorkerState.IDLE  # memory waits resolve in 1 cycle in this model

        else:  # IDLE
            self.idle_cycles += 1

        return finished_task
