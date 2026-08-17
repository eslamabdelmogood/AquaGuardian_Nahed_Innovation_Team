"""
simulator.py
============
Top-level cycle-by-cycle simulator for the HEXA-TPU-RT architecture
(Phase 1.5): MasterControlCore -> Scheduler -> DMAController (DDR
latency + burst) -> PingPongMemory / ColdOutputMemory, with an AXIBus
arbitrating per-cycle weight-stream bandwidth to N x MicroTPU workers,
gated by a shared WeightCache.

Usage:
    from config import Config
    from simulator import HexaTPUSimulator
    from models.cnn import build_tiny_cnn

    cfg = Config(NUM_WORKERS=10)
    layers = build_tiny_cnn(cfg.NUM_WORKERS)
    sim = HexaTPUSimulator(cfg)
    report = sim.run(layers)
    print(report.render())
    print(sim.timeline.render_ascii())
"""

import time as _time

from config import Config
from memory import PingPongMemory, ColdOutputMemory
from dma import DMAController
from axi import AXIBus
from cache import WeightCache
from worker import MicroTPU, WorkerState
from scheduler import Scheduler
from master import MasterControlCore
from timeline import TimelineRecorder
from power import PowerModel


class Report:
    def __init__(self, cfg: Config, sim: "HexaTPUSimulator", wall_time_s: float):
        self.cfg = cfg
        self.sim = sim
        self.wall_time_s = wall_time_s

        self.total_cycles = sim.cycle
        self.busy_cycles = sum(w.busy_cycles for w in sim.workers)
        self.idle_cycles = sum(w.idle_cycles for w in sim.workers)
        self.waiting_cycles = sum(w.waiting_cycles for w in sim.workers)  # includes AXI-starved
        slot_cycles = self.total_cycles * len(sim.workers)

        self.mac_utilization = (
            sum(w.systolic.utilization() * w.busy_cycles for w in sim.workers) / self.busy_cycles
            if self.busy_cycles > 0 else 0.0
        )
        self.occupancy = self.busy_cycles / slot_cycles if slot_cycles else 0.0
        self.stall_pct = (self.waiting_cycles / slot_cycles * 100.0) if slot_cycles else 0.0

        self.memory_conflicts = sim.memory.conflicts + sim.cold_memory.conflicts
        self.deadline_misses = sum(w.deadline_misses for w in sim.workers)
        self.tasks_completed = sum(w.tasks_completed for w in sim.workers)

        finished = [t for _, tasks in sim.executed_layers for t in tasks if t.finish_cycle is not None]
        if finished:
            latencies_cycles = [t.finish_cycle - t.issue_cycle for t in finished]
            avg_latency_cycles = sum(latencies_cycles) / len(latencies_cycles)
        else:
            avg_latency_cycles = 0.0
        self.total_starved_cycles = sum(w.axi_starved_cycles for w in sim.workers)
        self.avg_latency_ms = avg_latency_cycles * cfg.cycle_time_ns() / 1e6

        # Throughput counts only work that actually finished and produced
        # a usable result -- NOT raw MAC-array activity. A task preempted
        # before completion burned real energy (see power model below,
        # which does use raw activity) but delivered zero useful output,
        # so it must not count toward "throughput delivered".
        total_effective_macs_completed = sum(t.effective_macs for t in finished)
        total_ops_done = total_effective_macs_completed * 2  # multiply + accumulate
        run_time_s = self.total_cycles * cfg.cycle_time_ns() / 1e9
        self.estimated_tops = (total_ops_done / run_time_s / 1e12) if run_time_s > 0 else 0.0
        self.peak_tops = cfg.peak_tops()

        # Phase 1.5 subsystem stats
        self.axi_utilization = sim.axi.utilization()
        self.axi_contention_rate = sim.axi.contention_rate()
        self.axi_bytes_transferred = sim.axi.total_bytes_transferred
        self.axi_bytes_demanded = sim.axi.total_bytes_demanded
        self.cache_hit_rate = sim.cache.hit_rate()
        self.dma_rejected_prefetches = sim.dma.rejected_enqueues
        self.dma_ddr_latency_cycles = sim.dma.ddr_latency_cycles_spent
        self.dma_burst_cycles = sim.dma.burst_cycles_spent

        # Phase 3: power model, computed from this run's own activity counters.
        self.power = PowerModel(cfg).compute(sim, self)

    def render(self) -> str:
        lines = []
        lines.append("=" * 50)
        lines.append("HEXA TPU RT Report")
        lines.append("=" * 50)
        lines.append(f"Workers:                {len(self.sim.workers)}")
        lines.append(f"Scheduler Policy:       {self.cfg.SCHEDULER_POLICY}")
        lines.append(f"Total Cycles:           {self.total_cycles}")
        lines.append(f"Busy Cycles:            {self.busy_cycles}")
        lines.append(f"Idle Cycles:            {self.idle_cycles}")
        lines.append(f"MAC Utilization:        {self.mac_utilization * 100:.1f}%")
        lines.append(f"Worker Occupancy:       {self.occupancy * 100:.1f}%")
        lines.append(f"Average Stall:          {self.stall_pct:.1f}%")
        lines.append(f"  of which AXI-starved: {self.total_starved_cycles} cycles")
        lines.append(f"Memory Conflicts:       {self.memory_conflicts}")
        lines.append(f"Deadline Misses:        {self.deadline_misses}")
        lines.append(f"Tasks Completed:        {self.tasks_completed}")
        lines.append(f"Average Latency:        {self.avg_latency_ms:.3f} ms")
        lines.append(f"Estimated Throughput:   {self.estimated_tops:.3f} TOPS "
                      f"(peak {self.peak_tops:.3f} TOPS)")
        lines.append("-" * 50)
        lines.append("Phase 1.5 subsystem detail:")
        lines.append(f"  AXI bus width:        {self.cfg.AXI_WIDTH_BYTES} B/cycle "
                      f"({self.cfg.AXI_ARBITRATION})")
        lines.append(f"  AXI utilization:      {self.axi_utilization * 100:.1f}%")
        lines.append(f"  AXI contention rate:  {self.axi_contention_rate * 100:.1f}% of demand-cycles")
        lines.append(f"  AXI bytes: {self.axi_bytes_transferred} transferred / "
                      f"{self.axi_bytes_demanded} demanded")
        lines.append(f"  Weight cache hit rate: {self.cache_hit_rate * 100:.1f}% "
                      f"(capacity {self.cfg.CACHE_CAPACITY_BLOCKS} blocks)")
        lines.append(f"  DMA rejected prefetches: {self.dma_rejected_prefetches} "
                      f"(prefetch queue depth {self.cfg.PREFETCH_QUEUE_DEPTH})")
        lines.append(f"  DDR latency cycles spent: {self.dma_ddr_latency_cycles}")
        lines.append(f"  DDR burst-transfer cycles: {self.dma_burst_cycles}")
        lines.append(self.power.render(self.estimated_tops))
        lines.append("=" * 50)
        lines.append("NOTE: throughput/latency figures are architectural")
        lines.append("estimates from this model, not measured silicon.")
        lines.append("=" * 50)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "workers": len(self.sim.workers),
            "scheduler_policy": self.cfg.SCHEDULER_POLICY,
            "total_cycles": self.total_cycles,
            "busy_cycles": self.busy_cycles,
            "idle_cycles": self.idle_cycles,
            "mac_utilization_pct": round(self.mac_utilization * 100, 2),
            "occupancy_pct": round(self.occupancy * 100, 2),
            "avg_stall_pct": round(self.stall_pct, 2),
            "axi_starved_cycles": self.total_starved_cycles,
            "memory_conflicts": self.memory_conflicts,
            "deadline_misses": self.deadline_misses,
            "tasks_completed": self.tasks_completed,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "estimated_tops": round(self.estimated_tops, 4),
            "peak_tops": round(self.peak_tops, 4),
            "axi_utilization_pct": round(self.axi_utilization * 100, 2),
            "axi_contention_rate_pct": round(self.axi_contention_rate * 100, 2),
            "cache_hit_rate_pct": round(self.cache_hit_rate * 100, 2),
            "dma_rejected_prefetches": self.dma_rejected_prefetches,
            "average_power_mw": round(self.power.average_power_mw, 3),
            "tops_per_watt": round(self.power.tops_per_watt, 4),
            "watts_per_tops": round(self.power.watts_per_tops, 4),
            "sparsity_gating_power_cut_pct": round(self.power.sparsity_gating_power_cut_pct, 2),
        }


class HexaTPUSimulator:
    def __init__(self, config: Config):
        self.cfg = config
        self.workers = [MicroTPU(i, config) for i in range(config.NUM_WORKERS)]
        self.memory = PingPongMemory(config)
        self.cold_memory = ColdOutputMemory(config, config.NUM_WORKERS)
        self.dma = DMAController(config)
        self.axi = AXIBus(config)
        self.cache = WeightCache(config)
        self.scheduler = Scheduler(config, self.workers, self.memory, self.dma, self.cache)
        self.master = MasterControlCore(config, self.scheduler, self.workers)
        self.timeline = TimelineRecorder(config, config.NUM_WORKERS)

        self.cycle = 0
        self.executed_layers = []   # filled in after run()

    def run(self, layers, max_cycles: int = 5_000_000) -> Report:
        t0 = _time.time()
        self.master.load_program(layers)
        self.executed_layers = layers

        while not self.master.program_complete and self.cycle < max_cycles:
            self.master.dispatch_next_layer_if_ready()
            self.scheduler.step(self.cycle)
            self.master.monitor_deadlines(self.cycle)

            self.memory.step()
            dma_snapshot = self.dma.step(self.memory)

            # --- AXI arbitration for this cycle -------------------------
            # Every BUSY worker whose current task missed the cache
            # re-requests its bytes-per-cycle need every cycle.
            requests = []
            for w in self.workers:
                if w.state == WorkerState.BUSY and w.current_task is not None \
                        and not w.current_task.cache_hit:
                    requests.append((w.id, self.cfg.AXI_BYTES_PER_MAC_CYCLE, w.current_task.priority))
            grants = self.axi.arbitrate(requests)
            axi_demand = sum(r[1] for r in requests)
            axi_supplied = sum(grants.values())

            # --- Worker compute step, gated by AXI feed -----------------
            self.cold_memory.step_reset()
            worker_busy_flags = []
            for w in self.workers:
                was_busy = w.state == WorkerState.BUSY
                fed = True
                if was_busy and w.current_task is not None and not w.current_task.cache_hit:
                    fed = grants.get(w.id, 0) >= self.cfg.AXI_BYTES_PER_MAC_CYCLE
                genuinely_computing = was_busy and fed  # for the timeline: nominal
                                                          # BUSY state alone is misleading
                                                          # when the array is data-starved
                finished = w.step(self.cycle, fed=fed)
                if finished is not None:
                    self.cold_memory.write(w.id, w.id % self.cfg.COLD_BANKS)
                worker_busy_flags.append(genuinely_computing)

            self.timeline.record_cycle(
                worker_busy_flags, dma_snapshot["dma_active"],
                axi_demand, axi_supplied, self.cfg.AXI_WIDTH_BYTES,
            )

            self.cycle += 1

        wall_time = _time.time() - t0
        return Report(self.cfg, self, wall_time)
