"""
config.py
=========
Central configuration for the HEXA-TPU-RT Architecture Simulator.

All numbers here are *target* architectural parameters taken from the
HEXA-TPU-RT specification (v1.0). Nothing in this file is a measured
hardware value -- it is the input assumption set the simulator uses to
produce estimates. Change these to explore design-space trade-offs.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Compute array -----------------------------------------------
    NUM_WORKERS: int = 1          # Phase 1 default. Spec target = 10.
    SYSTOLIC_ROWS: int = 8
    SYSTOLIC_COLS: int = 8

    # ---- Clocking -------------------------------------------------------
    CLOCK_FREQ_MHZ: float = 800.0

    # ---- Sparsity gating --------------------------------------------
    SPARSITY_GATING_ENABLED: bool = True
    # Fraction of zero-valued weights/activations assumed in the
    # workload. This is a *workload* property, not a hardware one --
    # exposed here so benchmarks can sweep it.
    DEFAULT_SPARSITY: float = 0.30

    # ---- Ping-Pong hot memory -----------------------------------------
    HOT_BANK_CAPACITY_ELEMS: int = 65536     # elements per bank (A/B)
    BANK_SWITCH_LATENCY_CYCLES: int = 1      # spec claims 1-cycle switch
    BYTES_PER_ELEMENT: int = 1               # INT8

    # ---- External memory tier (NOT in the spec) ------------------------
    # The spec shows on-chip SRAM only -- it never mentions where SRAM's
    # contents come from. A real chip needs an external DDR (or similar)
    # tier feeding the SRAM through a DMA engine. Modeled explicitly here
    # so this assumption is visible and tunable, not silently baked in.
    DDR_LATENCY_CYCLES: int = 100                 # fixed access latency per burst
    DDR_BANDWIDTH_BYTES_PER_CYCLE: int = 32        # off-chip interface width
    PREFETCH_QUEUE_DEPTH: int = 2                  # DMA descriptors in flight (2 = double buffer)

    # ---- High-Speed Parallel Data Interconnect (SRAM <-> Workers) ------
    # The spec names an "AXI4 Command Bus" only for Master<->Memory
    # Controller control traffic. It does NOT specify the bandwidth of
    # the *data* path from Hot SRAM to the 10 Micro-TPU workers. Modeled
    # here AXI4-style (shared, arbitrated, fixed per-cycle byte width)
    # because that data path is exactly where "every worker requests N
    # bytes/cycle" contention would actually occur.
    AXI_WIDTH_BYTES: int = 128                # shared interconnect width, bytes/cycle
    AXI_ARBITRATION: str = "round_robin"      # "round_robin" | "priority"
    AXI_BYTES_PER_MAC_CYCLE: int = 64         # bytes a worker needs each active cycle

    # ---- Weight cache (NOT in the spec) --------------------------------
    # Small shared cache in front of the AXI data path. Keyed per layer:
    # the first task of a layer must stream its weights in (AXI-gated);
    # every subsequent task of the SAME layer, on any worker, hits the
    # cache and streams unconstrained. This models weight reuse across
    # output-channel tiles, which is normal in real accelerators but is
    # not mentioned anywhere in the HEXA-TPU-RT spec.
    CACHE_ENABLED: bool = True
    CACHE_CAPACITY_BLOCKS: int = 4

    # ---- Tiling strategy (NOT in the spec) -----------------------------
    # Phase 1.5 silently assumed "spatial" tiling (every tile of a layer
    # shares the same weights -- optimistic, high cache reuse). Real
    # accelerators often tile by output channel instead, where each
    # tile needs a genuinely different weight slice -- no reuse at all.
    # The spec doesn't say which HEXA-TPU-RT uses, so both are now
    # explicit and selectable rather than one being a hidden default:
    #   "spatial"        -- all tiles of a layer share weight_block_id
    #                        (Phase 1.5 behavior, high cache hit rate)
    #   "output_channel"  -- every tile gets a unique weight_block_id
    #                        (no reuse possible, worst case for AXI)
    TILING_STRATEGY: str = "spatial"

    # Physical reality check on "zero-bus-conflict": the active ping-pong
    # bank is a single SRAM macro with a finite number of read ports.
    # This caps how many workers can be issued a *fresh* input tile in
    # the same cycle. The spec doesn't state a port count, so this is
    # an explicit, documented assumption -- set it to NUM_WORKERS to
    # simulate an (unrealistic) fully-multi-ported memory, or lower it
    # to see where contention appears.
    HOT_BANK_READ_PORTS: int = 2

    # ---- Cold output memory -----------------------------------------
    COLD_BANKS: int = 10                     # one per Micro-TPU (max)
    COLD_BANK_CAPACITY_ELEMS: int = 65536

    # ---- Real-time scheduler -----------------------------------------
    # Deadline given to a task as a multiplier over its *ideal*
    # (zero-stall, fully-utilized) cycle count. 1.0 = no slack.
    DEADLINE_SLACK_FACTOR: float = 1.5
    PREEMPTION_ENABLED: bool = True
    # "fifo" | "priority" | "edf" (earliest deadline first)
    # The spec says "Hard Real-Time Hardware Scheduler" but never states
    # the policy. FIFO was Phase 1's silent default -- now explicit and
    # selectable, since policy choice materially changes deadline misses.
    SCHEDULER_POLICY: str = "fifo"

    # Phase 1.5 found that deadlines computed from ideal systolic-only
    # cycle counts (no memory-wait allowance) produce a total-livelock
    # cliff under AXI contention rather than graceful degradation: below
    # a bandwidth threshold, EVERY task misses its deadline; above it,
    # NONE do. When True, the deadline instead accounts for expected AXI
    # contention at assignment time (see worker.py:estimate_deadline),
    # which should turn that cliff into a slope -- worth comparing both.
    DEADLINE_MEMORY_AWARE: bool = False

    # Phase 4 found the memory-aware deadline still misses on QKV
    # projection specifically: the contention estimate is a snapshot of
    # currently-busy workers at assignment time, blind to sibling tasks
    # in the same dispatch wave that haven't started yet but are
    # guaranteed to also need AXI (e.g. the other 7 attention heads,
    # sitting right there in the queue). When True, the estimate also
    # counts same-layer queued tasks with a genuinely different
    # weight_block_id (i.e. ones that WILL miss cache, unlike CNN's
    # same-block siblings which would hit) as future contenders.
    DEADLINE_LOOKAHEAD_ENABLED: bool = False

    # ---- Reporting -----------------------------------------------------
    VERBOSE_TRACE: bool = False    # print per-cycle state if True
    TRACE_EVERY_N_CYCLES: int = 1  # sampling rate for the trace log
    TIMELINE_BIN_WIDTH_CYCLES: int = 500   # cycles averaged per timeline column

    # ---- Power model (NOT in the spec beyond two headline numbers) -----
    # The spec claims "up to 40% power cut" from sparsity gating and a
    # "~0.35 W/TOPS" target. Neither is derivable from anything else in
    # the document -- there's no process-node power characterization, no
    # per-MAC or per-byte energy figures. Every constant below is an
    # explicit, order-of-magnitude assumption for a 16-28nm edge
    # accelerator, sourced from general published ranges for INT8 MAC
    # energy and the well-established rule that off-chip DRAM access
    # costs roughly two orders of magnitude more energy per byte than
    # on-chip SRAM -- NOT measured or vendor-verified for this design.
    # Treat every number here as "plausible ballpark", not fact.
    ENERGY_PER_MAC_ACTIVE_PJ: float = 0.20     # pJ per active INT8 MAC
    ENERGY_PER_MAC_GATED_PJ: float = 0.01      # residual pJ per sparsity-gated MAC (gating is imperfect)
    SRAM_ENERGY_PJ_PER_BYTE: float = 0.05      # on-chip SRAM/AXI read energy per byte
    DDR_ENERGY_PJ_PER_BYTE: float = 20.0       # off-chip DDR access energy per byte (~400x SRAM)
    STATIC_LEAKAGE_MW_PER_WORKER: float = 2.0  # always-on leakage per Micro-TPU
    STATIC_LEAKAGE_MW_MASTER: float = 3.0      # RISC-V master core leakage
    STATIC_LEAKAGE_MW_INTERCONNECT: float = 5.0  # AXI/memory-controller/misc leakage
    # If True, idle workers' leakage is derated (models clock/power
    # gating of idle cores). The spec doesn't claim this exists, so it
    # defaults off (worst case: always-on leakage).
    POWER_GATE_IDLE_WORKERS: bool = False
    IDLE_WORKER_LEAKAGE_FRACTION: float = 0.1

    def cycle_time_ns(self) -> float:
        return 1000.0 / self.CLOCK_FREQ_MHZ

    def macs_per_cycle_per_worker(self) -> int:
        return self.SYSTOLIC_ROWS * self.SYSTOLIC_COLS

    def total_peak_macs_per_cycle(self) -> int:
        return self.macs_per_cycle_per_worker() * self.NUM_WORKERS

    def peak_tops(self) -> float:
        """Peak (never-stalling, zero-sparsity) throughput in TOPS.
        One MAC == 2 ops (multiply + accumulate)."""
        macs_per_sec = self.total_peak_macs_per_cycle() * self.CLOCK_FREQ_MHZ * 1e6
        ops_per_sec = macs_per_sec * 2
        return ops_per_sec / 1e12
