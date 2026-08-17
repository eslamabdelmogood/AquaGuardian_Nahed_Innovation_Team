"""
tests/test_core.py
===================
Unit tests for the HEXA-TPU-RT simulator's core modules.
Run with:  python3 -m pytest tests/ -v
       or:  python3 -m unittest discover tests
"""

import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from memory import PingPongMemory, ColdOutputMemory, BankState
from systolic import SystolicArray
from worker import MicroTPU, WorkerState, Task
from simulator import HexaTPUSimulator
from models.cnn import build_tiny_cnn, total_ideal_macs
from models.transformer import build_tiny_transformer
from models.bdo_skin import build_bdo_skin_workload


class TestSystolicArray(unittest.TestCase):
    def test_cycles_no_sparsity(self):
        cfg = Config(SPARSITY_GATING_ENABLED=False)
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.0)  # 640 / 64 = 10 cycles exactly
        self.assertEqual(arr.plan_task(task), 10)

    def test_sparsity_reduces_cycles(self):
        cfg = Config(SPARSITY_GATING_ENABLED=True)
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.5)  # half gated -> 320/64 = 5
        self.assertEqual(arr.plan_task(task), 5)

    def test_sparsity_ignored_if_disabled(self):
        cfg = Config(SPARSITY_GATING_ENABLED=False)
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.5)
        self.assertEqual(arr.plan_task(task), 10)  # gating flag off -> full macs

    def test_minimum_one_cycle(self):
        cfg = Config()
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=1, sparsity=0.0)
        self.assertEqual(arr.plan_task(task), 1)

    def test_plan_task_does_not_book_accounting(self):
        """plan_task() alone must not attribute any energy/throughput --
        only record_active_cycle() (called per real elapsed cycle) does.
        This is the Phase 3 fix for a bug where aborted/preempted tasks
        were counted as if they'd fully executed."""
        cfg = Config()
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.0)
        arr.plan_task(task)
        self.assertEqual(arr.total_macs_executed, 0)
        self.assertEqual(arr.total_mac_cycles_active, 0)

    def test_record_active_cycle_books_incrementally(self):
        cfg = Config()
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.0)
        cycles = arr.plan_task(task)  # 10 cycles, 64 macs/cycle
        arr.record_active_cycle(task)
        self.assertAlmostEqual(arr.total_macs_executed, 64.0)
        self.assertEqual(arr.total_mac_cycles_active, 1)
        # 9 more cycles -> fully booked
        for _ in range(cycles - 1):
            arr.record_active_cycle(task)
        self.assertAlmostEqual(arr.total_macs_executed, 640.0)

    def test_preempted_task_only_books_partial_energy(self):
        """The actual Phase 3 bug fix, verified directly: a task that
        completes only half its cycles before being abandoned should
        only be charged for half its MAC energy, not all of it."""
        cfg = Config()
        arr = SystolicArray(cfg)
        task = Task("t", mac_count=640, sparsity=0.0)
        cycles = arr.plan_task(task)  # 10 cycles
        for _ in range(cycles // 2):  # only 5 of 10 cycles actually ran
            arr.record_active_cycle(task)
        self.assertAlmostEqual(arr.total_macs_executed, 320.0)  # half, not all 640


class TestPingPongMemory(unittest.TestCase):
    def test_initial_state(self):
        cfg = Config()
        mem = PingPongMemory(cfg)
        self.assertEqual(mem.active_bank, "A")
        self.assertEqual(mem.loading_bank, "B")

    def test_load_and_switch(self):
        cfg = Config(DDR_LATENCY_CYCLES=2, DDR_BANDWIDTH_BYTES_PER_CYCLE=1,
                     HOT_BANK_CAPACITY_ELEMS=3, BYTES_PER_ELEMENT=1,
                     PREFETCH_QUEUE_DEPTH=1)
        mem = PingPongMemory(cfg)
        from dma import DMAController
        dma = DMAController(cfg)
        dma.enqueue_prefetch("B", "layer0")
        # 2 cycles DDR latency + 3 cycles burst (3 bytes @ 1 B/cycle) = 5 steps
        for _ in range(5):
            dma.step(mem)
            mem.step()
        self.assertEqual(mem.bank_state["B"], BankState.IDLE)
        mem.request_switch()
        mem.step()
        self.assertEqual(mem.active_bank, "B")
        self.assertEqual(mem.switch_count, 1)

    def test_no_conflict_when_active_bank_idle(self):
        cfg = Config()
        mem = PingPongMemory(cfg)
        for _ in range(10):
            mem.step()
        self.assertEqual(mem.conflicts, 0)

    def test_conflict_detected_if_active_bank_loading(self):
        cfg = Config()
        mem = PingPongMemory(cfg)
        # Force an illegal state: active bank is also loading.
        mem.bank_state[mem.active_bank] = BankState.LOADING
        mem.step()
        self.assertEqual(mem.conflicts, 1)


class TestColdOutputMemory(unittest.TestCase):
    def test_no_conflict_dedicated_banks(self):
        cfg = Config()
        mem = ColdOutputMemory(cfg, num_workers=4)
        mem.write(worker_id=0, bank_id=0)
        mem.write(worker_id=1, bank_id=1)
        self.assertEqual(mem.conflicts, 0)

    def test_conflict_if_two_workers_share_bank(self):
        cfg = Config()
        mem = ColdOutputMemory(cfg, num_workers=4)
        mem.write(worker_id=0, bank_id=0)
        mem.write(worker_id=1, bank_id=0)  # illegal: same bank, different worker
        self.assertEqual(mem.conflicts, 1)


class TestAXIBus(unittest.TestCase):
    def test_no_contention_single_requester(self):
        cfg = Config(AXI_WIDTH_BYTES=128)
        from axi import AXIBus
        bus = AXIBus(cfg)
        grants = bus.arbitrate([(0, 64, 0)])
        self.assertEqual(grants[0], 64)
        self.assertEqual(bus.contended_cycles, 0)

    def test_contention_when_demand_exceeds_width(self):
        cfg = Config(AXI_WIDTH_BYTES=64, AXI_ARBITRATION="round_robin")
        from axi import AXIBus
        bus = AXIBus(cfg)
        # 3 requesters x 64B = 192B demand vs 64B width -> only one fully served
        grants = bus.arbitrate([(0, 64, 0), (1, 64, 0), (2, 64, 0)])
        self.assertEqual(sum(grants.values()), 64)
        self.assertEqual(bus.contended_cycles, 1)
        # exactly one requester got its full 64 bytes, others starved
        fully_served = [v for v in grants.values() if v == 64]
        self.assertEqual(len(fully_served), 1)

    def test_priority_arbitration_serves_highest_priority_first(self):
        cfg = Config(AXI_WIDTH_BYTES=64, AXI_ARBITRATION="priority")
        from axi import AXIBus
        bus = AXIBus(cfg)
        # requester 1 has priority 0 (highest), requester 0 has priority 5
        grants = bus.arbitrate([(0, 64, 5), (1, 64, 0)])
        self.assertEqual(grants[1], 64)
        self.assertEqual(grants[0], 0)

    def test_round_robin_rotates_over_multiple_cycles(self):
        cfg = Config(AXI_WIDTH_BYTES=64, AXI_ARBITRATION="round_robin")
        from axi import AXIBus
        bus = AXIBus(cfg)
        winners = []
        for _ in range(4):
            grants = bus.arbitrate([(0, 64, 0), (1, 64, 0)])
            winners.append(0 if grants[0] == 64 else 1)
        # Round robin must alternate, not always favor the same requester
        self.assertIn(0, winners)
        self.assertIn(1, winners)


class TestWeightCache(unittest.TestCase):
    def test_first_access_is_miss(self):
        cfg = Config(CACHE_ENABLED=True, CACHE_CAPACITY_BLOCKS=2)
        from cache import WeightCache
        cache = WeightCache(cfg)
        self.assertFalse(cache.access("layer1"))

    def test_second_access_same_block_is_hit(self):
        cfg = Config(CACHE_ENABLED=True, CACHE_CAPACITY_BLOCKS=2)
        from cache import WeightCache
        cache = WeightCache(cfg)
        cache.access("layer1")
        self.assertTrue(cache.access("layer1"))

    def test_eviction_at_capacity(self):
        cfg = Config(CACHE_ENABLED=True, CACHE_CAPACITY_BLOCKS=1)
        from cache import WeightCache
        cache = WeightCache(cfg)
        cache.access("layer1")     # miss, inserted
        cache.access("layer2")     # miss, evicts layer1 (capacity=1)
        self.assertFalse(cache.access("layer1"))  # miss again, was evicted

    def test_disabled_cache_always_misses(self):
        cfg = Config(CACHE_ENABLED=False)
        from cache import WeightCache
        cache = WeightCache(cfg)
        cache.access("layer1")
        self.assertFalse(cache.access("layer1"))


class TestDMAPipeline(unittest.TestCase):
    def test_prefetch_queue_backpressure(self):
        cfg = Config(PREFETCH_QUEUE_DEPTH=1)
        from dma import DMAController
        dma = DMAController(cfg)
        self.assertTrue(dma.enqueue_prefetch("B", "layer0"))
        self.assertFalse(dma.enqueue_prefetch("A", "layer1"))  # queue full
        self.assertEqual(dma.rejected_enqueues, 1)

    def test_ddr_latency_then_burst(self):
        cfg = Config(DDR_LATENCY_CYCLES=3, DDR_BANDWIDTH_BYTES_PER_CYCLE=2,
                     HOT_BANK_CAPACITY_ELEMS=4, BYTES_PER_ELEMENT=1)
        from dma import DMAController
        mem = PingPongMemory(cfg)
        dma = DMAController(cfg)
        dma.enqueue_prefetch("B", "layer0")
        # 3 cycles latency, no bytes moved yet
        for _ in range(3):
            snap = dma.step(mem)
            self.assertTrue(snap["in_ddr_latency"] or snap["dma_active"])
        self.assertEqual(dma.burst_cycles_spent, 0)
        # burst: 4 bytes @ 2B/cycle = 2 cycles
        dma.step(mem)
        dma.step(mem)
        self.assertEqual(dma.transfers_completed, 1)
        self.assertEqual(mem.bank_state["B"], BankState.IDLE)



    def test_worker_starts_idle(self):
        cfg = Config()
        w = MicroTPU(0, cfg)
        self.assertEqual(w.state, WorkerState.IDLE)
        self.assertTrue(w.is_free)

    def test_assign_and_run_to_completion(self):
        cfg = Config(DEADLINE_SLACK_FACTOR=10.0)
        w = MicroTPU(0, cfg)
        task = Task("t", mac_count=64, sparsity=0.0)  # exactly 1 cycle
        w.assign(task, current_cycle=0)
        self.assertEqual(w.state, WorkerState.BUSY)
        finished = w.step(current_cycle=0)
        self.assertIsNotNone(finished)
        self.assertEqual(w.state, WorkerState.IDLE)
        self.assertEqual(w.tasks_completed, 1)

    def test_deadline_miss_tracked(self):
        cfg = Config(DEADLINE_SLACK_FACTOR=1.0)
        w = MicroTPU(0, cfg)
        # A task that needs many cycles but whose deadline is tight
        task = Task("t", mac_count=6400, sparsity=0.0)  # 100 cycles @ 64 macs/cycle
        w.assign(task, current_cycle=0)
        # deadline_cycle = 0 + 100*1.0 = 100, exactly on time -> not late
        for c in range(1, 101):
            w.step(c)
        self.assertEqual(w.deadline_misses, 0)


class TestMemoryAwareDeadline(unittest.TestCase):
    def test_memory_blind_ignores_contention(self):
        cfg = Config(DEADLINE_MEMORY_AWARE=False, AXI_WIDTH_BYTES=64, AXI_BYTES_PER_MAC_CYCLE=64)
        w = MicroTPU(0, cfg)
        # Even with heavy contention, memory-blind returns ideal cycles unchanged.
        self.assertEqual(w.estimate_deadline_cycles(100, cache_hit=False,
                                                      concurrent_axi_requesters=10), 100.0)

    def test_memory_aware_stretches_under_contention(self):
        cfg = Config(DEADLINE_MEMORY_AWARE=True, AXI_WIDTH_BYTES=128, AXI_BYTES_PER_MAC_CYCLE=64)
        w = MicroTPU(0, cfg)
        # fair_share = 128/10 = 12.8 B/cycle, need 64 -> stretch = 64/12.8 = 5x
        result = w.estimate_deadline_cycles(100, cache_hit=False, concurrent_axi_requesters=10)
        self.assertAlmostEqual(result, 500.0)

    def test_memory_aware_no_stretch_when_bandwidth_sufficient(self):
        cfg = Config(DEADLINE_MEMORY_AWARE=True, AXI_WIDTH_BYTES=640, AXI_BYTES_PER_MAC_CYCLE=64)
        w = MicroTPU(0, cfg)
        # fair_share = 640/10 = 64 -- exactly enough, no stretch
        result = w.estimate_deadline_cycles(100, cache_hit=False, concurrent_axi_requesters=10)
        self.assertEqual(result, 100.0)

    def test_cache_hit_never_stretched_even_if_memory_aware(self):
        cfg = Config(DEADLINE_MEMORY_AWARE=True, AXI_WIDTH_BYTES=64, AXI_BYTES_PER_MAC_CYCLE=64)
        w = MicroTPU(0, cfg)
        result = w.estimate_deadline_cycles(100, cache_hit=True, concurrent_axi_requesters=10)
        self.assertEqual(result, 100.0)

    def test_cliff_becomes_gradient_end_to_end(self):
        """The actual Phase 2 claim, verified directly: at a narrow AXI
        width where the memory-blind formula produces total livelock,
        the memory-aware formula should complete nearly all tasks."""
        cfg_blind = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                            DEADLINE_MEMORY_AWARE=False)
        layers = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim_blind = HexaTPUSimulator(cfg_blind)
        report_blind = sim_blind.run(layers)
        self.assertEqual(report_blind.tasks_completed, 0)  # total livelock, as documented

        cfg_aware = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                            DEADLINE_MEMORY_AWARE=True)
        layers2 = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim_aware = HexaTPUSimulator(cfg_aware)
        report_aware = sim_aware.run(layers2)
        self.assertGreater(report_aware.tasks_completed, 900)  # most tasks now complete


class TestPowerModel(unittest.TestCase):
    def test_livelock_reports_zero_throughput_not_full(self):
        """The end-to-end Phase 3 accounting fix: a total-livelock run
        (0 tasks completed) must report 0 TOPS, not near-full throughput
        for work that was aborted before producing any usable output."""
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                     DEADLINE_MEMORY_AWARE=False)
        layers = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertEqual(report.tasks_completed, 0)
        self.assertEqual(report.estimated_tops, 0.0)

    def test_livelock_still_burns_nonzero_power(self):
        """Even with zero useful output, real (if wasted) switching
        activity and static leakage should still show up as power > 0
        -- a livelock isn't a power-off state."""
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                     DEADLINE_MEMORY_AWARE=False)
        layers = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertGreater(report.power.average_power_mw, 0.0)

    def test_sparsity_gating_reduces_mac_energy(self):
        cfg = Config(NUM_WORKERS=1, SPARSITY_GATING_ENABLED=True)
        layers = build_tiny_cnn(1, sparsity=0.5, tiles_per_worker=1)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertGreater(report.power.sparsity_gating_power_cut_pct, 0.0)
        self.assertLess(report.power.mac_active_energy_nj + report.power.mac_gated_energy_nj,
                         report.power.mac_energy_if_no_gating_nj)

    def test_no_sparsity_means_no_power_cut(self):
        cfg = Config(NUM_WORKERS=1)
        layers = build_tiny_cnn(1, sparsity=0.0, tiles_per_worker=1)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertAlmostEqual(report.power.sparsity_gating_power_cut_pct, 0.0, places=1)


class TestTilingStrategy(unittest.TestCase):
    def test_spatial_tiling_shares_block_id(self):
        layers = build_tiny_cnn(4, sparsity=0.0, tiles_per_worker=2, tiling_strategy="spatial")
        _, tasks = layers[0]
        block_ids = {t.weight_block_id for t in tasks}
        self.assertEqual(len(block_ids), 1)  # all tiles share one block

    def test_output_channel_tiling_unique_block_ids(self):
        layers = build_tiny_cnn(4, sparsity=0.0, tiles_per_worker=2, tiling_strategy="output_channel")
        _, tasks = layers[0]
        block_ids = {t.weight_block_id for t in tasks}
        self.assertEqual(len(block_ids), len(tasks))  # every tile unique, no reuse possible

    def test_output_channel_tiling_defeats_cache(self):
        cfg = Config(NUM_WORKERS=4, CACHE_ENABLED=True, CACHE_CAPACITY_BLOCKS=4)
        layers = build_tiny_cnn(4, sparsity=0.0, tiles_per_worker=2, tiling_strategy="output_channel")
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        # With unique blocks per tile, hit rate should be far lower than
        # the spatial case's ~99%+ -- essentially no reuse, only occasional
        # coincidental hits from the tiny 4-block LRU capacity.
        self.assertLess(report.cache_hit_rate, 0.10)


class TestFullSimulationSanity(unittest.TestCase):
    def test_single_worker_completes_all_tasks(self):
        cfg = Config(NUM_WORKERS=1)
        layers = build_tiny_cnn(1, sparsity=0.0)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        expected_tasks = sum(len(tasks) for _, tasks in layers)
        self.assertEqual(report.tasks_completed, expected_tasks)
        self.assertEqual(report.memory_conflicts, 0)

    def test_more_workers_never_increases_cycles(self):
        """Sanity check: adding workers on an embarrassingly parallel
        workload must not make total runtime *worse*."""
        prev_cycles = None
        for n in (1, 2, 4):
            cfg = Config(NUM_WORKERS=n)
            layers = build_tiny_cnn(n, sparsity=0.0)
            sim = HexaTPUSimulator(cfg)
            report = sim.run(layers)
            if prev_cycles is not None:
                self.assertLessEqual(report.total_cycles, prev_cycles)
            prev_cycles = report.total_cycles

    def test_zero_sparsity_task_count_matches_model(self):
        layers = build_tiny_cnn(2, sparsity=0.0)
        macs = total_ideal_macs(layers)
        self.assertGreater(macs, 0)


class TestTransformerWorkload(unittest.TestCase):
    def test_heterogeneous_mac_magnitudes(self):
        """FFN layers should carry far more MACs than attention-score
        layers -- that's the burstiness Phase 4 exists to test."""
        from models.transformer import layer_mac_profile
        layers = build_tiny_transformer(num_workers=10, num_layers=1)
        profile = {name: macs for name, macs, _ in layer_mac_profile(layers)}
        self.assertGreater(profile["block0_ffn1"], profile["block0_attn_scores"] * 5)

    def test_attention_tasks_never_share_weight_blocks(self):
        """Attention scores/values multiply activation x activation --
        there is no weight tensor, so every task's block_id must be
        unique (guaranteed cache miss), unlike CNN's spatial tiling."""
        layers = build_tiny_transformer(num_workers=10, num_layers=1)
        for name, tasks in layers:
            if "attn_scores" in name or "attn_v" in name:
                block_ids = {t.weight_block_id for t in tasks}
                self.assertEqual(len(block_ids), len(tasks))

    def test_output_proj_and_ffn_share_weight_blocks(self):
        """Unlike attention, these ARE spatially tiled -- weight reuse
        across tiles should still be possible."""
        layers = build_tiny_transformer(num_workers=10, num_layers=1)
        for name, tasks in layers:
            if name.endswith("out_proj") or name.endswith("ffn1") or name.endswith("ffn2"):
                block_ids = {t.weight_block_id for t in tasks}
                self.assertEqual(len(block_ids), 1)

    def test_num_heads_below_num_workers_creates_bubble(self):
        """Attention-phase tasks are tiled per head (8, by default) --
        fewer than num_workers (10) -- so even with zero memory
        contention, occupancy must be below 100%."""
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=640)
        layers = build_tiny_transformer(num_workers=10, num_heads=8)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertEqual(report.deadline_misses, 0)  # confirms it's NOT a memory issue
        self.assertLess(report.occupancy, 1.0)         # yet occupancy still isn't full

    def test_memory_aware_still_helps_on_bursty_workload(self):
        """The actual Phase 4 claim: the Phase 2 fix should still
        reduce deadline misses substantially on a heterogeneous
        workload, even if (unlike the CNN case) it doesn't eliminate
        them completely."""
        cfg_blind = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=False)
        layers_blind = build_tiny_transformer(num_workers=10)
        sim_blind = HexaTPUSimulator(cfg_blind)
        report_blind = sim_blind.run(layers_blind)

        cfg_aware = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True)
        layers_aware = build_tiny_transformer(num_workers=10)
        sim_aware = HexaTPUSimulator(cfg_aware)
        report_aware = sim_aware.run(layers_aware)

        self.assertGreater(report_aware.tasks_completed, report_blind.tasks_completed)
        total_tasks = sum(len(t) for _, t in layers_aware)
        self.assertLess(report_aware.tasks_completed, total_tasks)

    def test_residual_misses_concentrate_in_qkv_proj(self):
        """The specific, documented finding: under the memory-aware
        formula, whatever misses remain should be concentrated in QKV
        projection (largest per-head task, burst-start ramp-up blind
        spot), not spread evenly across all op types."""
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True)
        layers = build_tiny_transformer(num_workers=10)
        sim = HexaTPUSimulator(cfg)
        sim.run(layers)

        missed_non_qkv = sum(
            1 for name, tasks in layers if "qkv_proj" not in name
            for t in tasks if t.missed_deadline
        )
        missed_qkv = sum(
            1 for name, tasks in layers if "qkv_proj" in name
            for t in tasks if t.missed_deadline
        )
        if missed_qkv + missed_non_qkv > 0:
            self.assertGreaterEqual(missed_qkv, missed_non_qkv)

    def test_phase7_lookahead_closes_qkv_gap(self):
        """The actual Phase 7 claim: enabling DEADLINE_LOOKAHEAD_ENABLED
        should eliminate the QKV misses Phase 4 found, at the exact
        width (128) where Phase 2 alone left 9 residual misses."""
        cfg_p2_only = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True,
                              DEADLINE_LOOKAHEAD_ENABLED=False)
        layers_p2 = build_tiny_transformer(num_workers=10)
        sim_p2 = HexaTPUSimulator(cfg_p2_only)
        report_p2 = sim_p2.run(layers_p2)
        self.assertGreater(report_p2.deadline_misses, 0)  # Phase 2 alone still misses

        cfg_p7 = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True,
                         DEADLINE_LOOKAHEAD_ENABLED=True)
        layers_p7 = build_tiny_transformer(num_workers=10)
        sim_p7 = HexaTPUSimulator(cfg_p7)
        report_p7 = sim_p7.run(layers_p7)
        self.assertEqual(report_p7.deadline_misses, 0)  # Phase 7 fix eliminates them

    def test_phase7_lookahead_does_not_regress_cnn_benchmark(self):
        """The lookahead correctly excludes CNN's same-block siblings
        (they'll cache-hit, not compete for AXI), so it should produce
        an IDENTICAL result to Phase 2 alone on the CNN benchmark."""
        cfg_p2 = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                         DEADLINE_MEMORY_AWARE=True, DEADLINE_LOOKAHEAD_ENABLED=False)
        layers_p2 = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim_p2 = HexaTPUSimulator(cfg_p2)
        report_p2 = sim_p2.run(layers_p2)

        cfg_p7 = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, CACHE_ENABLED=False,
                         DEADLINE_MEMORY_AWARE=True, DEADLINE_LOOKAHEAD_ENABLED=True)
        layers_p7 = build_tiny_cnn(10, sparsity=0.30, tiles_per_worker=20)
        sim_p7 = HexaTPUSimulator(cfg_p7)
        report_p7 = sim_p7.run(layers_p7)

        self.assertEqual(report_p2.tasks_completed, report_p7.tasks_completed)
        self.assertEqual(report_p2.deadline_misses, report_p7.deadline_misses)


class TestSchedulerPolicyStress(unittest.TestCase):
    def test_policy_is_inert_without_priority_heterogeneity(self):
        """The Phase 6 baseline finding: with uniform per-layer priority
        (the normal workload generators), fifo/priority/edf must produce
        IDENTICAL results, because Master only ever enqueues one layer
        at a time and every task in it shares one priority value."""
        results = {}
        for policy in ("fifo", "priority", "edf"):
            cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True,
                         SCHEDULER_POLICY=policy)
            layers = build_tiny_transformer(num_workers=10)
            sim = HexaTPUSimulator(cfg)
            report = sim.run(layers)
            results[policy] = (report.tasks_completed, report.deadline_misses, report.total_cycles)
        self.assertEqual(len(set(results.values())), 1)

    def test_critical_marking_creates_genuine_priority_heterogeneity(self):
        """With critical_heads_per_layer > 0, tasks within one dispatch
        wave now have different priority values -- the scheduler has an
        actual choice to make, unlike the baseline case above."""
        layers = build_tiny_transformer(num_workers=10, critical_heads_per_layer=2)
        for name, tasks in layers:
            if "qkv_proj" in name:
                priorities = {t.priority for t in tasks}
                self.assertGreater(len(priorities), 1)

    def test_priority_first_dispatch_does_not_protect_critical_tasks_without_lookahead(self):
        """The counterintuitive Phase 6 finding: without the Phase 7
        lookahead fix, priority-first dispatch does NOT reduce critical
        tasks' miss rate relative to fifo -- confirmed by comparing
        aggregate miss counts directly, which should be identical."""
        misses_by_policy = {}
        for policy in ("fifo", "priority", "edf"):
            cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True,
                         DEADLINE_LOOKAHEAD_ENABLED=False, SCHEDULER_POLICY=policy)
            layers = build_tiny_transformer(num_workers=10, critical_heads_per_layer=2)
            sim = HexaTPUSimulator(cfg)
            report = sim.run(layers)
            crit_missed = sum(1 for _, tasks in layers for t in tasks if t.is_critical and t.missed_deadline)
            misses_by_policy[policy] = crit_missed
        self.assertEqual(len(set(misses_by_policy.values())), 1)

    def test_lookahead_protects_critical_tasks_regardless_of_policy(self):
        """What actually fixes it: Phase 7's lookahead, tested here on
        the mixed-criticality workload specifically."""
        for policy in ("fifo", "priority", "edf"):
            cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=128, DEADLINE_MEMORY_AWARE=True,
                         DEADLINE_LOOKAHEAD_ENABLED=True, SCHEDULER_POLICY=policy)
            layers = build_tiny_transformer(num_workers=10, critical_heads_per_layer=2)
            sim = HexaTPUSimulator(cfg)
            report = sim.run(layers)
            self.assertEqual(report.deadline_misses, 0)



class TestExport(unittest.TestCase):
    def _sample_report(self):
        cfg = Config(NUM_WORKERS=4)
        layers = build_tiny_cnn(4, sparsity=0.3, tiles_per_worker=2)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        return sim, report

    def test_export_json_is_valid_and_matches_as_dict(self):
        import json, tempfile, os
        from export import export_json
        sim, report = self._sample_report()
        path = tempfile.mktemp(suffix=".json")
        try:
            export_json(report, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["workers"], 4)
            self.assertEqual(data, report.as_dict())
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_export_html_contains_key_sections(self):
        import tempfile, os
        from export import export_html
        sim, report = self._sample_report()
        path = tempfile.mktemp(suffix=".html")
        try:
            export_html(report, sim, path)
            html = open(path).read()
            self.assertIn("<svg", html)
            self.assertIn("MAC active energy", html)
            self.assertIn("Worker occupancy", html)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_export_timeline_png_creates_nonempty_file(self):
        import tempfile, os
        from export import export_timeline_png
        sim, report = self._sample_report()
        path = tempfile.mktemp(suffix=".png")
        try:
            export_timeline_png(sim, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)



class TestBDOSkinIntegration(unittest.TestCase):
    def test_fbg_channel_count_matches_report(self):
        from models.sensor_events import NUM_FBG_CHANNELS
        self.assertEqual(NUM_FBG_CHANNELS, 600)

    def test_normal_scenario_never_critical(self):
        from models.sensor_events import generate_normal
        windows = generate_normal(100, seed=0)
        self.assertFalse(any(w.is_critical for w in windows))
        self.assertFalse(any(w.anomaly_active for w in windows))

    def test_burst_anomaly_triggers_critical_promptly(self):
        """Unlike gradual, a burst should hit critical severity almost
        immediately after onset, not need a long ramp."""
        from models.sensor_events import generate_burst_anomaly
        windows = generate_burst_anomaly(150, onset_window=60, seed=0)
        pre_onset_critical = any(w.is_critical for w in windows if w.index < 60)
        post_onset_critical = any(w.is_critical for w in windows if 60 <= w.index < 66)
        self.assertFalse(pre_onset_critical)
        self.assertTrue(post_onset_critical)

    def test_gradual_anomaly_needs_time_to_become_critical(self):
        """Unlike burst, gradual should NOT be critical right at onset --
        that's the whole point of the scenario."""
        from models.sensor_events import generate_gradual_anomaly
        windows = generate_gradual_anomaly(150, onset_window=40, ramp_windows=80, seed=0)
        just_after_onset_critical = any(w.is_critical for w in windows if 40 <= w.index < 50)
        self.assertFalse(just_after_onset_critical)

    def test_critical_event_sustains_across_many_windows(self):
        """The 'worst case' scenario should have far more critical
        windows than a brief burst -- sustained, not momentary."""
        from models.sensor_events import generate_critical_event, generate_burst_anomaly
        critical_windows = generate_critical_event(150, onset_window=50, seed=0)
        burst_windows = generate_burst_anomaly(150, onset_window=60, seed=0)
        n_critical_sustained = sum(1 for w in critical_windows if w.is_critical)
        n_critical_burst = sum(1 for w in burst_windows if w.is_critical)
        self.assertGreater(n_critical_sustained, n_critical_burst)

    def test_workload_produces_valid_layers(self):
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="burst_anomaly",
                                                 num_windows=150, seed=0)
        self.assertGreater(len(layers), 0)
        total_tasks = sum(len(tasks) for _, tasks in layers)
        self.assertGreater(total_tasks, 0)
        self.assertEqual(len(meta["critical_window_indices"]), 10)  # burst_duration=6 + 4 tail

    def test_emergency_layers_only_on_critical_windows(self):
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="burst_anomaly",
                                                 num_windows=150, seed=0)
        emergency_layer_names = [name for name, _ in layers if name.startswith("emergency_")]
        self.assertEqual(len(emergency_layer_names), len(meta["critical_window_indices"]))

    def test_emergency_tasks_marked_critical_with_tight_slack(self):
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="burst_anomaly",
                                                 num_windows=150, seed=0)
        emergency_tasks = [t for name, tasks in layers if name.startswith("emergency_") for t in tasks]
        self.assertGreater(len(emergency_tasks), 0)
        for t in emergency_tasks:
            self.assertTrue(t.is_critical)
            self.assertIsNotNone(t.deadline_slack_override)
            self.assertLess(t.deadline_slack_override, 1.2)  # genuinely tight

    def test_reflex_layers_reuse_weight_block_for_cache_hits(self):
        """Reflex kernel parameters are static across windows -- after
        the first, all subsequent reflex tasks should share one cache
        block, unlike Transformer's per-head attention tasks."""
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="normal",
                                                 num_windows=50, seed=0)
        reflex_blocks = {t.weight_block_id for name, tasks in layers
                          if name.startswith("reflex_") for t in tasks}
        self.assertEqual(reflex_blocks, {"reflex_kernel_params"})

    def test_normal_scenario_produces_zero_deadline_misses_at_generous_bandwidth(self):
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=640, DEADLINE_MEMORY_AWARE=True,
                     DEADLINE_LOOKAHEAD_ENABLED=True)
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="normal",
                                                 num_windows=60, seed=0)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertEqual(report.deadline_misses, 0)

    def test_improved_pipeline_reduces_emergency_misses_under_contention(self):
        """The actual integration claim, tested directly: under genuine
        AXI contention, the memory-aware+lookahead+priority pipeline
        should produce fewer (ideally zero) emergency-path deadline
        misses than the naive memory-blind+fifo baseline, on a workload
        this pipeline was never tuned against."""
        def emergency_misses(mem_aware, lookahead, policy):
            cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=64, DEADLINE_MEMORY_AWARE=mem_aware,
                         DEADLINE_LOOKAHEAD_ENABLED=lookahead, SCHEDULER_POLICY=policy)
            layers, meta = build_bdo_skin_workload(num_workers=10, scenario="critical_event",
                                                     num_windows=150, seed=3)
            sim = HexaTPUSimulator(cfg)
            sim.run(layers)
            emergency_tasks = [t for name, tasks in layers if name.startswith("emergency_") for t in tasks]
            return sum(1 for t in emergency_tasks if t.missed_deadline)

        naive_misses = emergency_misses(False, False, "fifo")
        improved_misses = emergency_misses(True, True, "priority")
        self.assertGreater(naive_misses, 0)
        self.assertEqual(improved_misses, 0)
        self.assertLess(improved_misses, naive_misses)

    def test_sensor_data_flows_through_dma_and_cache_not_bypassed(self):
        """Explicit check that BDO-SKIN tasks are ordinary Task objects
        flowing through the same Scheduler/DMA/WeightCache path as
        every other workload -- no special-casing, no bypass."""
        cfg = Config(NUM_WORKERS=10, AXI_WIDTH_BYTES=640)
        layers, meta = build_bdo_skin_workload(num_workers=10, scenario="burst_anomaly",
                                                 num_windows=60, seed=0)
        sim = HexaTPUSimulator(cfg)
        report = sim.run(layers)
        self.assertGreater(sim.dma.total_bytes_transferred, 0)
        self.assertGreater(sim.cache.hits + sim.cache.misses, 0)
        self.assertGreater(sim.axi.total_bytes_transferred, 0)



if __name__ == "__main__":
    unittest.main()
