"""
models/bdo_skin.py
===================
Translates the BDO-SKIN sensor stream (models/sensor_events.py) into
the exact (layer_name, [Task, ...]) format every other workload in
this project uses -- Master.load_program() and everything downstream
of it (Scheduler, DMA, AXI, WeightCache, workers, Power model) needs
NO changes to run this. That is deliberate: BDO-SKIN is a workload
generator, not a new simulator.

Per sensor window, up to three kinds of layers are dispatched, mapping
onto the BDO-SKIN report's own architecture (Section 2.2-2.5):

  - Reflex layer (every window): the Spiking Reflex Kernel's per-channel
    processing of all 600 FBG channels. Lightweight and frequent, tiled
    across a modest SUBSET of workers (REFLEX_TILES) -- most of the
    chip stays free for the emergency path, matching real low-power
    always-on monitoring.
  - BHS cognitive layers (every `bhs_period` windows, plus immediately
    after any critical window): Bat (per-cell trend fit + forecast),
    Hermit Crab (candidate-action stability scoring), Squid (objective
    reweighting) -- three separate layers, moderate MAC cost, tiled
    across roughly half the chip.
  - Emergency/Reflex layer (ONLY when a window's is_critical flag is
    set): full-mesh urgent reflex processing + a fast localized Bat
    pass on the affected region. High priority (task.priority very low
    = most urgent, task.is_critical=True, reusing the Phase 6
    convention), a TIGHT deadline_slack_override (a genuine sub-second
    hard-RT requirement independent of the rest of the workload's
    slack), and tiled across ALL workers -- this is the "sudden burst
    of critical work" the integration exists to stress-test.

MAC-cost formulas below are a stimulus generator calibrated to be
architecturally plausible (lightweight reflex, heavier periodic
cognition, a genuine burst on emergencies) -- they are NOT derived from
BDO-SKIN's own compute cost (the source report doesn't state one; it's
a physics/control simulation, not a hardware specification). Every
constant is named and adjustable below.
"""

from worker import Task
from models.sensor_events import (
    NUM_FBG_CHANNELS, generate_scenario, SensorWindow,
)

MESH_CELLS = 60 * 40  # full reconstructed field, 2400 cells (BDO-SKIN report Section 2.1)

# --- MAC-cost assumptions (stimulus generator, not a BDO-SKIN spec) ------
REFLEX_OPS_PER_CHANNEL = 16       # per-channel LIF-style integrate/threshold/reset, per window
BHS_HISTORY_LEN = 15              # samples used for Bat's per-cell trend fit
BAT_OPS_PER_CELL = 4
HERMIT_CRAB_ACTIONS = 5
HERMIT_CRAB_OPS_PER_ACTION_PER_CELL = 4
SQUID_OBJECTIVES = 4
SQUID_OPS = 100
EMERGENCY_OPS_PER_CHANNEL = 40    # elevated urgency-mode reflex processing

# --- Priorities (lower = more urgent, matches existing convention) -------
PRIORITY_REFLEX = 5
PRIORITY_BHS = 3
PRIORITY_EMERGENCY = -1000

# --- Deadline slack ---------------------------------------------------
EMERGENCY_DEADLINE_SLACK = 1.05   # tight: genuine sub-second hard-RT requirement, minimal margin


def _tile_macs(total_macs, num_tiles, sparsity, layer_name, weight_block_id, priority,
                deadline_slack_override=None, is_critical=False):
    tasks = []
    if num_tiles <= 0:
        return tasks
    base = total_macs // num_tiles
    rem = total_macs % num_tiles
    for i in range(num_tiles):
        macs = base + (1 if i < rem else 0)
        if macs <= 0:
            continue
        t = Task(layer_name, macs, sparsity, weight_block_id=weight_block_id, priority=priority)
        t.deadline_slack_override = deadline_slack_override
        t.is_critical = is_critical
        tasks.append(t)
    return tasks


def build_bdo_skin_workload(num_workers: int, scenario: str = "burst_anomaly",
                             num_windows: int = 150, seed: int = 0, bhs_period: int = 10,
                             sparsity_reflex: float = 0.10, sparsity_bhs: float = 0.05):
    """Returns (layers, metadata):
      layers   -- list of (layer_name, [Task, ...]), ready for
                  MasterControlCore.load_program()
      metadata -- dict with the SensorWindow list, critical window
                  indices, and a map from window index -> the layer
                  names dispatched for it, so a benchmark script can
                  trace end-to-end latency for specific events without
                  needing any changes to the simulator's own reporting.
    """
    windows = generate_scenario(scenario, num_windows, seed=seed)

    reflex_tiles = max(1, num_workers // 3)
    bhs_tiles = max(1, num_workers // 2)
    emergency_tiles = num_workers  # all-hands-on-deck for a critical burst

    layers = []
    window_layer_map = {}

    for w in windows:
        this_window_layers = []

        # --- Reflex layer: every window, all 600 channels -----------
        severity_load_factor = 1.0 + (w.anomaly_severity if w.anomaly_active else 0.0)
        reflex_macs = int(NUM_FBG_CHANNELS * REFLEX_OPS_PER_CHANNEL * severity_load_factor)
        reflex_name = f"reflex_w{w.index}"
        reflex_tasks = _tile_macs(
            reflex_macs, reflex_tiles, sparsity_reflex, reflex_name,
            weight_block_id="reflex_kernel_params", priority=PRIORITY_REFLEX,
        )
        layers.append((reflex_name, reflex_tasks))
        this_window_layers.append(reflex_name)

        # --- BHS cognitive layers: periodic, or right after a critical
        # window (the system re-evaluates its objective weights and
        # candidate actions immediately once something urgent has
        # happened, not just on the routine cadence) --------------------
        dispatch_bhs = (w.index % bhs_period == 0) or w.is_critical
        if dispatch_bhs:
            bat_macs = MESH_CELLS * BHS_HISTORY_LEN * BAT_OPS_PER_CELL
            bat_name = f"bat_forecast_w{w.index}"
            bat_tasks = _tile_macs(
                bat_macs, bhs_tiles, sparsity_bhs, bat_name,
                weight_block_id="bat_forecast_params", priority=PRIORITY_BHS,
            )
            layers.append((bat_name, bat_tasks))
            this_window_layers.append(bat_name)

            hc_macs = MESH_CELLS * HERMIT_CRAB_ACTIONS * HERMIT_CRAB_OPS_PER_ACTION_PER_CELL
            hc_name = f"hermit_crab_w{w.index}"
            hc_tasks = _tile_macs(
                hc_macs, bhs_tiles, sparsity_bhs, hc_name,
                weight_block_id="hermit_crab_params", priority=PRIORITY_BHS,
            )
            layers.append((hc_name, hc_tasks))
            this_window_layers.append(hc_name)

            squid_macs = SQUID_OBJECTIVES * SQUID_OPS
            squid_name = f"squid_reweight_w{w.index}"
            squid_tasks = _tile_macs(
                squid_macs, 1, sparsity_bhs, squid_name,
                weight_block_id="squid_params", priority=PRIORITY_BHS,
            )
            layers.append((squid_name, squid_tasks))
            this_window_layers.append(squid_name)

        # --- Emergency/Reflex layer: only on critical windows --------
        # Represents "drop everything, full urgent reassessment now":
        # elevated full-mesh reflex processing PLUS an immediate full
        # Bat+HermitCrab+Squid re-evaluation (not just a local patch --
        # a real emergency response can't assume the danger is confined
        # to where it was first detected). This is deliberately as heavy
        # as a full periodic BHS dispatch, because that's what "the
        # system drops its routine cadence and re-evaluates everything
        # right now" actually costs.
        if w.is_critical:
            emergency_reflex_macs = NUM_FBG_CHANNELS * EMERGENCY_OPS_PER_CHANNEL
            emergency_bat_macs = MESH_CELLS * BHS_HISTORY_LEN * BAT_OPS_PER_CELL
            emergency_hc_macs = MESH_CELLS * HERMIT_CRAB_ACTIONS * HERMIT_CRAB_OPS_PER_ACTION_PER_CELL
            emergency_squid_macs = SQUID_OBJECTIVES * SQUID_OPS
            emergency_macs = (emergency_reflex_macs + emergency_bat_macs
                               + emergency_hc_macs + emergency_squid_macs)
            emergency_name = f"emergency_w{w.index}"
            emergency_tasks = _tile_macs(
                emergency_macs, emergency_tiles, 0.0, emergency_name,
                weight_block_id="emergency_kernel_params", priority=PRIORITY_EMERGENCY,
                deadline_slack_override=EMERGENCY_DEADLINE_SLACK, is_critical=True,
            )
            layers.append((emergency_name, emergency_tasks))
            this_window_layers.append(emergency_name)

        window_layer_map[w.index] = this_window_layers

    metadata = {
        "windows": windows,
        "critical_window_indices": [w.index for w in windows if w.is_critical],
        "window_layer_map": window_layer_map,
        "scenario": scenario,
        "num_workers": num_workers,
        "reflex_tiles": reflex_tiles,
        "bhs_tiles": bhs_tiles,
        "emergency_tiles": emergency_tiles,
    }
    return layers, metadata


def total_ideal_macs(layers):
    return sum(t.mac_count for _, tasks in layers for t in tasks)
