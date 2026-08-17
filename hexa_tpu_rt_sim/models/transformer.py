"""
models/transformer.py
======================
Workload generator for a small Transformer encoder, built specifically
as Phase 4's stress test: does the Phase 2 memory-aware deadline fix
(validated only on the CNN benchmark's uniform, static demand) survive
a workload that is genuinely heterogeneous? Unlike models/cnn.py,
which tiles each layer into num_tiles roughly-equal, always-reusable
chunks, this workload has properties no toggle in Phase 1-3 produced:

  - QKV projection is tiled per attention head. Each head needs a
    genuinely different weight slice -- no cross-tile reuse, like
    Phase 2's "output_channel" tiling, but here it's structural, not a
    config flag.
  - Attention scores (Q.K^T) and attention-weighted values (softmax(.)V)
    multiply two ACTIVATION tensors, not an activation by a stored
    weight tensor. There is nothing to cache -- weight_block_id is
    always unique for these, by construction.
  - Output projection and the two FFN layers ARE spatially tiled (like
    CNN benchmark), so weight reuse stays high there.
  - Per-op-type MAC counts differ by orders of magnitude: attention
    scores/values scale with seq_len^2, FFN scales with seq_len*d_ff.
    That makes AXI demand burst layer-to-layer instead of the CNN
    benchmark's roughly steady demand.
  - num_heads is typically LESS than num_workers (e.g. 8 heads on a
    10-worker chip), so attention phases structurally can't use every
    worker -- a parallelism-limited bubble that has nothing to do with
    memory bandwidth, and that the CNN benchmark could never surface.
  - Optional load imbalance: FFN tiles can be skewed in size to model
    uneven per-worker load from batch padding / variable sequence
    lengths in real transformer inference.

Still a MAC-cost profile generator, not an actual transformer --
softmax/layernorm aren't modeled (no MACs), only the matmuls.
"""

import random
from worker import Task


def _matmul_macs(m, k, n):
    return m * k * n


def _make_tasks(layer_name, total_macs, sparsity, num_tiles, priority,
                 weight_block_id_fn, imbalance=False, seed=0, critical_indices=None,
                 critical_priority=-1000):
    """critical_indices, if given, is a set of tile indices within this
    op that get priority=critical_priority instead of the uniform
    per-layer priority -- the only way to give the scheduler genuine
    choice among differently-prioritized ready tasks (see Phase 6:
    without this, every task dispatched together shares one priority
    value and fifo/priority/edf are indistinguishable)."""
    tasks = []
    if num_tiles <= 0:
        return tasks

    if imbalance:
        rng = random.Random(seed)
        weights = [rng.uniform(0.5, 1.5) for _ in range(num_tiles)]
        total_w = sum(weights)
        macs_list = [int(total_macs * w / total_w) for w in weights]
        diff = total_macs - sum(macs_list)
        macs_list[0] += diff  # absorb rounding remainder
    else:
        base = total_macs // num_tiles
        rem = total_macs % num_tiles
        macs_list = [base + (1 if i < rem else 0) for i in range(num_tiles)]

    critical_indices = critical_indices or set()
    for i, macs in enumerate(macs_list):
        if macs <= 0:
            continue
        block_id = weight_block_id_fn(i)
        task_priority = critical_priority if i in critical_indices else priority
        task = Task(layer_name, macs, sparsity, weight_block_id=block_id, priority=task_priority)
        task.is_critical = i in critical_indices
        tasks.append(task)
    return tasks


def build_tiny_transformer(num_workers: int, seq_len: int = 128, d_model: int = 256,
                            num_heads: int = 8, d_ff: int = 1024, num_layers: int = 2,
                            sparsity_attn: float = 0.05, sparsity_ffn: float = 0.20,
                            load_imbalance: bool = True, critical_heads_per_layer: int = 0):
    """critical_heads_per_layer marks that many QKV-projection heads
    (indices 0..N-1) per block as high-priority (Phase 6: safety-
    critical work sharing the chip with best-effort work), instead of
    every task in a layer sharing one priority value. See models/
    transformer.py module docstring and README Phase 6 section."""
    assert d_model % num_heads == 0, "d_model must divide evenly by num_heads"
    d_head = d_model // num_heads
    spatial_tiles = max(num_workers, 1)
    critical_indices = set(range(min(critical_heads_per_layer, num_heads)))

    layers = []
    priority = 0
    for layer_idx in range(num_layers):
        lp = f"block{layer_idx}"

        # --- QKV projection: tiled per head, no cross-head weight reuse ---
        total_qkv_macs = _matmul_macs(seq_len, d_model, 3 * d_model)
        qkv_tasks = _make_tasks(
            f"{lp}_qkv_proj", total_qkv_macs, sparsity_attn, num_heads, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_qkv_head{i}",
            critical_indices=critical_indices,
        )
        layers.append((f"{lp}_qkv_proj", qkv_tasks))
        priority += 1

        # --- Attention scores Q.K^T per head: activation x activation, never cacheable ---
        total_score_macs = _matmul_macs(seq_len, d_head, seq_len) * num_heads
        score_tasks = _make_tasks(
            f"{lp}_attn_scores", total_score_macs, 0.0, num_heads, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_scores_head{i}_unique_{layer_idx}",
        )
        layers.append((f"{lp}_attn_scores", score_tasks))
        priority += 1

        # --- Attention * V per head: also activation x activation, never cacheable ---
        total_av_macs = _matmul_macs(seq_len, seq_len, d_head) * num_heads
        av_tasks = _make_tasks(
            f"{lp}_attn_v", total_av_macs, 0.0, num_heads, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_av_head{i}_unique_{layer_idx}",
        )
        layers.append((f"{lp}_attn_v", av_tasks))
        priority += 1

        # --- Output projection: spatially tiled, weight reuse across tiles ---
        out_proj_macs = _matmul_macs(seq_len, d_model, d_model)
        out_tasks = _make_tasks(
            f"{lp}_out_proj", out_proj_macs, sparsity_attn, spatial_tiles, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_out_proj",
        )
        layers.append((f"{lp}_out_proj", out_tasks))
        priority += 1

        # --- FFN1: d_model -> d_ff, spatially tiled, optional load imbalance ---
        ffn1_macs = _matmul_macs(seq_len, d_model, d_ff)
        ffn1_tasks = _make_tasks(
            f"{lp}_ffn1", ffn1_macs, sparsity_ffn, spatial_tiles, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_ffn1",
            imbalance=load_imbalance, seed=layer_idx * 10 + 1,
        )
        layers.append((f"{lp}_ffn1", ffn1_tasks))
        priority += 1

        # --- FFN2: d_ff -> d_model, spatially tiled, optional load imbalance ---
        ffn2_macs = _matmul_macs(seq_len, d_ff, d_model)
        ffn2_tasks = _make_tasks(
            f"{lp}_ffn2", ffn2_macs, sparsity_ffn, spatial_tiles, priority,
            weight_block_id_fn=lambda i, lp=lp: f"{lp}_ffn2",
            imbalance=load_imbalance, seed=layer_idx * 10 + 2,
        )
        layers.append((f"{lp}_ffn2", ffn2_tasks))
        priority += 1

    return layers


def total_ideal_macs(layers):
    return sum(t.mac_count for _, tasks in layers for t in tasks)


def layer_mac_profile(layers):
    """Returns [(layer_name, total_macs, num_tasks), ...] -- useful for
    directly inspecting how bursty the per-layer demand actually is."""
    return [(name, sum(t.mac_count for t in tasks), len(tasks)) for name, tasks in layers]
