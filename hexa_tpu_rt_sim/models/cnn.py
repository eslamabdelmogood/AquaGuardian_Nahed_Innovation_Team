"""
models/cnn.py
=============
Builds a small, realistic CNN as a list of (layer_name, [Task, ...])
that the Master/Scheduler can execute. Convolutions are expressed via
their im2col-equivalent MAC count (output_elems * kernel_elems *
in_channels), which is the standard way to map a conv layer onto a
systolic matmul engine. Each layer is split into tiles so it can be
spread across however many workers the config specifies.

This is a *workload generator*, not the neural network itself -- no
actual tensors are computed, only the MAC-cycle cost profile a real
CNN of this shape would present to the hardware.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import Task


def _conv_layer_macs(in_c, out_c, k, out_h, out_w):
    return out_h * out_w * out_c * in_c * k * k


def _tile_layer(layer_name, total_macs, sparsity, num_tiles, priority=0, tiling_strategy="spatial"):
    """Split a layer's MACs into `num_tiles` roughly equal Task objects
    so independent tiles can run on different workers concurrently.

    tiling_strategy controls the cache key:
      "spatial"         -- all tiles share weight_block_id = layer_name
                            (models same-weights-different-region tiling)
      "output_channel"  -- each tile gets a unique weight_block_id
                            (models different-weight-slice-per-tile
                            tiling; no cache reuse is possible)
    """
    tasks = []
    base = total_macs // num_tiles
    remainder = total_macs % num_tiles
    for i in range(num_tiles):
        macs = base + (1 if i < remainder else 0)
        if macs > 0:
            if tiling_strategy == "output_channel":
                block_id = f"{layer_name}#tile{i}"
            else:
                block_id = layer_name
            tasks.append(Task(layer_name, macs, sparsity, weight_block_id=block_id,
                               priority=priority))
    return tasks


def build_tiny_cnn(num_workers: int, sparsity: float = 0.30, tiles_per_worker: int = 1,
                    tiling_strategy: str = "spatial"):
    """A small 5-layer CNN (similar in spirit to a MobileNet-style edge
    vision backbone stem) used as the Phase-1 benchmark workload.

    Layer shapes (in_c, out_c, kernel, out_h, out_w):
      conv1: 3   ->  16, 3x3, 112x112   (stem, stride 2 from 224x224 input)
      conv2: 16  ->  32, 3x3, 56x56
      conv3: 32  ->  64, 3x3, 28x28
      conv4: 64  -> 128, 3x3, 14x14
      conv5: 128 -> 128, 3x3, 14x14  (depth-preserving refinement layer)
    """
    shapes = [
        ("conv1_3x3_s2",  3,  16, 3, 112, 112),
        ("conv2_3x3",     16, 32, 3, 56, 56),
        ("conv3_3x3",     32, 64, 3, 28, 28),
        ("conv4_3x3",     64, 128, 3, 14, 14),
        ("conv5_3x3",     128, 128, 3, 14, 14),
    ]

    layers = []
    num_tiles = max(num_workers, 1) * max(tiles_per_worker, 1)
    for layer_idx, (name, in_c, out_c, k, oh, ow) in enumerate(shapes):
        macs = _conv_layer_macs(in_c, out_c, k, oh, ow)
        # Priority: earlier layers = lower number = more critical, since a
        # missed early layer stalls everything downstream of it. This is
        # a workload-generator choice, not a hardware property.
        tasks = _tile_layer(name, macs, sparsity, num_tiles=num_tiles, priority=layer_idx,
                             tiling_strategy=tiling_strategy)
        layers.append((name, tasks))
    return layers


def total_ideal_macs(layers):
    return sum(t.mac_count for _, tasks in layers for t in tasks)
