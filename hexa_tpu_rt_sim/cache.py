"""
cache.py
========
A small shared cache sitting in front of the AXI data path, keyed per
weight block (in this model: one block per layer, since output-channel
tiling means every tile of the same layer reads the same kernel
weights). LRU eviction over CACHE_CAPACITY_BLOCKS blocks.

NOT in the spec. Modeled because weight reuse across tiles is normal
in real DNN accelerators and materially changes AXI traffic -- without
it, every single task would re-fetch weights the previous task on a
different worker just loaded, which is unrealistically pessimistic.
"""

from collections import OrderedDict


class WeightCache:
    def __init__(self, config):
        self.cfg = config
        self.capacity = config.CACHE_CAPACITY_BLOCKS
        self.enabled = config.CACHE_ENABLED
        self._store = OrderedDict()  # block_id -> True, ordered by recency
        self.hits = 0
        self.misses = 0

    def access(self, block_id) -> bool:
        """Look up block_id; returns True on hit (and marks it
        most-recently-used), False on miss (and inserts it, evicting
        the LRU block if at capacity)."""
        if not self.enabled:
            self.misses += 1
            return False

        if block_id in self._store:
            self._store.move_to_end(block_id)
            self.hits += 1
            return True

        self.misses += 1
        self._store[block_id] = True
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict LRU
        return False

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
