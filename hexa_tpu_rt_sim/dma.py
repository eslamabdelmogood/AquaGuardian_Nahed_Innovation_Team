"""
dma.py
======
Models the DDR -> DMA -> SRAM pipeline (Phase 1.5 addition; DDR is not
in the spec, see config.py). Each transfer has two phases:

  1. DDR access latency (fixed cycles before any data arrives)
  2. Burst transfer at the DDR interface's bytes/cycle bandwidth

The DMA also owns a prefetch queue (depth = config.PREFETCH_QUEUE_DEPTH)
so the scheduler can queue up the *next* layer's load ahead of time
instead of only starting it the instant the current layer drains --
this is what actually lets ping-pong hide DDR latency behind compute.
If the queue is ever full when the scheduler wants to prefetch further
ahead, that prefetch request is rejected and retried later -- a real,
counted limitation rather than infinite lookahead.
"""

from collections import deque
from memory import BankState


class DMAController:
    def __init__(self, config):
        self.cfg = config
        self.queue = deque()          # pending (bank, layer_name) descriptors
        self.active_transfer = None   # dict describing the in-flight transfer
        self.rejected_enqueues = 0

        # Stats
        self.total_active_cycles = 0
        self.ddr_latency_cycles_spent = 0
        self.burst_cycles_spent = 0
        self.transfers_completed = 0
        self.total_bytes_transferred = 0

    def enqueue_prefetch(self, bank: str, layer_name: str, memory=None) -> bool:
        """Scheduler asks the DMA to start loading `bank` with the data
        for `layer_name`. Returns False (and counts a rejection) if the
        prefetch queue is already full -- i.e. real backpressure.

        The bank is marked LOADING immediately (reserved) even though
        the transfer itself may not start for a few cycles if the DMA
        engine is still busy with a prior transfer. This closes a real
        race: if the bank only flipped to LOADING once the DMA engine
        got around to it, the scheduler could see it as still IDLE and
        request an early swap into a bank whose transfer hadn't
        actually started yet -- a genuine bus conflict."""
        if len(self.queue) >= self.cfg.PREFETCH_QUEUE_DEPTH:
            self.rejected_enqueues += 1
            return False
        self.queue.append((bank, layer_name))
        if memory is not None:
            from memory import BankState
            memory.bank_state[bank] = BankState.LOADING
        return True

    def step(self, memory) -> dict:
        active = False

        # Start the next queued transfer if the DMA engine is free.
        if self.active_transfer is None and self.queue:
            bank, layer_name = self.queue.popleft()
            total_bytes = self.cfg.HOT_BANK_CAPACITY_ELEMS * self.cfg.BYTES_PER_ELEMENT
            self.active_transfer = {
                "bank": bank,
                "layer_name": layer_name,
                "ddr_latency_remaining": self.cfg.DDR_LATENCY_CYCLES,
                "bytes_remaining": total_bytes,
            }
            memory.bank_state[bank] = BankState.LOADING

        if self.active_transfer is not None:
            active = True
            t = self.active_transfer
            if t["ddr_latency_remaining"] > 0:
                t["ddr_latency_remaining"] -= 1
                self.ddr_latency_cycles_spent += 1
            else:
                transferred = min(self.cfg.DDR_BANDWIDTH_BYTES_PER_CYCLE, t["bytes_remaining"])
                t["bytes_remaining"] -= transferred
                self.burst_cycles_spent += 1
                self.total_bytes_transferred += transferred
                if t["bytes_remaining"] <= 0:
                    memory.bank_state[t["bank"]] = BankState.IDLE
                    self.transfers_completed += 1
                    self.active_transfer = None

        if active:
            self.total_active_cycles += 1

        return {
            "dma_active": active,
            "queue_depth": len(self.queue),
            "in_ddr_latency": (self.active_transfer is not None
                                and self.active_transfer["ddr_latency_remaining"] > 0),
        }
