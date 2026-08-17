"""
memory.py
=========
Models the HEXA-TPU-RT "Zero-Bus Conflict Shared Memory" subsystem:

  * PingPongMemory   -- dual-bank (A/B) hot input SRAM. One bank feeds
                         the workers while the DMA fills the other.
  * ColdOutputMemory -- one dedicated write bank per worker, so by
                         construction workers never collide on writes.
                         The class still *checks* for illegal
                         concurrent access so a bug (or a future
                         design change, e.g. sharing banks) shows up
                         as a detected conflict rather than silently
                         passing.

Both classes expose a `.state` snapshot each cycle for the cycle-by-
cycle trace, and both count real conflicts/stalls rather than
returning canned numbers.
"""

from enum import Enum


class BankState(Enum):
    IDLE = "idle"
    READING = "reading"
    LOADING = "loading"


class PingPongMemory:
    """Dual-bank hot input SRAM (Bank A / Bank B).

    NOTE (Phase 1.5 change): DMAController now owns the load-progress
    state machine directly (DDR latency + burst transfer) and sets
    bank_state on this object itself. This class no longer runs its
    own cycle-count-based loader -- it keeps the bank_state dict as
    shared, observable state and focuses on what it's actually
    responsible for: active/loading bank bookkeeping, the swap, and
    conflict detection.
    """

    def __init__(self, config):
        self.cfg = config
        self.active_bank = "A"          # bank currently being read by workers
        self.loading_bank = "B"         # bank currently being filled by DMA
        self.bank_state = {"A": BankState.IDLE, "B": BankState.IDLE}
        self.conflicts = 0
        self.switch_count = 0
        self._pending_switch = False

    def request_switch(self):
        """Scheduler asks to swap active/loading banks once the
        currently-loading bank is full. Executed on the next step()."""
        self._pending_switch = True

    def step(self):
        """Advance memory state by one cycle. Returns dict for tracing."""
        # Conflict check: it would be illegal for the *active* bank to
        # simultaneously be in LOADING state (workers reading while DMA
        # writes the same bank). By construction this never happens if
        # the DMA is only ever pointed at the loading_bank, but we
        # verify it explicitly every cycle so any future scheduling bug
        # is caught rather than hidden.
        if self.bank_state[self.active_bank] == BankState.LOADING:
            self.conflicts += 1

        # Mark active bank as READING for the trace (workers consume it)
        if self.bank_state[self.active_bank] == BankState.IDLE:
            self.bank_state[self.active_bank] = BankState.READING

        # Perform the swap if requested and the loading bank has finished
        if self._pending_switch and self.bank_state[self.loading_bank] == BankState.IDLE:
            self.active_bank, self.loading_bank = self.loading_bank, self.active_bank
            self.switch_count += 1
            self._pending_switch = False
            # new active bank becomes READING next cycle naturally

        return {
            "active_bank": self.active_bank,
            "loading_bank": self.loading_bank,
            "bank_A_state": self.bank_state["A"].value,
            "bank_B_state": self.bank_state["B"].value,
        }


class ColdOutputMemory:
    """One dedicated output bank per worker -- collisions should be
    structurally impossible; this class detects them anyway."""

    def __init__(self, config, num_workers: int):
        self.cfg = config
        self.num_workers = num_workers
        self.write_owner = {}  # bank_id -> worker_id writing this cycle
        self.conflicts = 0

    def write(self, worker_id: int, bank_id: int):
        if bank_id in self.write_owner and self.write_owner[bank_id] != worker_id:
            self.conflicts += 1
        self.write_owner[bank_id] = worker_id

    def step_reset(self):
        self.write_owner = {}
