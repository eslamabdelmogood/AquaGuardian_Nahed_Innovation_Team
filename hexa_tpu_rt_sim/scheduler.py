"""
scheduler.py
============
Hardware task scheduler. Pulls tasks off the queue built by the Master
and assigns them to free Micro-TPUs, subject to:

  * SCHEDULER_POLICY -- which queued task gets picked next when more
    than one is ready and more than one worker is free:
      "fifo"     -- dispatch order == arrival order (Phase 1 default)
      "priority" -- lowest Task.priority value first
      "edf"      -- earliest deadline first, using a *provisional*
                    deadline estimated at enqueue time (ideal-case
                    cycles from now); ties broken by priority
  * HOT_BANK_READ_PORTS -- how many workers can be handed a *new* tile
    (start a fresh task) in the same cycle -- an SRAM address/port
    limit, separate from the AXI bus's per-cycle data bandwidth.
  * WeightCache -- looked up per task before assignment; a hit means
    the task streams unconstrained (no AXI contention), a miss means
    it competes for AXI bandwidth every cycle it runs.

Also triggers DMA prefetch of the next layer once the current layer's
tasks have all been dispatched, and requests the ping-pong bank swap
once that prefetch completes. Prefetch requests can be rejected by the
DMA if its queue is full (see dma.py) -- the scheduler just retries.
"""

from collections import deque


class Scheduler:
    def __init__(self, config, workers, memory, dma, cache):
        self.cfg = config
        self.workers = workers
        self.memory = memory
        self.dma = dma
        self.cache = cache
        self.queue = deque()
        self.dispatched_count = 0
        self.policy = config.SCHEDULER_POLICY

        # Layer-prefetch bookkeeping
        self._current_layer_tasks_total = 0
        self._current_layer_tasks_dispatched = 0
        self._prefetch_started_for_layer = None
        self._current_layer_name = None
        self._prefetch_pending_bank = None  # set if enqueue was rejected, retry target

    def enqueue_layer(self, tasks, layer_name):
        for t in tasks:
            self.queue.append(t)
        self._current_layer_tasks_total = len(tasks)
        self._current_layer_tasks_dispatched = 0
        self._prefetch_started_for_layer = None
        self._current_layer_name = layer_name

    def _pick_next_task(self):
        """Remove and return the task the policy says should run next.
        Assumes self.queue is non-empty."""
        if self.policy == "fifo" or len(self.queue) == 1:
            return self.queue.popleft()

        if self.policy == "priority":
            best = min(self.queue, key=lambda t: t.priority)
        elif self.policy == "edf":
            # Provisional deadline: task.priority doubles as a coarse
            # "criticality class" here since true per-task deadlines
            # aren't known until a worker's ideal cycle-cost is
            # computed on assignment. EDF here approximates "most
            # urgent class first, FIFO within a class" -- a real EDF
            # would need cost estimation before assignment, which is
            # a Phase 2+ scheduler-cost-model item.
            best = min(self.queue, key=lambda t: (t.priority, t.id))
        else:
            best = self.queue[0]

        self.queue.remove(best)
        return best

    def step(self, current_cycle: int):
        started_this_cycle = 0
        for w in self.workers:
            if w.is_free and self.queue:
                if started_this_cycle >= self.cfg.HOT_BANK_READ_PORTS:
                    w.wait_on_memory()
                    continue

                task = self._pick_next_task()
                task.issue_cycle = current_cycle
                cache_hit = self.cache.access(task.weight_block_id)

                # Estimate how many workers will be contending for AXI
                # once this task starts: currently-busy non-cache-hit
                # workers, plus this one if it's also a miss.
                concurrent = sum(
                    1 for other in self.workers
                    if other.current_task is not None and not other.current_task.cache_hit
                )
                if not cache_hit:
                    concurrent += 1

                if self.cfg.DEADLINE_LOOKAHEAD_ENABLED and not cache_hit:
                    # Phase 7: also count sibling tasks still waiting in
                    # this same dispatch wave that are guaranteed to also
                    # miss cache (genuinely different weight block, e.g.
                    # other attention heads) -- they haven't started yet,
                    # but they WILL be competing for AXI within the next
                    # few cycles, and a deadline set blind to that is
                    # exactly what caused Phase 4's QKV projection misses.
                    concurrent += sum(
                        1 for qt in self.queue
                        if qt.layer_name == task.layer_name and qt.weight_block_id != task.weight_block_id
                    )

                w.assign(task, current_cycle, cache_hit=cache_hit,
                         concurrent_axi_requesters=concurrent)

                self.dispatched_count += 1
                self._current_layer_tasks_dispatched += 1
                started_this_cycle += 1

                # Once the whole layer has been handed out, start
                # prefetching the *next* layer into the inactive bank.
                if (self._current_layer_tasks_dispatched >= self._current_layer_tasks_total
                        and self._prefetch_started_for_layer != self._current_layer_name):
                    if self.dma.enqueue_prefetch(self.memory.loading_bank, self._current_layer_name,
                                                  memory=self.memory):
                        self._prefetch_started_for_layer = self._current_layer_name
                    else:
                        self._prefetch_pending_bank = self.memory.loading_bank

        # Retry a prefetch that was previously rejected for lack of
        # queue depth.
        if (self._prefetch_pending_bank is not None
                and self._prefetch_started_for_layer != self._current_layer_name):
            if self.dma.enqueue_prefetch(self._prefetch_pending_bank, self._current_layer_name,
                                          memory=self.memory):
                self._prefetch_started_for_layer = self._current_layer_name
                self._prefetch_pending_bank = None

        # If the loading bank just finished filling, request the swap.
        loading_bank_state = self.memory.bank_state[self.memory.loading_bank].value
        if loading_bank_state == "idle" and self._prefetch_started_for_layer is not None:
            self.memory.request_switch()

    @property
    def pending(self) -> int:
        return len(self.queue)

    @property
    def all_workers_idle(self) -> bool:
        return all(w.is_free for w in self.workers)
