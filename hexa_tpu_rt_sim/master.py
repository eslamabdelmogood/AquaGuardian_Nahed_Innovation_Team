"""
master.py
=========
Models the RISC-V Master Control Core: it owns the Hardware Task
Dispatcher (feeds whole layers to the Scheduler) and the Hard
Real-Time Deadline Monitor (watches every in-flight task and triggers
preemption if a worker blows through its deadline).
"""


class MasterControlCore:
    def __init__(self, config, scheduler, workers):
        self.cfg = config
        self.scheduler = scheduler
        self.workers = workers
        self.layers = []          # list of (layer_name, [Task, ...])
        self._layer_idx = 0
        self.preemptions = 0

    def load_program(self, layers):
        """layers: list of (layer_name, list[Task])"""
        self.layers = layers
        self._layer_idx = 0

    def dispatch_next_layer_if_ready(self):
        """Dispatcher: hand the next layer to the scheduler once the
        previous layer's tasks have all been consumed from the queue."""
        if self._layer_idx >= len(self.layers):
            return False
        if self.scheduler.pending == 0:
            layer_name, tasks = self.layers[self._layer_idx]
            self.scheduler.enqueue_layer(tasks, layer_name)
            self._layer_idx += 1
            return True
        return False

    def monitor_deadlines(self, current_cycle: int):
        """Hard real-time deadline monitor. If PREEMPTION_ENABLED and a
        worker's current task has blown its deadline, preempt it: the
        task is abandoned (counted as a miss) and the worker is freed
        immediately rather than left to run indefinitely."""
        if not self.cfg.PREEMPTION_ENABLED:
            return
        for w in self.workers:
            t = w.current_task
            if t is not None and current_cycle > t.deadline_cycle and not t.missed_deadline:
                t.missed_deadline = True
                w.deadline_misses += 1
                self.preemptions += 1
                # Free the worker immediately (hardware preemption)
                w.current_task = None
                from worker import WorkerState
                w.state = WorkerState.IDLE

    @property
    def program_complete(self) -> bool:
        return (self._layer_idx >= len(self.layers)
                and self.scheduler.pending == 0
                and self.scheduler.all_workers_idle)
