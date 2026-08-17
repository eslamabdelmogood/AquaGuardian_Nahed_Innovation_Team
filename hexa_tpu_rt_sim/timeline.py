"""
timeline.py
===========
Records per-cycle busy/idle activity for workers, DMA, and the AXI bus
into fixed-width bins (config.TIMELINE_BIN_WIDTH_CYCLES cycles per
bin), so a multi-hundred-thousand-cycle run can still be rendered as a
compact ASCII gantt/heatmap without storing a value per cycle.

Each bin stores the *fraction* of its cycles that a given unit was
busy/active/contended, then render_ascii() maps that fraction to a
unicode block-height character so bubbles (dips in occupancy) are
visually obvious.
"""

BLOCKS = " ▁▂▃▄▅▆▇█"  # 9 levels, index 0..8


def _level_char(fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    idx = round(fraction * (len(BLOCKS) - 1))
    return BLOCKS[idx]


class TimelineRecorder:
    def __init__(self, config, num_workers: int):
        self.cfg = config
        self.bin_width = max(1, config.TIMELINE_BIN_WIDTH_CYCLES)
        self.num_workers = num_workers

        self._cycle_in_bin = 0
        self._worker_busy_acc = [0] * num_workers
        self._dma_active_acc = 0
        self._axi_contended_acc = 0
        self._axi_util_acc = 0.0

        self.worker_bins = [[] for _ in range(num_workers)]
        self.dma_bins = []
        self.axi_contention_bins = []
        self.axi_util_bins = []

    def record_cycle(self, worker_busy_flags, dma_active, axi_demand_bytes, axi_supplied_bytes, axi_width):
        for i, busy in enumerate(worker_busy_flags):
            if busy:
                self._worker_busy_acc[i] += 1
        if dma_active:
            self._dma_active_acc += 1
        if axi_demand_bytes > axi_width:
            self._axi_contended_acc += 1
        if axi_width > 0:
            self._axi_util_acc += min(1.0, axi_supplied_bytes / axi_width)

        self._cycle_in_bin += 1
        if self._cycle_in_bin >= self.bin_width:
            self._flush_bin()

    def _flush_bin(self):
        n = self._cycle_in_bin
        if n == 0:
            return
        for i in range(self.num_workers):
            self.worker_bins[i].append(self._worker_busy_acc[i] / n)
        self.dma_bins.append(self._dma_active_acc / n)
        self.axi_contention_bins.append(self._axi_contended_acc / n)
        self.axi_util_bins.append(self._axi_util_acc / n)

        self._cycle_in_bin = 0
        self._worker_busy_acc = [0] * self.num_workers
        self._dma_active_acc = 0
        self._axi_contended_acc = 0
        self._axi_util_acc = 0.0

    def finalize(self):
        if self._cycle_in_bin > 0:
            self._flush_bin()

    def render_ascii(self, max_columns: int = 160) -> str:
        """Render a compact gantt. If there are more bins than
        max_columns, downsample further by averaging groups of bins."""
        self.finalize()
        n_bins = len(self.dma_bins)
        if n_bins == 0:
            return "(no cycles recorded)"

        def downsample(series):
            if n_bins <= max_columns:
                return series
            group = n_bins / max_columns
            out = []
            for c in range(max_columns):
                lo = int(c * group)
                hi = max(lo + 1, int((c + 1) * group))
                chunk = series[lo:hi]
                out.append(sum(chunk) / len(chunk) if chunk else 0.0)
            return out

        lines = []
        cycles_per_col = self.bin_width * (n_bins / min(n_bins, max_columns))
        lines.append(f"Timeline -- each column ~ {cycles_per_col:.0f} cycles "
                      f"({min(n_bins, max_columns)} columns, {n_bins * self.bin_width} total cycles)")
        lines.append("Legend: " + BLOCKS[1] + "=low occupancy .. " + BLOCKS[-1] + "=saturated, blank=idle")
        lines.append("")

        for i in range(self.num_workers):
            series = downsample(self.worker_bins[i])
            row = "".join(_level_char(v) for v in series)
            lines.append(f"Worker{i:<2} |{row}|")

        dma_row = "".join(_level_char(v) for v in downsample(self.dma_bins))
        lines.append(f"DMA     |{dma_row}|")

        axi_util_row = "".join(_level_char(v) for v in downsample(self.axi_util_bins))
        lines.append(f"AXI-Use |{axi_util_row}|")

        axi_cont_row = "".join(_level_char(v) for v in downsample(self.axi_contention_bins))
        lines.append(f"AXI-Cont|{axi_cont_row}|  <- fraction of cycles where demand > bus width")

        return "\n".join(lines)
