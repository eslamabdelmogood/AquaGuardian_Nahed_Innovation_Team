"""
export.py
=========
Phase 5: turns a Report (and its TimelineRecorder) into files someone
can actually open and share, instead of only terminal output.

  - export_json(report, path)       -- machine-readable, report.as_dict()
  - export_html(report, sim, path)  -- a single self-contained HTML file:
                                        summary table + power breakdown +
                                        an inline SVG timeline (no JS
                                        dependency, no matplotlib needed,
                                        so it always works)
  - export_timeline_png(sim, path)  -- matplotlib stacked-area PNG of the
                                        same timeline data the ASCII
                                        gantt shows, for anyone who wants
                                        a shareable plot instead of a
                                        terminal render. Optional: only
                                        called if matplotlib is
                                        importable, fails loudly with a
                                        clear message otherwise rather
                                        than silently no-op'ing.
"""

import json


def export_json(report, path: str):
    with open(path, "w") as f:
        json.dump(report.as_dict(), f, indent=2)
    return path


def _svg_timeline(sim, max_columns: int = 120) -> str:
    """Inline SVG rendering of the same binned timeline data as
    timeline.py's ASCII gantt -- kept dependency-free (no matplotlib
    required) so the HTML report always renders standalone."""
    tl = sim.timeline
    tl.finalize()
    n_bins = len(tl.dma_bins)
    if n_bins == 0:
        return "<p>(no cycles recorded)</p>"

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

    rows = []
    for i in range(tl.num_workers):
        rows.append((f"Worker{i}", downsample(tl.worker_bins[i]), "#4f8ef7"))
    rows.append(("DMA", downsample(tl.dma_bins), "#f7a94f"))
    rows.append(("AXI use", downsample(tl.axi_util_bins), "#4ff7a9"))
    rows.append(("AXI contention", downsample(tl.axi_contention_bins), "#f74f6f"))

    row_h = 18
    col_w = 4
    label_w = 100
    width = label_w + max_columns * col_w + 20
    height = len(rows) * row_h + 20

    svg = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="monospace" font-size="11">']
    for r, (label, series, color) in enumerate(rows):
        y = 10 + r * row_h
        svg.append(f'<text x="0" y="{y + row_h - 5}" fill="#333">{label}</text>')
        for c, v in enumerate(series):
            x = label_w + c * col_w
            opacity = max(0.06, min(1.0, v))
            svg.append(f'<rect x="{x}" y="{y}" width="{col_w}" height="{row_h - 2}" '
                       f'fill="{color}" opacity="{opacity:.2f}"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def export_html(report, sim, path: str, title: str = "HEXA-TPU-RT Simulation Report"):
    """Single self-contained HTML file -- no external assets, works by
    double-clicking it, no server or matplotlib required."""
    d = report.as_dict()
    power = report.power

    def row(label, value):
        return f"<tr><td>{label}</td><td>{value}</td></tr>"

    summary_rows = "\n".join([
        row("Workers", d["workers"]),
        row("Scheduler policy", d["scheduler_policy"]),
        row("Total cycles", f"{d['total_cycles']:,}"),
        row("Tasks completed", d["tasks_completed"]),
        row("Deadline misses", d["deadline_misses"]),
        row("Worker occupancy", f"{d['occupancy_pct']:.1f}%"),
        row("MAC utilization", f"{d['mac_utilization_pct']:.1f}%"),
        row("Average stall", f"{d['avg_stall_pct']:.1f}%"),
        row("AXI-starved cycles", d["axi_starved_cycles"]),
        row("Memory conflicts", d["memory_conflicts"]),
        row("Average latency", f"{d['avg_latency_ms']:.3f} ms"),
        row("Estimated throughput", f"{d['estimated_tops']:.3f} TOPS "
                                     f"(peak {d['peak_tops']:.3f})"),
        row("AXI utilization", f"{d['axi_utilization_pct']:.1f}%"),
        row("AXI contention rate", f"{d['axi_contention_rate_pct']:.1f}%"),
        row("Weight cache hit rate", f"{d['cache_hit_rate_pct']:.1f}%"),
    ])

    power_rows = "\n".join([
        row("MAC active energy", f"{power.mac_active_energy_nj:.2f} nJ"),
        row("MAC gated residual", f"{power.mac_gated_energy_nj:.2f} nJ"),
        row("MAC energy w/o gating (hypothetical)", f"{power.mac_energy_if_no_gating_nj:.2f} nJ"),
        row("SRAM/AXI data movement", f"{power.sram_axi_energy_nj:.2f} nJ"),
        row("DDR data movement", f"{power.ddr_energy_nj:.2f} nJ"),
        row("Static leakage", f"{power.static_energy_nj:.2f} nJ"),
        row("Average power", f"{power.average_power_mw:.2f} mW"),
        row("Efficiency", f"{power.tops_per_watt:.3f} TOPS/W ({power.watts_per_tops:.3f} W/TOPS)"),
        row("Sparsity-gating MAC power cut", f"{power.sparsity_gating_power_cut_pct:.1f}% "
                                              f"(spec claims 'up to 40%')"),
    ])

    svg_timeline = _svg_timeline(sim)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  h2 {{ font-size: 1.1em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
  td:first-child {{ color: #555; width: 60%; }}
  .note {{ color: #888; font-size: 0.85em; margin-top: 1em; }}
  .timeline-wrap {{ overflow-x: auto; border: 1px solid #eee; padding: 8px; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="note">Architectural simulation estimate -- not a measurement of real silicon.
Generated by the HEXA-TPU-RT simulator (Phase 5 export).</p>

<h2>Summary</h2>
<table>{summary_rows}</table>

<h2>Power model (Phase 3 -- assumptions, not measurements; see config.py)</h2>
<table>{power_rows}</table>

<h2>Timeline</h2>
<div class="timeline-wrap">{svg_timeline}</div>
<p class="note">Each column is a binned average over
{sim.timeline.bin_width} cycles. Opacity = occupancy/utilization/contention fraction.</p>

</body></html>"""

    with open(path, "w") as f:
        f.write(html)
    return path


def export_timeline_png(sim, path: str, max_columns: int = 160):
    """Matplotlib stacked-row heatmap of worker/DMA/AXI activity, same
    data as the ASCII gantt in timeline.py. Raises ImportError with a
    clear message if matplotlib isn't installed, rather than silently
    doing nothing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "export_timeline_png requires matplotlib (pip install matplotlib "
            "--break-system-packages). The ASCII timeline (sim.timeline.render_ascii()) "
            "and the HTML export's inline SVG timeline both work without it."
        ) from e

    tl = sim.timeline
    tl.finalize()
    n_bins = len(tl.dma_bins)
    if n_bins == 0:
        raise ValueError("No cycles recorded -- run the simulation before exporting.")

    def downsample(series):
        arr = np.array(series)
        if n_bins <= max_columns:
            return arr
        edges = np.linspace(0, n_bins, max_columns + 1).astype(int)
        return np.array([arr[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
                          for i in range(max_columns)])

    labels = [f"Worker{i}" for i in range(tl.num_workers)] + ["DMA", "AXI use", "AXI contention"]
    series_list = [downsample(tl.worker_bins[i]) for i in range(tl.num_workers)]
    series_list += [downsample(tl.dma_bins), downsample(tl.axi_util_bins),
                     downsample(tl.axi_contention_bins)]
    matrix = np.array(series_list)

    fig, ax = plt.subplots(figsize=(12, 0.35 * len(labels) + 1))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Cycle (binned, ~{tl.bin_width * n_bins / min(n_bins, max_columns):.0f} cycles/column)")
    ax.set_title("HEXA-TPU-RT Activity Timeline")
    fig.colorbar(im, ax=ax, label="Occupancy / utilization fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
