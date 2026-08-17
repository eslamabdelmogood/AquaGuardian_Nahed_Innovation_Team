"""
power.py
========
Turns the simulator's already-tracked activity counters into an energy
and power estimate, specifically so the spec's two power claims are
checkable rather than asserted:

  1. "Event-Driven Sparsity Gating ... cutting active power by up to 40%"
  2. "Energy Efficiency ~0.35 Watt / TOPS (with Sparsity Gating enabled)"

Every constant this model depends on (config.ENERGY_PER_MAC_ACTIVE_PJ,
DDR_ENERGY_PJ_PER_BYTE, etc.) is an explicit assumption documented in
config.py -- NOT derived from the spec, which contains no process-node
power characterization at all. This model can tell you whether the
spec's claims are *internally consistent* with a plausible set of
energy assumptions; it cannot verify them against silicon.

Energy accounting, three buckets:
  - MAC compute energy: active MACs at full cost, gated MACs at a small
    residual cost (gating is never perfectly zero-power in real silicon)
  - Data movement energy: AXI (on-chip, cheap) + DDR (off-chip, ~400x
    more expensive per byte by the default assumption) -- this is
    usually where realistic edge accelerators actually spend their
    power budget, not in the MAC array
  - Static leakage: per-worker + master + interconnect, held on for the
    whole run unless POWER_GATE_IDLE_WORKERS derates idle cycles
"""

from dataclasses import dataclass


@dataclass
class PowerBreakdown:
    mac_active_energy_nj: float
    mac_gated_energy_nj: float
    mac_energy_if_no_gating_nj: float   # hypothetical: same run, gating disabled
    sram_axi_energy_nj: float
    ddr_energy_nj: float
    static_energy_nj: float
    total_energy_nj: float
    total_time_s: float
    average_power_mw: float
    tops_per_watt: float
    watts_per_tops: float
    sparsity_gating_power_cut_pct: float  # measured against the MAC-only budget,
                                            # matching what the spec's claim covers

    def render(self, estimated_tops: float) -> str:
        lines = []
        lines.append("-" * 50)
        lines.append("Power Model (Phase 3 -- assumptions, not measurements)")
        lines.append("-" * 50)
        lines.append(f"  MAC active energy:      {self.mac_active_energy_nj:>10.2f} nJ")
        lines.append(f"  MAC gated residual:      {self.mac_gated_energy_nj:>10.2f} nJ")
        lines.append(f"  MAC energy w/o gating*:  {self.mac_energy_if_no_gating_nj:>10.2f} nJ  "
                      f"(*hypothetical, same workload)")
        lines.append(f"  SRAM/AXI data movement:  {self.sram_axi_energy_nj:>10.2f} nJ")
        lines.append(f"  DDR data movement:       {self.ddr_energy_nj:>10.2f} nJ")
        lines.append(f"  Static leakage:          {self.static_energy_nj:>10.2f} nJ")
        lines.append(f"  " + "-" * 40)
        lines.append(f"  Total energy:            {self.total_energy_nj:>10.2f} nJ")
        lines.append(f"  Average power:           {self.average_power_mw:>10.2f} mW")
        lines.append(f"  Efficiency:              {self.tops_per_watt:>10.3f} TOPS/W  "
                      f"({self.watts_per_tops:.3f} W/TOPS)")
        lines.append(f"  Spec target:                    2.857 TOPS/W  (0.350 W/TOPS)")
        lines.append(f"  Sparsity-gating MAC power cut: {self.sparsity_gating_power_cut_pct:>6.1f}%  "
                      f"(spec claims 'up to 40%')")
        lines.append("-" * 50)
        return "\n".join(lines)


class PowerModel:
    def __init__(self, config):
        self.cfg = config

    def compute(self, sim, report) -> PowerBreakdown:
        cfg = self.cfg
        cycle_time_s = cfg.cycle_time_ns() * 1e-9
        total_time_s = report.total_cycles * cycle_time_s

        total_macs_active = sum(w.systolic.total_macs_executed for w in sim.workers)
        total_macs_gated = sum(w.systolic.total_macs_gated for w in sim.workers)

        e_mac_active_pj = total_macs_active * cfg.ENERGY_PER_MAC_ACTIVE_PJ
        e_mac_gated_pj = total_macs_gated * cfg.ENERGY_PER_MAC_GATED_PJ
        # Hypothetical: same total MAC count, but as if every gated MAC
        # had instead run at full active cost (i.e. gating disabled).
        e_mac_no_gating_pj = (total_macs_active + total_macs_gated) * cfg.ENERGY_PER_MAC_ACTIVE_PJ

        e_sram_pj = sim.axi.total_bytes_transferred * cfg.SRAM_ENERGY_PJ_PER_BYTE
        e_ddr_pj = sim.dma.total_bytes_transferred * cfg.DDR_ENERGY_PJ_PER_BYTE

        if cfg.POWER_GATE_IDLE_WORKERS and report.total_cycles > 0:
            worker_leak_mw = 0.0
            for w in sim.workers:
                active_frac = w.busy_cycles / report.total_cycles
                idle_frac = 1.0 - active_frac
                worker_leak_mw += cfg.STATIC_LEAKAGE_MW_PER_WORKER * (
                    active_frac + idle_frac * cfg.IDLE_WORKER_LEAKAGE_FRACTION
                )
        else:
            worker_leak_mw = cfg.STATIC_LEAKAGE_MW_PER_WORKER * len(sim.workers)

        static_mw = worker_leak_mw + cfg.STATIC_LEAKAGE_MW_MASTER + cfg.STATIC_LEAKAGE_MW_INTERCONNECT
        e_static_pj = static_mw * 1e-3 * total_time_s * 1e12  # mW * s -> pJ

        # Convert everything to nJ (1 nJ = 1000 pJ) for display sanity
        # at these MAC counts (hundreds of thousands to millions of MACs).
        mac_active_nj = e_mac_active_pj / 1000.0
        mac_gated_nj = e_mac_gated_pj / 1000.0
        mac_no_gating_nj = e_mac_no_gating_pj / 1000.0
        sram_nj = e_sram_pj / 1000.0
        ddr_nj = e_ddr_pj / 1000.0
        static_nj = e_static_pj / 1000.0

        total_energy_nj = mac_active_nj + mac_gated_nj + sram_nj + ddr_nj + static_nj
        total_energy_j = total_energy_nj * 1e-9
        average_power_w = total_energy_j / total_time_s if total_time_s > 0 else 0.0
        average_power_mw = average_power_w * 1000.0

        tops_per_watt = (report.estimated_tops / average_power_w) if average_power_w > 0 else 0.0
        watts_per_tops = (average_power_w / report.estimated_tops) if report.estimated_tops > 0 else 0.0

        # Power cut attributable to sparsity gating, measured only over
        # the MAC-array energy budget (matches what the spec's claim is
        # actually about -- it does not claim anything about SRAM/DDR).
        mac_energy_with_gating_nj = mac_active_nj + mac_gated_nj
        if mac_no_gating_nj > 0:
            power_cut_pct = (1.0 - mac_energy_with_gating_nj / mac_no_gating_nj) * 100.0
        else:
            power_cut_pct = 0.0

        return PowerBreakdown(
            mac_active_energy_nj=mac_active_nj,
            mac_gated_energy_nj=mac_gated_nj,
            mac_energy_if_no_gating_nj=mac_no_gating_nj,
            sram_axi_energy_nj=sram_nj,
            ddr_energy_nj=ddr_nj,
            static_energy_nj=static_nj,
            total_energy_nj=total_energy_nj,
            total_time_s=total_time_s,
            average_power_mw=average_power_mw,
            tops_per_watt=tops_per_watt,
            watts_per_tops=watts_per_tops,
            sparsity_gating_power_cut_pct=power_cut_pct,
        )
