"""
benchmarks/onem2m_mec_benchmark.py
====================================
Runs the full standards-integrated pipeline (oneM2M middleware -> 5G
URLLC link -> ETSI MEC BHS Cognitive Engine -> existing HEXA-TPU-RT
TPU simulator) across all four BDO-SKIN scenarios and reports:

  - oneM2M: total CRUD primitives handled by the MN-CSE (audit trail).
  - Network: URLLC latency distribution and sub-10ms SLA compliance.
  - MEC: RUL/veto/control-decision summary, including how often
    Hermit Crab's veto logic actually blocked a candidate action.
  - TPU: unchanged report from simulator.py, for continuity with the
    existing bdo_skin_integration report.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from config import Config
from integration.edge_to_mec_pipeline import EdgeToMecPipeline

SCENARIOS = ["normal", "gradual_anomaly", "burst_anomaly", "critical_event"]


def run_scenario(scenario: str, num_workers: int = 10, num_windows: int = 150, seed: int = 0,
                  cse_http_url: str = None):
    cfg = Config(NUM_WORKERS=num_workers)
    edge_name = f"hexa-tpu-edge-{scenario}-{int(time.time()*1000)}" if cse_http_url else "hexa-tpu-edge-01"
    pipeline = EdgeToMecPipeline(cfg=cfg, seed=seed, cse_http_url=cse_http_url, edge_node_name=edge_name)
    report = pipeline.run(scenario=scenario, num_windows=num_windows, seed=seed,
                           num_workers=num_workers)

    print(f"\n--- Scenario: {scenario} (windows={num_windows}, "
          f"critical={len(report.critical_window_indices)}) ---")
    print(f"  [oneM2M]  MN-CSE handled {report.onem2m_request_log_len} CRUD primitives "
          f"(AE registration + 2 containers + {num_windows} sensor + "
          f"{num_windows} reflex contentInstances)")

    u = report.urllc_stats
    print(f"  [URLLC]   n={u['count']}  mean={u['mean_latency_ms']:.3f}ms  "
          f"p99={u['p99_latency_ms']:.3f}ms  max={u['max_latency_ms']:.3f}ms  "
          f"SLA(<= {u['sla_target_ms']:.0f}ms) compliance={u['sla_compliance_rate']*100:.2f}%  "
          f"retransmit_rate={u['retransmission_rate']*100:.3f}%")

    vetoes = report.mec_vetoes()
    total_vetoed_actions = sum(len(v) for v in vetoes.values())
    windows_with_veto = len(vetoes)
    rul_defined = [t.cognitive.rul.remaining_useful_life_windows for t in report.telemetry
                   if t.cognitive.rul.remaining_useful_life_windows is not None]
    print(f"  [MEC]     windows with >=1 vetoed action: {windows_with_veto}/{num_windows}  "
          f"(total vetoed-action instances: {total_vetoed_actions})")
    if rul_defined:
        print(f"  [MEC]     Bat RUL projected in {len(rul_defined)} windows  "
              f"min={min(rul_defined):.1f}  mean={sum(rul_defined)/len(rul_defined):.1f} windows")
    chosen_actions = [t.cognitive.control.chosen_action for t in report.telemetry
                       if t.cognitive.control.chosen_action]
    if chosen_actions:
        from collections import Counter
        counts = Counter(chosen_actions)
        print(f"  [MEC]     Squid action distribution: "
              f"{dict(counts.most_common())}")

    sla_violations = report.urllc_sla_violations()
    if sla_violations:
        print(f"  [URLLC]   SLA violations at windows: {sla_violations[:10]}"
              f"{' ...' if len(sla_violations) > 10 else ''}")

    tr = report.tpu_report
    print(f"  [TPU]     cycles={tr.total_cycles}  tasks={tr.tasks_completed}  "
          f"deadline_misses={tr.deadline_misses}  avg_power={tr.power.average_power_mw:.1f}mW")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-http-url", default=None, metavar="URL",
                         help="Point at a real, running oneM2M CSE (e.g. "
                              "http://127.0.0.1:8080) instead of the in-memory "
                              "simulation. Start one first with "
                              "'./deploy/run_acme_cse.sh'.")
    parser.add_argument("--num-windows", type=int, default=150)
    args = parser.parse_args()

    print("=" * 78)
    print("Standardization & Middleware (oneM2M) + MEC Compute Layer (ETSI ISG MEC)")
    print("integration benchmark, run through the existing HEXA-TPU-RT TPU pipeline")
    if args.cse_http_url:
        print(f"oneM2M transport: REAL CSE at {args.cse_http_url}")
    else:
        print("oneM2M transport: in-memory simulation (pass --cse-http-url for a real CSE)")
    print("=" * 78)
    for scenario in SCENARIOS:
        run_scenario(scenario, num_windows=args.num_windows, cse_http_url=args.cse_http_url)


if __name__ == "__main__":
    main()
