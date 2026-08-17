from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aquaguardian.digital_twin import simulate
from aquaguardian.policy import ACTIONS, validate
from aquaguardian.scenarios import SCENARIOS


def main() -> None:
    rows = []
    profiles = [
        ("nominal", 1.0, 0.0, False),
        ("high_severity", 1.4, 0.02, False),
        ("noisy_sensors", 1.2, 0.10, False),
        ("offline_degraded", 1.35, 0.06, True),
        ("extreme_combined", 1.7, 0.12, True),
    ]

    for scenario, frame in SCENARIOS.items():
        for action in ACTIONS[scenario]:
            for profile, severity, noise, offline in profiles:
                result = simulate(
                    scenario,
                    frame,
                    action,
                    severity=severity,
                    sensor_noise=noise,
                    communication_loss=offline,
                    seed=42,
                )
                passed, reasons = validate(result)
                row = result.to_dict()
                row.update(
                    {
                        "scenario": scenario,
                        "action": action.name,
                        "profile": profile,
                        "passed": passed,
                        "reasons": reasons,
                    }
                )
                rows.append(row)

    summary = {
        "total_runs": len(rows),
        "passed_runs": sum(1 for row in rows if row["passed"]),
        "failed_runs": sum(1 for row in rows if not row["passed"]),
        "rows": rows,
    }
    out = ROOT / "outputs" / "stress_matrix.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"Stress evidence written to {out}")


if __name__ == "__main__":
    main()
