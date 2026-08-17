from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aquaguardian.comparison import aggregate_comparisons, compare_scenario  # noqa: E402
from aquaguardian.scenarios import SCENARIOS  # noqa: E402


def main() -> None:
    comparisons = [compare_scenario(name, frame) for name, frame in SCENARIOS.items()]
    payload = {
        "summary": aggregate_comparisons(comparisons),
        "comparisons": comparisons,
    }
    output = ROOT / "outputs" / "strategy_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("AquaGuardian strategy comparison completed")
    print(f"Output: {output}")
    for strategy, metrics in payload["summary"]["strategies"].items():
        print(
            f"{strategy:28} pass_rate={metrics['average_stress_pass_rate']:.2f} "
            f"water_loss={metrics['average_water_loss_l']:.1f}L "
            f"safety_risk={metrics['average_safety_risk']:.3f} "
            f"delay={metrics['average_response_delay_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
