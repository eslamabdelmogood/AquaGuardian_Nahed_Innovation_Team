from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aquaguardian.engine import ClosedLoopEngine
from aquaguardian.scenarios import SCENARIOS


def main() -> None:
    engine = ClosedLoopEngine()
    results = {}
    print("AquaGuardian AI — closed-loop engineering demo")
    print("=" * 58)
    for name, frame in SCENARIOS.items():
        decision = engine.decide(frame, name)
        results[name] = decision.to_dict()
        print(
            f"{name:24s} risk={decision.detected_risk:.3f} "
            f"action={decision.selected_action:20s} "
            f"validated={decision.executed} reliability={decision.reliability:.2f}"
        )

    out = ROOT / "outputs" / "demo_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEvidence written to {out}")


if __name__ == "__main__":
    main()
