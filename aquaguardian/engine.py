from __future__ import annotations
import time
from .detector import analyze
from .digital_twin import simulate
from .models import Decision, SensorFrame
from .policy import ACTIONS, score, validate


class ClosedLoopEngine:
    def decide(self, frame: SensorFrame, scenario: str) -> Decision:
        started = time.perf_counter()
        risks = analyze(frame)
        detected_risk = risks[scenario]
        candidates = []
        iterations = 0

        # Nominal simulation followed by increasingly difficult stress tests.
        stress_profiles = [
            {"severity": 1.0, "sensor_noise": 0.00, "communication_loss": False},
            {"severity": 1.20, "sensor_noise": 0.03, "communication_loss": False},
            {"severity": 1.45, "sensor_noise": 0.06, "communication_loss": True},
        ]

        for action_index, action in enumerate(ACTIONS[scenario]):
            iterations += 1
            runs = []
            all_passed = True
            reasons = []
            for profile_index, profile in enumerate(stress_profiles):
                result = simulate(
                    scenario,
                    frame,
                    action,
                    seed=1000 + action_index * 10 + profile_index,
                    **profile,
                )
                passed, run_reasons = validate(result)
                result.passed = passed
                result.reasons = run_reasons
                result.score = score(result)
                runs.append(result)
                all_passed = all_passed and passed
                reasons.extend(run_reasons)

            worst_score = min(r.score for r in runs)
            candidates.append(
                {
                    "action": action.name,
                    "passed_all_stress_tests": all_passed,
                    "worst_score": worst_score,
                    "stress_runs": [r.to_dict() for r in runs],
                    "rejection_reasons": sorted(set(reasons)),
                }
            )

        valid = [c for c in candidates if c["passed_all_stress_tests"]]
        selected = max(valid or candidates, key=lambda c: c["worst_score"])
        executed = bool(valid) and detected_risk >= 0.35

        spread = sorted((c["worst_score"] for c in candidates), reverse=True)
        margin = spread[0] - spread[1] if len(spread) > 1 else spread[0]
        confidence = min(0.99, 0.55 + detected_risk * 0.30 + margin / 100.0)
        reliability = min(
            0.99,
            sum(1 for r in selected["stress_runs"] if r["passed"]) / len(selected["stress_runs"])
            * 0.90
            + 0.09,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        evidence = [
            f"detected_risk={detected_risk:.3f}",
            f"selected_worst_score={selected['worst_score']:.3f}",
            f"stress_profiles={len(stress_profiles)}",
            f"decision_latency_ms={latency_ms:.3f}",
            "offline_edge_path=true",
        ]
        return Decision(
            scenario=scenario,
            detected_risk=detected_risk,
            selected_action=selected["action"],
            confidence=round(confidence, 4),
            reliability=round(reliability, 4),
            iterations=iterations,
            executed=executed,
            evidence=evidence,
            alternatives=candidates,
        )
