# Comparative Engineering Validation

AquaGuardian AI is evaluated against two simplified reference strategies:

1. **Reactive baseline** — a conventional alarm/operator response with a modeled five-minute delay and no pre-execution simulation.
2. **Detection-only AI** — an immediate rule-based action selected from detected risk, but without candidate simulation or stress testing.
3. **AquaGuardian closed loop** — candidate actions are simulated, stress-tested, scored, and validated before execution.

## Evaluation protocol

Each strategy is evaluated in four scenarios:

- Pipeline leak
- Pump degradation
- Water contamination
- Drought / wildfire risk

Each selected action is tested under four deterministic profiles:

- Nominal
- Moderate stress
- Severe offline conditions
- Extreme combined conditions

The generated report compares:

- Stress-test pass rate
- Water loss
- Response delay
- Safety risk
- Service disruption
- Contamination exposure
- Crop stress
- Fire risk

Run:

```bash
python scripts/run_comparison.py
```

Output:

```text
outputs/strategy_comparison.json
```

## Evidence boundary

These figures are deterministic outputs from the included proof-of-concept simulator. They are intended to compare engineering strategies under the same modeled assumptions. They are **not field-trial measurements** and must not be presented as real-world performance until validated with calibrated sensors, hydraulic models, and pilot deployments.
