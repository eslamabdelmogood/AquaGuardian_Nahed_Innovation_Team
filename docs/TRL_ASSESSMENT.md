# Technology Readiness Level Assessment

## Current classification: TRL 3 — experimental proof of concept

AquaGuardian currently demonstrates the core closed-loop engineering hypothesis in a controlled software environment:

- sensor frames are converted into risk evidence;
- candidate actions are simulated;
- unsafe plans can be rejected under stress profiles;
- an optimized plan is validated before execution;
- results are compared with reactive and detection-only baselines;
- outputs are reproducible through tests and JSON evidence.

The current package does **not** claim field deployment, hardware-in-the-loop validation, utility integration, or calibrated performance against a real hydraulic network. For that reason, TRL 3 is the most credible classification.

## Evidence supporting TRL 3

1. A working deterministic software prototype.
2. A digital-twin abstraction for multiple water-risk scenarios.
3. Automated tests for engine, API and comparative evidence.
4. Dockerized reproducible execution.
5. Explicit model limitations and separation between simulated and field evidence.

## Path to TRL 4

TRL 4 will require validation of the integrated prototype in a laboratory or controlled test environment.

Planned evidence:

- connect real pressure, flow, soil-moisture and vibration sensors;
- replay recorded fault traces through the engine;
- calibrate the digital twin against measured system behavior;
- compare predicted and observed outcomes;
- verify fail-safe behavior during sensor noise, packet loss and actuator unavailability;
- document accuracy, false alarms, latency and water-loss estimation error.

## Path to TRL 5–6

A controlled shadow-mode pilot will run alongside an operating farm, irrigation district or small water utility without commanding actuators. After acceptance criteria are met, a limited supervised pilot can validate selected low-risk actions.

## Readiness claim

AquaGuardian is submitted as a **Discovery-stage, TRL 3 proof of concept** with a defined validation path toward TRL 4 and field piloting.
