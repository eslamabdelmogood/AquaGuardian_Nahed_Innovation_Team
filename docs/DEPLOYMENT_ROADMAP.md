# Deployment and Validation Roadmap

## Phase 0 — Current submission: TRL 3

- deterministic simulation;
- reproducible comparison baselines;
- API, UI, tests and Docker;
- no field-control claim.

## Phase 1 — Laboratory validation: target TRL 4

Duration assumption: 3–6 months.

- connect representative sensors and a small pump/valve test loop;
- replay leak, pump and drought traces;
- calibrate simulation parameters;
- measure detection latency, false alarms, model error and command safety;
- complete security and hazard reviews.

Exit criteria:

- predefined test cases pass;
- model error is reported and bounded;
- unsafe commands are rejected;
- all decisions are auditable.

## Phase 2 — Shadow-mode pilot: target TRL 5

Duration assumption: 6–9 months.

- deploy beside an existing farm or irrigation operation;
- ingest live sensor data without controlling equipment;
- compare recommendations with operator actions;
- quantify water, energy, maintenance and response-time impact;
- obtain operator feedback.

Exit criteria:

- agreed KPI improvement or credible avoided-loss evidence;
- acceptable false-alarm burden;
- operator acceptance of the decision trail;
- signed plan for a supervised pilot.

## Phase 3 — Supervised pilot: target TRL 6

- enable operator-approved low-risk actions;
- integrate identity, audit and command signing;
- validate fail-safe and rollback behavior;
- conduct independent cybersecurity and safety review.

## Phase 4 — Commercial deployment

- packaged site templates;
- support and service-level agreements;
- device lifecycle and secure-update process;
- partner channel for sensors and integration.
