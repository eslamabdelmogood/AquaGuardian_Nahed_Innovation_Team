# Security and Resilience Roadmap

## Current PoC security posture

This competition package is a local demonstration and is not intended to control production infrastructure. It has no user identity system, no actuator credentials and no direct SCADA connection.

The application now defaults CORS access to local development origins. Production deployments must set an explicit allow-list through configuration.

## Primary threats

- unauthorized API access;
- tampered sensor data;
- replayed telemetry;
- malicious or faulty actuator commands;
- compromised edge nodes;
- model or policy configuration changes without approval;
- denial of service or loss of connectivity;
- leakage of operational infrastructure data.

## Production controls

### Identity and access

- mutual TLS for edge devices;
- OAuth2/OIDC for human users;
- role-based access control;
- separate read, approve and execute permissions;
- short-lived credentials and secret rotation.

### Command safety

- signed commands and monotonically increasing sequence numbers;
- allow-listed actuator actions and bounded parameters;
- two-person approval for high-impact operations;
- local hard safety limits independent of AI decisions;
- manual override and safe-state fallback.

### Data integrity

- timestamp validation and replay protection;
- sensor-quality scoring and anomaly detection;
- encrypted transport and storage;
- immutable audit records for inputs, candidate actions and approvals.

### Edge resilience

- secure boot and signed firmware;
- encrypted device storage;
- watchdog and health monitoring;
- store-and-forward operation during network outages;
- signed over-the-air updates with rollback.

## Deployment principle

AquaGuardian should progress through four control modes:

1. **Replay mode** — recorded data only.
2. **Shadow mode** — live data, recommendations only.
3. **Supervised mode** — operator approves each action.
4. **Bounded autonomy** — only pre-approved low-risk actions execute automatically.

No production claim should be made before independent security review and site-specific hazard analysis.
