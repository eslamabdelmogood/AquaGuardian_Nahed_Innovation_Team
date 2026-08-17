# Concise Threat Model

## Assets to protect

- water availability and quality;
- pumps, valves and irrigation actuators;
- sensor integrity;
- operator accounts;
- network topology and operational data;
- decision and audit records.

## Trust boundaries

1. Sensor/edge-device boundary.
2. Edge-to-platform communication boundary.
3. Human operator and approval boundary.
4. Platform-to-SCADA/actuator boundary.
5. Software update and configuration boundary.

## Highest-consequence failure

A false or manipulated decision that causes unsafe valve, pump or irrigation behavior. The production architecture must therefore keep deterministic safety constraints outside any probabilistic model and require bounded, authenticated commands.

## PoC exclusions

The current repository does not include real credentials, production SCADA drivers, remote actuator control or a production identity provider. These exclusions are intentional at TRL 3.
