# Translating Simulation Metrics into Deployment Impact

## Purpose

The software reports scenario-level proxy metrics. This document shows how those metrics should be translated into a real deployment case without presenting simulated values as field evidence.

## Example deployment frame

Consider a 500-hectare managed irrigation district. Let:

- `W` be measured annual water supplied to the district;
- `L` be the independently measured baseline fraction lost to avoidable leakage, runoff or poor scheduling;
- `R` be the validated reduction in that avoidable loss during a pilot.

Then estimated annual water saved is:

`Annual water saved = W × L × R`

### Illustrative example only

If the district supplies 5,000,000 m³/year, independently verifies 8% avoidable loss, and a pilot validates a 20% reduction in that avoidable loss:

`5,000,000 × 0.08 × 0.20 = 80,000 m³/year`

This equals 80 million liters per year. It is an illustration of the calculation method, **not a claim about AquaGuardian's current performance**.

## Pilot KPIs

- cubic meters of water supplied and saved;
- pumping kWh per cubic meter;
- leak containment time;
- irrigation uniformity and root-zone uptake;
- unplanned pump downtime;
- false alarms per operating month;
- percentage of recommendations accepted by operators;
- percentage of candidate actions rejected by safety constraints.

## Evidence rule

AquaGuardian will only convert simulation percentages into customer savings after baseline data, system boundaries and measurement methodology are agreed with a pilot partner.
