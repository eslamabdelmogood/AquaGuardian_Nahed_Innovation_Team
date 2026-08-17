# AquaGuardian AI Architecture

```mermaid
flowchart LR
  A[Living Sensors<br/>Flow · Pressure · Quality · Soil · Plant Bio-signals] --> B[Green Box Edge Core]
  B --> C[Risk Analysis]
  C --> D[Living Digital Twin]
  D --> E[Stress Test Matrix]
  E --> F[Multi-objective Optimizer]
  F --> G{Validated?}
  G -- No --> D
  G -- Yes --> H[Safety Gate]
  H --> I[Execute / Recommend Action]
  I --> J[Evidence & Lessons Learned]
  J --> B
```

## Design principles

1. Edge-first and offline-capable
2. No automatic action without simulation and validation
3. Fail-safe fallback under communication loss
4. Evidence attached to every decision
5. Human approval required for real deployments
