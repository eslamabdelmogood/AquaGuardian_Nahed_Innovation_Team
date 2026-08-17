# HEXA-TPU-RT Architecture Simulator — v6.0

## Phase 9: Domain-neutral Reflex Layer — Aviation DAA + Water Infrastructure

The platform's central claim — "the same core protects an aircraft in
the air and an underground water network" — depends entirely on the
local reflex path actually deciding and acting on a hard sub-1ms
deadline, **independent of** the oneM2M/URLLC/MEC pipeline (Phase 8,
below), for either domain. Before this phase, "reflex" was only a
MAC-cost workload label fed into the TPU scheduler — it computed
something, but nothing in the codebase decided-and-acted on an
enforced deadline, and nothing was structurally separate from the
network path. This phase closes that gap.

```
reflex/reflex_kernel.py            -- domain-neutral core (NEW)
  ReflexKernel: evaluate(sample) -> ReflexEvent, deadline-checked,
  independent of scheduler.py/worker.py by design (a hard real-time
  path cannot wait on the main TPU's shared, contended pipeline)

reflex/domains/aviation.py         -- domain profile #1 (NEW)
  AviationDAATrigger: fog-derated time-to-collision heuristic
  generate_aviation_scenario(): clear_sky / light_haze /
    dense_fog_approach / sudden_incursion

reflex/domains/water.py            -- domain profile #2 (NEW)
  WaterPressureTransientTrigger: pressure drop-rate (water-hammer)
    heuristic
  generate_water_scenario(): steady_demand / daily_cycle /
    gradual_leak / sudden_burst
```

Only the trigger (and the sample shape it reads) is domain-specific;
`ReflexKernel`, `ActuationDecision`, `ReflexEvent`, and the
deadline-enforcement logic are identical Python objects reused
unmodified across both domains — `tests/test_reflex_kernel.py`'s
`TestDomainNeutrality` asserts this directly (`assertIs(type(...),
type(...))`), not just by convention.

Run it: `python benchmarks/reflex_domain_benchmark.py`. Tests:
`tests/test_reflex_kernel.py` (17 tests, all passing).

### What was found running it

- **Both domains, same kernel, deadline never missed**: across all 8
  scenarios (4 aviation + 4 water, 200 samples each), 0 deadline
  misses. At the modeled reflex-kernel clock (1200MHz, separate from
  the main TPU's 800MHz — this is deliberately distinct, simpler
  silicon per the pitch's "Micro-TPU / Spiking Reflex Kernel"), a
  3-channel aviation decision and a 2-channel water decision both cost
  well under a microsecond — several orders of magnitude inside the
  1ms budget, not just barely under it.
- **Detection accuracy against each scenario's ground truth**: the
  `sudden_incursion` (aviation) and `sudden_burst` (water) scenarios
  each embed a known ground-truth event; the reflex kernel's trigger
  count lines up with that ground truth in both domains (exact match
  on `sudden_burst`'s single embedded event) without any
  domain-specific tuning of the kernel itself — only the two triggers
  differ.
- **The "without waiting for the network" number, made concrete**:
  `benchmarks/reflex_domain_benchmark.py` measures both sides from
  this same codebase — the reflex kernel's own worst-case latency
  against `network/urllc_link.py`'s measured mean/p99 latency from a
  real run (not a spec figure) — and reports the ratio directly rather
  than asserting it.
- **`dense_fog_approach` triggers earlier than the scenario's own
  `near_miss` flag** (97/200 triggered vs. 60/200 flagged as
  near-miss): this is by design, not a false-positive count — the
  reflex physics (TTC vs. fog-derated margin) is what should decide
  when to react, and reacting *before* a human-labeled "near miss"
  threshold is exactly the point of a preventive reflex layer, not a
  bug to tune away.

### What this does *not* claim

- **The trigger rules are demonstration heuristics, not certified
  algorithms.** `AviationDAATrigger`'s time-to-collision/fog-margin
  model is a simplified illustration of the kind of local rule a real
  DAA system would run, not a substitute for TCAS/ACAS-X-class
  certified collision-avoidance logic; `WaterPressureTransientTrigger`'s
  drop-rate threshold is a simplified illustration, not a substitute
  for real water-hammer/transient hydraulic analysis. Both are
  explicitly named as such in their own docstrings, with every
  numeric constant adjustable.
- **The sub-1ms figure is a cycle-count deadline check** (same
  cycle-accurate, not-wall-clock methodology the rest of this
  simulator uses — see `simulator.py`/`scheduler.py`), calibrated
  against a stated, adjustable clock frequency and per-channel cost —
  not a measurement from real embedded hardware, real ADC sampling
  jitter, or real interrupt latency.
- **The scenario generators are illustrative, not flight-test or SCADA
  data.** Every constant (visibility ranges, closing speeds, pressure
  thresholds, timing) is named and adjustable in
  `reflex/domains/aviation.py` and `reflex/domains/water.py`.

### The MEC layer follows the same reuse pattern

The reflex layer isn't the only place the "one core, two industries"
claim has to hold up — the MEC Compute Layer (Phase 8, below) does too.
`mec/domains/water_cognitive_engine.py` extends that pattern upward:

- **Aviation needed zero new MEC code.** `mec/bhs_cognitive_engine.py`
  was already written as structural/airframe monitoring — resonance,
  RUL, safety/productivity/power balancing under vibration and thermal
  stress. That literally *is* the aviation domain's cognitive engine,
  since BDO-SKIN's original framing is an aircraft-wing optical skin.
- **Water reuses Bat Forecaster and Squid Controller as the literal
  same classes** (`mec.bhs_cognitive_engine.BatForecaster`,
  `SquidController` — not subclasses, not reimplementations;
  `type(aviation.bat) is type(water.bat)` is asserted directly in
  `tests/test_water_cognitive_engine.py`). Structural RUL forecasts a
  *rising* trend toward a failure ceiling; a growing leak shows up as
  a *declining* pressure baseline instead, so `WaterBHSCognitiveEngine`
  transforms the input into a rising "cumulative pressure deficit"
  series and feeds that into the unmodified forecaster. Squid's
  weighted safety/productivity/power scoring is duck-typed against
  whatever candidate-action-shaped objects it receives, so
  `WaterCandidateAction` (a plain, unrelated dataclass with matching
  attribute names) works without any inheritance relationship to the
  aviation `CandidateAction`.
- **Only Hermit Crab's veto physics genuinely differs**:
  `WaterHammerHermitCrabEvaluator` vetoes any candidate flow-rate
  change whose estimated Joukowsky surge pressure (`dP = rho * a *
  dV`) exceeds a hard threshold — the water-domain equivalent of
  vetoing actions inside a structural resonance band. Both play the
  identical role (a hard constraint Squid's scoring can't override,
  regardless of how good the vetoed action's raw safety score looks):
  in the benchmark run, `emergency_full_shutdown` scores highest on
  safety_benefit alone but is vetoed at every severity level, forcing
  the controller toward `staged_valve_shutdown` instead as the leak
  worsens.

`benchmarks/reflex_domain_benchmark.py` prints this directly —
`type(aviation.bat) is type(water.bat)` and the equivalent for Squid —
alongside the reflex-layer numbers, so both halves of the neutrality
claim are demonstrated by one script. Tests:
`tests/test_water_cognitive_engine.py` (9 tests).

## Phase 8: Standardization & Middleware (oneM2M) + MEC Compute Layer (ETSI ISG MEC)

Adds two standards-modeled layers in front of the existing,
**unmodified** BDO-SKIN -> TPU pipeline below, plus the network hop
between them. Nothing under `scheduler.py` / `dma.py` / `axi.py` /
`cache.py` / `worker.py` / `power.py` changed — same principle the
BDO-SKIN integration itself follows: new front end feeding the
existing engine, not a second simulator.

```
SensorWindow (models/sensor_events.py)
    v
ADN-AE -> MN-CSE                middleware/onem2m.py     (simulated)
                                 middleware/onem2m_http.py (real CSE)
  (oneM2M TS-0001/TS-0004: AE registration, <container>,
   <contentInstance> CRUD primitives)
    v
5G URLLC link                   network/urllc_link.py
  (log-normal latency + rare HARQ-retransmission model,
   sub-10ms transmission-latency SLA)
    v
MEC platform + BHS Cognitive    mec/mec_platform.py
Engine app (ETSI GS MEC 003)    mec/bhs_cognitive_engine.py
  - Bat Forecaster: RUL/stress-trend via an explicit least-squares
    matrix solve (normal-equation form, not polyfit's black box)
  - Hermit Crab Evaluator: stability score + veto logic blocking any
    candidate action whose actuation frequency falls inside a
    resonance guard-band around the estimated natural frequency
  - Squid Controller: weighted multi-objective scoring over
    safety/productivity/power, re-weighted continuously by severity,
    operating only on Hermit Crab's non-vetoed action set
    v
models/bdo_skin.py's existing (layer_name, [Task,...]) generator (UNCHANGED)
    v
HexaTPUSimulator.run()          simulator.py (UNCHANGED)
```

`integration/edge_to_mec_pipeline.py` wires all of the above into one
`EdgeToMecPipeline.run()` call; `benchmarks/onem2m_mec_benchmark.py`
runs it across all four BDO-SKIN scenarios and reports oneM2M CRUD
counts, URLLC latency/SLA-compliance stats, and MEC veto/RUL/action
decisions alongside the existing (unchanged) TPU report. Tests:
`tests/test_middleware_mec.py`.

### Two oneM2M transports: simulated and real

`ADN_AE` (middleware/onem2m.py) only depends on a `handle_primitive(RequestPrimitive) ->
ResponsePrimitive` method, so the actual CSE backend is swappable
without touching `ADN_AE`, the MEC layer, or the TPU pipeline:

- **`middleware/onem2m.py`'s `MN_CSE`** — in-memory simulation of the
  resource tree, used by default (no external process, instant, good
  for unit tests and offline development).
- **`middleware/onem2m_http.py`'s `HttpCSEClient`** — real oneM2M
  HTTP binding (TS-0004 clause 8.2) against an actual, running CSE.
  Developed and verified against a locally-run **ACME oneM2M CSE**
  (`pip install acmecse`) — one of the three implementations
  (ACME, tinyIoT, Mobius) the ETSI ESTIMED hackathon names explicitly
  — but speaks the standard binding, not anything ACME-specific, so
  tinyIoT or Mobius should work as drop-in alternatives (only
  `base_url`/`cse_id` change).

To run against a real CSE:

```bash
./deploy/run_acme_cse.sh          # installs + starts ACME on :8080 (headless, HTTP-only, IN-CSE)
python benchmarks/onem2m_mec_benchmark.py --cse-http-url http://127.0.0.1:8080
```

or in code:

```python
pipeline = EdgeToMecPipeline(cse_http_url="http://127.0.0.1:8080")
```

`tests/test_middleware_mec.py`'s `TestRealCSEHttpTransport` class
exercises this transport with real HTTP round trips and skips
automatically (rather than failing) when no CSE is reachable.

**What ACME's `--headless` mode needed that isn't obvious from its
own docs**: its normal onboarding wizard is interactive and refuses to
run headless against a fresh directory; `deploy/run_acme_cse.sh`
auto-generates a valid `acme.ini` from the package's own default
template plus the `[basic.config]` values (CSE type/ID/name, DB
backend, HTTP port) the wizard would otherwise ask for. Two other
non-obvious things found while integrating: (1) AE registration's
`api` field must be a valid oneM2M App-ID (start with `R` or `N` —
`ADN_AE`'s default was fixed from `"BDO-SKIN-edge"` to
`"NBDO-SKIN-edge"`); (2) structured (name-based) resource addressing
did not reliably resolve against a self-registered AE on the ACME
version tested, so `HttpCSEClient` tracks each resource's real
`resourceID` from its own CREATE responses and always addresses
children by `ri` afterward — more robust across CSE implementations
regardless of the cause.

### What was found running it

- **URLLC SLA**: at the default link parameters (2.5ms mean, 0.6ms
  jitter, 0.1% simulated radio-link-failure rate), 100% of the
  150-window runs across all four scenarios meet the sub-10ms
  end-to-end transmission-latency target — the rare retransmission
  case pays roughly one extra round-trip and still clears the SLA
  comfortably. This is a link-layer statistical result, not a claim
  about real 5G NR conformance. Latency/SLA numbers are effectively
  identical between the simulated and real-CSE transports, since the
  URLLC link sits between them either way — the real CSE swap changes
  *where the oneM2M CRUD primitives actually execute*, not this layer.
- **Veto logic actually fires, but only when relevant**: 0/150 windows
  in the `normal` scenario ever trigger a veto (no anomaly => natural
  frequency stays clear of every candidate's actuation band); 23-46
  out of 150 windows do in the three anomaly scenarios, concentrated
  where severity is high enough to shift the estimated natural
  frequency into a candidate's resonance guard-band. This is a
  deliberately included stress case
  (`active_damping_resonant_risk` at 33 Hz in
  `mec/bhs_cognitive_engine.py`), not an artifact.
- **Squid's action distribution shifts with severity, not just
  criticality**: baseline weights favor low-disruption actions
  (`hold_position` dominates the `normal` scenario, 60/60 windows);
  `active_damping_resonant_risk` and `full_stop` only start winning as
  `anomaly_severity` climbs in the anomaly scenarios, via the same
  continuous re-weighting for every window rather than a hand-coded
  threshold rule.
- **Bat's RUL projection is undefined, not wrong, on quiet/flat
  history**: when the least-squares trend slope is ~0 (or the fit
  window is under 2 samples), `remaining_useful_life_windows` is
  `None` rather than a meaningless huge or infinite number — surfaced
  directly in the benchmark's "RUL projected in N/150 windows" line
  rather than silently defaulting to something misleading.

### What this does *not* claim

- **`middleware/onem2m_http.py` is a real interop client, but only
  verified against one CSE implementation (ACME).** The HTTP binding
  it speaks (TS-0004 clause 8.2 primitives, `X-M2M-*` headers,
  `m2m:ae`/`m2m:cnt`/`m2m:cin` JSON bodies) is the standard one, so
  tinyIoT/Mobius *should* interoperate without code changes beyond
  `base_url`/`cse_id` — but that hasn't been exercised here, and the
  two ACME-specific quirks noted above (App-ID format strictness,
  structured-addressing behavior) may or may not reproduce identically
  on a different CSE.
- **`middleware/onem2m.py`'s in-memory `MN_CSE`** implements the
  resource-tree shape and CRUD primitive semantics in-process with no
  wire protocol — it's a fast simulation for unit tests, not a second
  interoperability claim; `HttpCSEClient` is what actually satisfies
  "leveraging existing oneM2M...implementations."
- `network/urllc_link.py` is a statistical latency/reliability model
  calibrated to published URLLC figures, not a PHY/MAC radio
  simulator; `mec/mec_platform.py` implements GS MEC 003's
  app-lifecycle and Mp1 service-registry *shapes*, not its NFV/MANO
  orchestration stack or the ETSI MEC sandbox itself. See each
  module's docstring for the exact standard section it follows and
  what it deliberately leaves out.
- **BHS Cognitive Engine's numeric constants are a stimulus/decision
  model, not a re-derivation of BDO-SKIN's own control scheme** — same
  caveat `models/bdo_skin.py` already states for its MAC-cost model.
  Every constant (failure threshold, resonance guard-band, controller
  weights, candidate-action costs) is named and adjustable in
  `mec/bhs_cognitive_engine.py`.

## BDO-SKIN integration: a target-application case study

The `Black_Dragon_Optical_Skin_Research_Report.docx` is a **separate,
unrelated research report** — a structural-health-monitoring digital
twin (distributed fiber-optic strain/temperature sensing + a
Bat/Hermit-Crab/Squid bio-inspired control architecture) for a steel
panel, with no technical connection to HEXA-TPU-RT. Its own reported
numbers come from its own physics/control simulation and are not
re-validated, re-derived, or assumed correct here. What this
integration does is narrower and more honest: treat BDO-SKIN's sensing
workload as a plausible **target application** and ask whether
HEXA-TPU-RT's simulated pipeline (built up over Phases 1-7) can
actually run it — as a new workload/stimulus feeding the *existing*
engine, not a second simulator.

### What was built

Source document: `docs/Black_Dragon_Optical_Skin_Research_Report.docx`
(kept in the repo for provenance/reference — not code, not modified).

| File | Role |
|---|---|
| `models/sensor_events.py` | Sensor/physics-adjacent layer only — no Task, no MAC counts. 600-channel FBG grid (20×30, matching the report's "every 2 cells / every 2 rows" grating spacing over its 60×40 mesh), and four scenario generators: `normal`, `gradual_anomaly`, `burst_anomaly`, `critical_event`. |
| `models/bdo_skin.py` | Translates sensor windows into ordinary `(layer_name, [Task, ...])` tuples — the exact format `Master.load_program()` already consumes. No changes needed anywhere in the core engine for this to work. |
| `benchmarks/bdo_skin_benchmark.py` | Runs all four scenarios, plus a naive-vs-improved pipeline comparison under contention, and prints a verdict. |

Every task BDO-SKIN generates is a normal `Task` object flowing through
the same `Scheduler` → `DMAController` → `PingPongMemory`/`AXIBus` →
`WeightCache` → `MicroTPU`/`SystolicArray` → `PowerModel` path as CNN
and Transformer tasks — confirmed directly by
`test_sensor_data_flows_through_dma_and_cache_not_bypassed`, which
checks `sim.dma`, `sim.cache`, and `sim.axi` all show real traffic
after a BDO-SKIN run, not zero (i.e. not silently bypassed).

Per sensor window, up to three kinds of layers are dispatched:

- **Reflex** (every window): lightweight per-channel processing of all
  600 FBG channels, tiled across a modest subset of workers
  (`num_workers // 3`) — most of the chip stays free, matching
  always-on low-power monitoring. Static kernel parameters, one shared
  cache block, so only the very first window ever pays real AXI cost.
- **BHS cognitive** (periodic, plus immediately after any critical
  window): Bat trend-fit, Hermit Crab candidate-action scoring, Squid
  objective reweighting — three separate layers, moderate cost, tiled
  across about half the chip.
- **Emergency/Reflex** (only on critical windows): full-mesh urgent
  reflex + a full BHS re-evaluation, `priority=-1000` and
  `is_critical=True` (reusing the Phase 6 convention), a genuinely
  tight `deadline_slack_override=1.05` independent of the rest of the
  workload's slack (a new, minimal `Task` field added for this), and
  tiled across **all** workers — the "sudden burst" the integration
  exists to test.

### Finding #1: at generous bandwidth, everything trivially passes — as expected by now

Every scenario completes with zero deadline misses at the default
640 B/cycle AXI width. Consistent with every prior phase's pattern
(Phase 1.5 onward): a generous configuration was never going to be the
real test. The 64 B/cycle comparison below is.

### Finding #2: the Phase 2/6/7 fixes transfer cleanly to a genuinely independent workload

At 64 B/cycle — narrower than what `HOT_BANK_READ_PORTS`-limited
dispatch alone would stress (see the note in
`run_naive_vs_improved_comparison`'s docstring: 128 B/cycle, used
elsewhere in this project, happens to exactly match
`HOT_BANK_READ_PORTS(2) × AXI_BYTES_PER_MAC_CYCLE(64)`, so it never
actually stresses AXI bandwidth on its own for this workload) —
the naive baseline (memory-blind deadlines, FIFO) misses real
emergency-path deadlines in every anomaly scenario:

| Scenario | Naive: emergency misses | Improved: emergency misses |
|---|---|---|
| gradual_anomaly | 38/390 | **0/390** |
| burst_anomaly | 9/100 | **0/100** |
| critical_event (sustained) | 47/480 | **0/480** |

"Improved" = memory-aware + lookahead deadlines (Phase 2/7) + priority
scheduling (Phase 6), the exact combination validated on the
Transformer workload. It reduces emergency-path misses to zero in
every scenario here too — real evidence the fix generalizes, not just
a re-run of Phase 4 under a new name. Full table:
`reports/bdo_skin_integration_v6.txt`.

### Finding #3: DDR energy dominates power here, for a reason worth flagging, not hiding

`PowerModel` reports DDR as ~78-79% of total energy for this workload
— but tracing it down: `HOT_BANK_CAPACITY_ELEMS` (65,536 bytes,
`config.py`'s default) was sized for CNN/Transformer-scale layer
weight loads. BDO-SKIN's actual per-window payload (600 channels) is
tiny by comparison, yet the DMA still fills a full 64KB bank on every
prefetch regardless of how little new data a window's tasks need — a
mismatch between the *reused* memory model's fixed-size assumption and
this workload's much finer granularity, not a property of BDO-SKIN or
of HEXA-TPU-RT's real hardware. Confirmed directly, not just asserted:
right-sizing the bank via `HOT_BANK_CAPACITY_ELEMS=8192` (a config
change, no core rewrite) drops DDR energy from 7857 nJ to 5898 nJ and
total power from 620mW to 499mW, with the completion/deadline results
unchanged. Left at the default in the headline numbers above
deliberately, specifically so this mismatch stays visible rather than
quietly tuned away.

### Finding #4: the reused cache is one block short of BDO-SKIN's real working set

BDO-SKIN's steady-state access pattern touches 5 persistent weight
blocks (`reflex_kernel_params`, `bat_forecast_params`,
`hermit_crab_params`, `squid_params`, `emergency_kernel_params`) — one
more than `CACHE_CAPACITY_BLOCKS`'s default of 4. Measured effect:
93.4% cache hit rate at capacity 4, jumping to 99.4% at capacity 5,
with no other change. A one-line config fix once you know to look for
it — which is exactly why this project keeps running things at
defaults first rather than hand-tuning before reporting.

### What this integration does *not* claim

- **Not a re-validation of BDO-SKIN's own results.** That report's
  numbers come from its own finite-difference panel-physics simulation
  — a completely different kind of model, answering a completely
  different question (does the sensing/control *scheme* work) than
  this one (can *this specific simulated chip* keep up with the data
  it would produce). See `models/sensor_events.py`'s module docstring
  for exactly what is and isn't reproduced from the source document.
- **Not a claim about real HEXA-TPU-RT silicon.** Every number is this
  simulator's architectural model under `config.py`'s documented
  assumptions — the same caveat that has applied to every phase since
  Phase 1.
- **MAC-cost calibration is a stimulus generator, not a spec.** BDO-SKIN's
  report doesn't state a compute cost for its own algorithms (it's a
  physics/control simulation, not a hardware target) — every constant
  in `models/bdo_skin.py` (`REFLEX_OPS_PER_CHANNEL`,
  `EMERGENCY_DEADLINE_SLACK`, etc.) is a named, adjustable assumption
  calibrated to be architecturally plausible, not derived from the
  source document.

Full output: `reports/bdo_skin_integration_v6.txt`.

## Phase 7: closing the gap Phase 4 found

Phase 4 traced 9 residual deadline misses to QKV projection tasks
specifically — the contention estimate is a snapshot at assignment
time, blind to sibling heads about to join the same burst. Phase 7's
fix (`worker.py`'s deadline estimate, gated by
`config.DEADLINE_LOOKAHEAD_ENABLED`) adds same-layer queued tasks with
a genuinely different `weight_block_id` (i.e. guaranteed future AXI
contenders, like the other 7 attention heads) to the concurrency
estimate, instead of only counting workers already running.

Result, directly tested rather than assumed:

| AXI Width | Phase 2 only | Phase 2 + Phase 7 lookahead |
|---|---|---|
| 128 B/cycle | 99/108 | **108/108** |
| 256 B/cycle | 99/108 | **108/108** |
| 384 B/cycle | 102/108 | **108/108** |
| 448+ B/cycle | 108/108 | 108/108 |

Zero misses at every width tested. And critically, it does **not**
regress the CNN benchmark: the lookahead correctly excludes CNN's
same-block tiles (they'll cache-hit, not compete for AXI), so CNN
results are bit-for-bit identical with or without it — confirmed by
`test_phase7_lookahead_does_not_regress_cnn_benchmark`.

## Phase 6: does scheduling policy actually protect critical work?

Two findings, the second more interesting than the first:

**Finding #1 — policy was inert.** Running the transformer workload
under `fifo`, `priority`, and `edf` produced **bit-for-bit identical**
results (99/108 completed, 9 misses, 600116 cycles, all three). Not a
coincidence: Master only ever enqueues one layer at a time, and every
task within that layer shares one priority value — so `min(queue,
key=priority)` always ties and falls back to arrival order. The policy
code works; nothing in any workload before Phase 6 ever gave it a
genuine choice to make.

**Finding #2 — with genuine choice, priority-first dispatch actively
backfires** (without the Phase 7 fix). `models/transformer.py` now
supports `critical_heads_per_layer`, marking specific attention heads
high-priority so the scheduler has real decisions to make. Under
`priority`/`edf`, critical tasks get dispatched *first* — but that
means they're assigned with the *least* information about contention
that hasn't started yet, so their deadline is the most optimistic
(least accurate) one in the batch. Measured result: critical tasks
missed their deadline **100% of the time** vs. 5% for non-critical
ones, identical under `fifo`, `priority`, and `edf` alike:

```
Policy     | Lookahead | TotalMiss |  Critical miss rate | Non-critical miss rate
fifo       |     False |         9 |          4/4 (100%) |             5/104 (5%)
priority   |     False |         9 |          4/4 (100%) |             5/104 (5%)
edf        |     False |         9 |          4/4 (100%) |             5/104 (5%)
```

Enabling Phase 7's lookahead eliminates misses for both groups under
all three policies. **The lesson: for this failure mode, estimate
accuracy dominates scheduling policy entirely** — reordering who goes
first doesn't help if the deadline computed at that moment is wrong.
This is a genuinely useful (and non-obvious) result for whoever
designs the real scheduler: don't assume a priority queue alone buys
safety margin for critical work without also fixing what the deadline
is computed from.

**A gap this exposed, not fixed:** the current deadline is always
relative to a task's own dispatch time, never to when the underlying
request actually arrived. A real hard-RT system under sustained
overload needs "this frame's absolute deadline already passed before
we even started it, drop it" — not representable here, since a task
queued for a long time still gets a fresh, achievable-looking deadline
the moment it finally starts. Flagged as a future architectural gap,
not fixed in this phase.

Full output: `reports/priority_stress_v5.txt`.

## Phase 5: report export

Three export paths, all wired into `benchmark.py single`:

- `--export-json path.json` — `report.as_dict()`, machine-readable
- `--export-html path.html` — single self-contained file (inline SVG
  timeline, no external assets, no matplotlib dependency, opens by
  double-click)
- `--export-png path.png` — matplotlib heatmap of the same timeline
  data as the ASCII gantt, for anyone who wants a shareable plot;
  raises a clear `ImportError` if matplotlib isn't installed rather
  than silently doing nothing

```bash
python3 benchmark.py single --workers 10 --tiles-per-worker 20 --no-cache \
    --axi-width 128 --memory-aware-deadline --lookahead \
    --export-json report.json --export-html report.html --export-png timeline.png
```

Sample outputs: `reports/sample_report_v5.{json,html}`,
`reports/sample_timeline_v5.png`.

## Phase 1-4 recap

Every prior phase's validation ran on the CNN benchmark: uniform task
sizes, symmetric demand across workers, one weight-reuse pattern per
run. Phase 4 asks the question that mattered most: does any of that
hold up on something heterogeneous? `models/transformer.py` builds a
small Transformer encoder specifically to break the easy assumptions:

- **Per-head tasks with zero weight reuse by construction.** Attention
  scores (`Q.K^T`) and attention-weighted values (`softmax(.)V`)
  multiply two *activation* tensors — there's no weight tensor to
  cache at all, unlike CNN's spatial tiling or even the "output_channel"
  worst case from Phase 2.
- **Orders-of-magnitude MAC differences per op type.** FFN layers carry
  ~8x more MACs than attention-score layers in the default config —
  real burstiness, not a synthetic sparsity dial.
- **Structural parallelism mismatch.** `num_heads` (8) < `num_workers`
  (10) by design, so attention phases can't use every worker — a
  bubble caused by insufficient parallel work, not memory bandwidth.

### Finding #1: the "cliff" was never a general property — it was a property of uniform demand

`python3 benchmark.py transformer` runs the same memory-blind vs.
memory-aware comparison from Phase 2, on this workload instead:

| AXI Width | Memory-blind completed | Memory-aware completed |
|---|---|---|
| 128 B/cycle | 59/108 | 99/108 |
| 256 B/cycle | 65/108 | 99/108 |
| 384 B/cycle | 76/108 | 102/108 |
| 448+ B/cycle | 108/108 | 108/108 |

The memory-blind formula **does not livelock here** — it degrades
gradually (59→65→76→108), nothing like the CNN benchmark's 0→1000
binary jump. That means the total-livelock cliff documented in Phase
1.5/2 is a property of *uniform, symmetric* demand specifically — when
every task is the same size and every worker wants the same thing at
the same time, round-robin arbitration either serves everyone or
starves everyone identically. Mixed task sizes break that symmetry and
produce graceful (if still bad) degradation on their own. **This
doesn't mean the Phase 1.5 finding was wrong — it means it was scoped
to a specific demand pattern, and that scope is now explicit rather
than implied.**

### Finding #2: the Phase 2 fix generalizes, but isn't complete

The memory-aware formula still roughly doubles completions at the
worst width tested (59→99 of 108) — a real, substantial improvement,
not an artifact of the easy case. But unlike the CNN benchmark's
near-perfect 994/1000, it leaves residual misses here. Tracing exactly
which tasks miss:

**100% of residual misses are QKV projection tasks** — the largest
per-head task (3.1M MACs vs 524K for attention scores/values), and the
first op dispatched in each block. Root cause: the memory-aware
estimate (`worker.py:estimate_deadline_cycles`) takes a *snapshot* of
concurrent AXI requesters at the moment a task is assigned. When all 8
heads' QKV tasks start in the same handful of cycles (limited to 2/cycle
by `HOT_BANK_READ_PORTS`), the first ones assigned see less contention
than what actually materializes once all 8 have ramped up — so their
deadline is computed too generously. This is the concrete case behind
a caveat Phase 2 already flagged in the abstract ("estimate is fixed
at assignment, doesn't adapt to contention that starts later") — Phase
4 is what turned it from a hypothetical into a specific, reproducible,
tested failure mode. Confirmed directly by
`tests/test_core.py::test_residual_misses_concentrate_in_qkv_proj`.

### Finding #3: a bubble that has nothing to do with memory at all

Even at 640 B/cycle — zero contention, zero deadline misses — worker
occupancy is only 93%, not 100%. `num_heads=8 < num_workers=10` means
attention-score and attention-value phases structurally cannot use two
of the ten workers, no matter how much bandwidth is available. This is
a pure data-parallelism ceiling, invisible to every benchmark run
before Phase 4 because the CNN workload always had at least as many
tiles as workers. Confirmed by
`test_num_heads_below_num_workers_creates_bubble`, which asserts zero
deadline misses alongside sub-100% occupancy specifically to isolate
this from a memory-caused bubble.

Full output: `reports/transformer_stress_v4.txt`.

## Phase 3 recap: power model, and a second real accounting bug it caught

Phase 3 adds an energy/power model specifically to check the spec's two
power claims. Building it surfaced another real bug, in the same
family as Phase 1.5's DMA race: **MAC energy and throughput were being
booked at task *assignment*, not at actual completion.** A task that
got preempted before finishing (livelock scenario from Phase 1.5/2)
was still counted as if it had fully executed — the simulator reported
~0.68 TOPS and near-full MAC energy for a run that completed **zero**
tasks. Fixed by moving accounting from `SystolicArray.cycles_for_task`
(one-shot, at assignment) to `SystolicArray.record_active_cycle`
(incremental, once per cycle of *real, AXI-fed* progress). Throughput
now only counts MACs from tasks that actually finished; energy still
counts all real switching activity, including partial work on tasks
later abandoned — because that energy really was spent, even though it
produced nothing useful. After the fix, the same livelock scenario
correctly reports 0.0 TOPS and a *lower*, more honest power draw
(~95mW of wasted switching + leakage, not ~144mW as if the work had
completed). Covered by 5 new tests, including one asserting the
end-to-end claim directly (`test_livelock_reports_zero_throughput_not_full`).

### Checking claim #1: "up to 40% power cut" from sparsity gating

`python3 benchmark.py power` sweeps sparsity and reports the actual
MAC-power reduction:

| Workload Sparsity | MAC Power Cut | Hits 40%? |
|---|---|---|
| 30% (the default benchmark's assumption) | 28.5% | no |
| 40% | 38.0% | no |
| **42%** | **39.9%** | **~yes** |
| 50% | 47.5% | yes |

The claim is real but **workload-dependent**: it requires roughly 42%+
sparsity to actually hit "40%", and the CNN benchmark used throughout
this project assumes 30% by default. Whether a real deployed model
hits 42% sparsity depends entirely on the model — some do, many edge
CNNs don't without explicit pruning. This should be presented as "up
to 40%, achieved at ~42%+ activation/weight sparsity", not as a
number the hardware guarantees on its own.

### Checking claim #2: "~0.35 W/TOPS"

Under this simulator's (explicitly assumed, unverified) energy
constants, **the design beats the target in every scenario tested that
actually completes work** — 0.187 W/TOPS best case, 0.211 W/TOPS worst
case (no cache, no weight reuse, realistic output-channel tiling).
That's a more favorable result than the spec claims, which is exactly
why it shouldn't be taken as confirmation: it means our assumed
per-MAC and per-byte energy figures happen to be generous relative to
whatever the spec's target was built on. This checks *internal
consistency* of a plausible assumption set, not silicon truth — see
`config.py` for every constant involved, all flagged as order-of-
magnitude assumptions, not vendor figures.

The one case where W/TOPS becomes meaningless rather than bad: below
the Phase 1.5/2 AXI cliff, the chip still burns ~95mW (wasted
switching + static leakage) while completing zero tasks. W/TOPS is
undefined there (0 useful TOPS), not just a bad number — the livelock
itself is the finding, not an efficiency figure.

Full output: `reports/power_analysis_v4.txt`.

## Phase 1/1.5/2 recap

Phase 1 built the pipeline. Phase 1.5 added AXI/DMA/cache/scheduler-
policy/timeline in response to review and found a livelock cliff.
Phase 2 (this version) does the two things Phase 1.5 promised: makes
the cache's tiling assumption explicit instead of hidden, and fixes
the memory-blind deadline formula that caused the cliff. Both are
now directly comparable, not just asserted.

## Phase 2 result #1: the cliff is fixable, and the fix works

`python3 benchmark.py axi-sweep --workers 10` now runs the AXI sweep
under **both** deadline formulas and prints a verdict:

| AXI Width | Memory-blind (Phase 1.5) completed | Memory-aware (Phase 2) completed |
|---|---|---|
| 128 B/cycle | **0 / 1000** | **994 / 1000** |
| 256 B/cycle | 0 / 1000 | 994 / 1000 |
| 384 B/cycle | 0 / 1000 | 994 / 1000 |
| 416–432 B/cycle | 0 / 1000 | 993 / 1000 |
| 448+ B/cycle | 1000 / 1000 | 1000 / 1000 |

The fix: instead of a deadline based only on ideal MAC-cycle count,
the scheduler now estimates each task's fair share of AXI bandwidth
(`bus_width / concurrent_requesters`) at assignment time and stretches
the deadline proportionally (see `worker.py:estimate_deadline_cycles`).
At the narrowest width tested (128 B/cycle, 20% of what 10 workers
actually need), the memory-blind system produces **zero** completed
tasks; the memory-aware one completes **994 of 1000**, with only 6
genuine misses. Full comparison table:
`reports/axi_sweep_phase1.5_vs_phase2_v2.txt`.

**Caveat, stated plainly: this result is almost suspiciously clean,
and that's worth being honest about rather than declaring victory.**
The fair-share estimate matches reality this well because the
benchmark workload has uniform, static, symmetric demand across all
10 workers — exactly the condition round-robin arbitration handles
best. Two things this hasn't tested yet:

1. The estimate is computed once, at task assignment, and held fixed
   for that task's whole duration. If contention changes mid-task
   (another workload phase starts, a worker joins or drops), the
   deadline doesn't adapt.
2. Real workloads aren't this uniform. A bursty or imbalanced demand
   pattern (which Phase 4's Transformer workload should exercise)
   could make the fair-share estimate wrong in either direction —
   too tight (spurious preemptions) or too loose (late results treated
   as on-time). This fix is validated for the case tested, not proven
   in general.

## Phase 2 result #2: the cache assumption is no longer hidden

`config.TILING_STRATEGY` is now explicit: `"spatial"` (Phase 1.5's
default — every tile of a layer shares one weight block, high cache
reuse) or `"output_channel"` (every tile needs a distinct weight
slice, no reuse possible). Both `scaling` and `axi-sweep` accept
`--tiling`, and now agree with each other:

```
$ python3 benchmark.py scaling                        # spatial + cache (optimistic)
   10 workers: 99.9% efficiency, near-linear scaling

$ python3 benchmark.py scaling --tiling output_channel  # realistic
   10 workers: 66.5% efficiency, 1000 deadline misses, sub-linear scaling
   [stall cycles (79.9%), deadline misses / preemption]
```

Under `output_channel` tiling, enabling the cache makes **no
difference** — confirmed directly:
`python3 benchmark.py axi-sweep --tiling output_channel --cache
--no-compare` produces the identical cliff table as cache-disabled.
That's the correct outcome: if every tile needs unique weights, a
cache has nothing to reuse. Phase 1.5's near-linear scaling number was
real, but only under the optimistic tiling assumption — that's no
longer implicit, it's a flag you choose.

Reports: `reports/scaling_spatial_optimistic_v2.txt` vs
`reports/scaling_output_channel_realistic_v2.txt`.

## What Phase 1.5 added (for reference)

| Addition | File | Why |
|---|---|---|
| AXI-style shared data interconnect | `axi.py` | Spec's "AXI4" label only covers the Master↔Memory-Controller *command* bus; the SRAM↔worker *data* path bandwidth isn't specified anywhere. |
| DDR → DMA → SRAM pipeline | `dma.py` | **The spec never mentions external memory at all** — only on-chip SRAM. |
| Weight cache | `cache.py` | Models weight reuse; tiling-strategy-dependent as of Phase 2 (see above). |
| Selectable scheduler policy | `scheduler.py` | Spec says "Hard Real-Time Hardware Scheduler," never states the policy. `fifo` / `priority` / `edf` via `--policy`. |
| ASCII timeline | `timeline.py` | Binned occupancy view per worker/DMA/AXI for spotting bubbles. |

## A bug this exercise caught (Phase 1.5)

Building the DDR/DMA pipeline surfaced a real race condition: the
scheduler used to request the ping-pong bank swap as soon as a
prefetch was *queued*, not once it had actually *completed*. That
produced exactly the kind of "zero-bus-conflict" violation the spec
claims can't happen — caught by a unit test that started failing with
`memory_conflicts=6441`. Fixed by reserving the bank at enqueue time.
See `dma.py`'s `enqueue_prefetch` docstring.

## Install & run

No dependencies outside the standard library.

```bash
cd hexa_tpu_rt_sim

# One simulation, with all the knobs
python3 benchmark.py single --workers 10 --tiles-per-worker 20 \
    --axi-width 128 --tiling output_channel --memory-aware-deadline --timeline

# THE key comparison: memory-blind vs memory-aware deadline, across AXI widths
python3 benchmark.py axi-sweep --workers 10
python3 benchmark.py axi-sweep --workers 10 --tiling output_channel --cache --no-compare

# Worker scaling under both tiling assumptions
python3 benchmark.py scaling
python3 benchmark.py scaling --tiling output_channel

# Compare against publicly known specs of other edge accelerators
python3 benchmark.py comparison --workers 10

# Power analysis: checks the spec's "40% power cut" and "0.35 W/TOPS" claims
python3 benchmark.py power --workers 10

# Transformer stress test: does the Phase 2 fix survive a bursty, heterogeneous workload?
python3 benchmark.py transformer --workers 10

# Priority/EDF scheduling stress test (Phase 6)
python3 benchmark.py priority-stress --workers 10

# Export a run as JSON / self-contained HTML / matplotlib PNG (Phase 5)
python3 benchmark.py single --workers 10 --tiles-per-worker 20 --no-cache \
    --axi-width 128 --memory-aware-deadline --lookahead \
    --export-json report.json --export-html report.html --export-png timeline.png

# BDO-SKIN target-application case study: all 4 scenarios + naive-vs-improved comparison
python3 benchmark.py bdo-skin --workers 10 --windows 150 --compare

# Tests (68, all passing)
python3 -m unittest discover tests -v
```

## What's modeled vs. assumed

| Component | Modeled as | Confidence |
|---|---|---|
| Systolic array cycles | `ceil(effective_macs / (rows*cols))` | Standard, reasonable |
| Sparsity gating | Cycles reduced by workload sparsity fraction | Cycle-savings only, no power model |
| Ping-pong bank switch | 1-cycle swap once DMA transfer completes | Matches spec directly |
| AXI data interconnect | Shared bus, fixed B/cycle, round-robin or priority arbitration | Not in spec — explicit assumption |
| DDR → DMA → SRAM | Fixed access latency + bandwidth-capped burst, prefetch queue with backpressure | Not in spec at all — external memory isn't mentioned |
| Weight cache + tiling strategy | Shared LRU cache; block-sharing depends on selectable `TILING_STRATEGY` | Not in spec — now an explicit choice, not a hidden default |
| Scheduler policy | FIFO / priority / EDF, selectable | Spec doesn't state a policy |
| Hot-bank read ports | N workers can start a new tile per cycle | Assumption, spec gives no port count |
| Cold output banks | One dedicated bank per worker | Matches spec |
| **Deadline formula** | Memory-blind, memory-aware (fair-share), or memory-aware + lookahead, selectable via `DEADLINE_MEMORY_AWARE`/`DEADLINE_LOOKAHEAD_ENABLED` | Lookahead version validated on both uniform (CNN) and heterogeneous (Transformer) workloads — see Phase 7 |
| **Scheduler policy effect** | FIFO / priority / EDF selectable, but inert unless a workload gives tasks genuine per-task priority heterogeneity within one dispatch wave | Confirmed empirically, not assumed — see Phase 6 findings #1 and #2 |
| **Power model** | MAC (active+gated) + SRAM/AXI + DDR + static leakage, all from documented per-unit energy assumptions | Every constant is order-of-magnitude, not vendor-verified — see Phase 3 section above |
| **Deadline model scope** | Relative to a task's own dispatch time, not to when the underlying request arrived | Cannot represent genuine sustained-overload/staleness scenarios — flagged as Phase 8 |
| **BDO-SKIN sensor workload** | 600 FBG channels, 4 scenario generators, MAC costs as a calibrated stimulus generator | Not derived from BDO-SKIN's own report (a physics/control sim, not a hardware spec) — see BDO-SKIN section above |
| **Per-task deadline slack** | `Task.deadline_slack_override`, defaults to `None` (uses global `DEADLINE_SLACK_FACTOR`) | New in the BDO-SKIN integration — lets emergency-path tasks have a genuinely tighter deadline than the rest of a workload |


## Project structure

```
hexa_tpu_rt_sim/
├── config.py       - architectural parameters + tiling strategy + deadline/power toggles
├── memory.py       - PingPongMemory (hot) + ColdOutputMemory (cold)
├── axi.py          - shared data interconnect, arbitrated, bandwidth-limited
├── cache.py        - shared weight cache, LRU, block key depends on tiling strategy
├── dma.py          - DDR latency + burst transfer + prefetch queue, byte tracking      [UPDATED]
├── systolic.py      - 8x8 MAC array; plan_task/record_active_cycle split               [REWRITTEN]
├── worker.py        - Micro-TPU state machine; estimate_deadline_cycles(), incremental
│                       accounting via record_active_cycle                              [UPDATED]
├── scheduler.py      - policy-selectable dispatch, passes AXI-contention estimate
├── master.py          - RISC-V master core: dispatcher + deadline monitor
├── simulator.py        - cycle-by-cycle orchestration; throughput now completion-only,
│                          power computed via PowerModel                                [UPDATED]
├── power.py             - energy/power model: MAC + SRAM/AXI + DDR + static leakage     [NEW]
├── export.py             - JSON / self-contained HTML / matplotlib PNG report export    [NEW]
├── timeline.py             - binned ASCII gantt for spotting bubbles
├── benchmark.py             - CLI: single / scaling / axi-sweep / power / transformer /
│                               priority-stress / bdo-skin / comparison                 [UPDATED]
├── models/cnn.py             - workload generator, tiling_strategy parameter
├── models/transformer.py     - heterogeneous/bursty workload generator,
│                                critical_heads_per_layer
├── models/sensor_events.py    - BDO-SKIN sensor/scenario model (physics-adjacent only)   [NEW]
├── models/bdo_skin.py          - BDO-SKIN -> Task/layer translation                       [NEW]
├── benchmarks/bdo_skin_benchmark.py - all 4 scenarios + naive-vs-improved comparison      [NEW]
└── tests/test_core.py         - 68 unit tests, all passing (12 new for BDO-SKIN)
```

## Roadmap

Phases 1 through 7 are done, plus the BDO-SKIN target-application
integration. What's left:

- **Phase 8**: Absolute (arrival-time) deadlines, not just relative
  (dispatch-time) ones — see Phase 6's finding on this. BDO-SKIN's
  sustained `critical_event` scenario would be a good test case: does
  a real system need to start dropping stale sensor windows under
  sustained overload, not just miss their (still-fresh-looking)
  deadlines?
- **Phase 9**: Multi-stream concurrency — e.g. BDO-SKIN's sensing
  stream running concurrently with a CNN or Transformer inference
  stream sharing the same chip, with real priority arbitration between
  genuinely independent workloads. Phase 6 found within-stream
  priority doesn't help much; this is the scenario where it might.
- **Phase 10**: Extend the power model with per-instance sensitivity
  analysis — sweep the assumed energy constants (config.py) across a
  plausible range, rather than reporting one point estimate. BDO-SKIN's
  DDR-dominated power profile (Finding #3 above) is a good candidate:
  how much of that 78-79% DDR share is the workload's genuine
  granularity vs. an artifact of a hot-bank size tuned for CNN/
  Transformer-scale loads?
- **Phase 11**: Make `HOT_BANK_CAPACITY_ELEMS` and `CACHE_CAPACITY_BLOCKS`
  workload-aware rather than fixed global defaults, so a fine-grained
  workload like BDO-SKIN doesn't need manual config tuning to avoid the
  mismatches Findings #3 and #4 documented.
