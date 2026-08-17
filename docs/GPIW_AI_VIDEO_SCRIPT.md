# GPIW 2026 — AI Video Production Script
### AquaGuardian AI · Target: 90 seconds · MP4, ≤50MB

Two production paths are given. **Option A is recommended.**

- **Option A — Hybrid (recommended):** AI-generated voiceover (e.g. ElevenLabs, Descript) laid
  over real screen-recorded footage of your live console. This is the strongest option because
  it shows the actual working system — the same evidence a juror can verify themselves by running
  the repository — rather than a generic AI-generated depiction of "water infrastructure."
- **Option B — Fully AI-generated visuals:** text-to-video prompts (Runway, Pika, Luma Dream
  Machine, Sora) for scenes where you don't have real footage. Use sparingly — mixing a few
  AI-generated establishing shots with real console footage reads better than an entirely
  synthetic video for a TRL 3 engineering submission.

Every VO line below is timed to speak comfortably in the seconds allotted (~2.4 words/second).
Read through once before recording — trim if you naturally speak faster or slower.

---

## Scene-by-Scene Script

### Scene 1 — Hook (0:00–0:12)
**VO:** "Water infrastructure is still reactive. Leaks are found after the water's gone. Pumps
fail before anyone's warned. Irrigation runs on a schedule, not on what the plant actually needs."

**Option A visual:** Screen recording — open the console, let the ticker strip and hero section
("Detect → React" struck through, "Predict → Prevent") sit on screen for the first few seconds.

**Option B prompt (text-to-video):** *"Slow, low-saturation aerial drone shot over cracked,
drought-stressed farmland transitioning to a close-up of water dripping from a corroded pipe
joint at night, lit by a single work light. Documentary realism, muted teal and amber color
grade, no text overlays, no people."*

**On-screen text:** none yet — let the hook breathe.

---

### Scene 2 — The idea (0:12–0:30)
**VO:** "AquaGuardian AI flips that loop. Before any action reaches a valve, a pump, or a drone,
it's simulated, stress-tested, and validated in a digital twin. If a plan fails — say, closing
a valve that would starve another sector — it's rejected, automatically re-optimized, and
re-tested. Nothing executes until it's proven safe."

**Option A visual:** Screen recording — scroll to the closed-loop pipeline diagram (Sense →
Analyze → Simulate → Stress Test → Optimize → Validate → Execute), run the leak scenario so the
viewer sees the stress-test rejection and retry animate live.

**Option B prompt:** *"Clean 2D motion-graphics animation: seven glowing nodes connected left to
right labeled SENSE, ANALYZE, SIMULATE, STRESS TEST, OPTIMIZE, VALIDATE, EXECUTE, on a dark
teal background. A pulse of light travels along the chain; at STRESS TEST it flashes red, loops
back to OPTIMIZE, then continues in green to EXECUTE. Minimalist, engineering-diagram aesthetic,
no photorealism."*

**On-screen text:** "Decision Before Execution" (fades in under the diagram).

---

### Scene 3 — The evidence (0:30–0:55)
**VO:** "This isn't a mockup. It's a working proof of concept: five water-risk scenarios, run
under four stress profiles each, benchmarked against a reactive baseline. On average: seventy-four
percent less water lost, response delay cut from five minutes to a quarter of a second, and a
stress-test pass rate that goes from zero percent to one hundred percent. All of it reproducible
— the numbers come from code in the repository, not a slide."

**Option A visual:** Screen recording only — this is the section where real footage matters most.
Show the "Quantified against a reactive baseline" stat cards (74%, 99.9%, 0%→100%), then a quick
cut to a terminal running `pytest -q` scrolling to `132 passed`.

**Option B prompt:** *(Not recommended for this scene — fabricated data visuals here would
directly undercut the "reproducible, not a slide" claim in the VO. Use Option A only.)*

**On-screen text:** Large stat callouts synced to the VO: "74% less water loss" → "99.9% faster
response" → "0% → 100% pass rate".

---

### Scene 4 — Honesty about maturity (0:55–1:15)
**VO:** "We're submitting this as what it is: a TRL 3 proof of concept. It hasn't been deployed
in the field, and it doesn't control real infrastructure yet. What it does have is a defined
path to TRL 4 — real sensors, calibration against measured behavior, fail-safe testing — and a
solo founder's background across four connected projects in edge AI, embedded systems, and
validation-before-action design."

**Option A visual:** Screen recording — the "What is proven, and what is not claimed" split panel
(green ✓ / red ✗ lists) if using the pitch deck as a visual source; or a simple slide-style cut
to the TRL roadmap (TRL 3 → 4 → 5–6).

**Option B prompt:** *"Simple, honest infographic style: a horizontal progress bar labeled TRL 1
through TRL 9, with a marker clearly at TRL 3, dark background, teal and white, no photorealistic
elements, text rendered clearly and legibly."* (Text-to-video models render text unreliably —
prefer a real slide export from the pitch deck over an AI-generated version for this shot.)

**On-screen text:** "TRL 3 — Proof of Concept" (persistent lower-third for this scene).

---

### Scene 5 — The ask / close (1:15–1:30)
**VO:** "What we need next is a design partner — a farm, an irrigation district, a small utility
— to run this in shadow mode and tell us where the model is wrong. That's what a GPIW pilot
pathway is built for. Thank you."

**Option A visual:** Screen recording — pull back to the full console homepage, then cut to a
simple title card: project name, "GPIW 2026 — Track A" and contact line.

**Option B prompt:** *"Clean title card animation: 'AquaGuardian AI' in white serif text
fading in on a dark teal background with a subtle animated water-ripple texture behind it,
'GPIW 2026 Submission · Track A' in smaller text below, no additional imagery."*

**On-screen text:** "AquaGuardian AI · GPIW 2026 · Track A (TRL 1–3) · Islam Abdelmogood"

---

## Consolidated Voiceover Script (paste into an AI voice tool)

Use this block as-is in ElevenLabs, Descript Overdub, or similar. Suggested voice settings:
calm, moderate pace, minimal emotional inflection — this is an engineering submission, not an
ad. Total spoken length ≈ 85–90 seconds at a natural pace.

```
Water infrastructure is still reactive. Leaks are found after the water's gone. Pumps fail
before anyone's warned. Irrigation runs on a schedule, not on what the plant actually needs.

AquaGuardian AI flips that loop. Before any action reaches a valve, a pump, or a drone, it's
simulated, stress-tested, and validated in a digital twin. If a plan fails — say, closing a
valve that would starve another sector — it's rejected, automatically re-optimized, and
re-tested. Nothing executes until it's proven safe.

This isn't a mockup. It's a working proof of concept: five water-risk scenarios, run under four
stress profiles each, benchmarked against a reactive baseline. On average: seventy-four percent
less water lost, response delay cut from five minutes to a quarter of a second, and a stress-test
pass rate that goes from zero percent to one hundred percent. All of it reproducible — the
numbers come from code in the repository, not a slide.

We're submitting this as what it is: a TRL 3 proof of concept. It hasn't been deployed in the
field, and it doesn't control real infrastructure yet. What it does have is a defined path to
TRL 4 — real sensors, calibration against measured behavior, fail-safe testing — and a solo
founder's background across four connected projects in edge AI, embedded systems, and
validation-before-action design.

What we need next is a design partner — a farm, an irrigation district, a small utility — to run
this in shadow mode and tell us where the model is wrong. That's what a GPIW pilot pathway is
built for. Thank you.
```

---

## Production Notes

- **Screen recording:** capture at 1080p minimum, 30fps. Run each scenario once beforehand to
  confirm timing before recording — the stress-test-retry animation in Scene 2 takes a few
  seconds; don't cut it off mid-animation.
- **Music:** if adding a background bed, keep it under the voiceover — instrumental, low-key,
  no lyrics, low volume (−20dB or lower relative to VO) so it doesn't compete for attention on a
  jury laptop with modest speakers.
- **File size:** 90 seconds at 1080p30 with a compressed H.264 export should land well under the
  50MB limit; if it doesn't, drop to 720p rather than raise compression artifacts on the on-screen
  text and stat numbers, which need to stay legible.
- **What not to do:** don't let a text-to-video tool generate the Scene 3 evidence numbers or
  the Scene 4 TRL bar — models still render text unreliably, and a garbled or wrong number in the
  one scene that's supposed to prove rigor would be worse than not having AI visuals there at all.
