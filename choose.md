# Lighting-Robustness Approaches — Decision Sheet

Context: controlled enclosure + own LEDs, and we may co-design the optics with the
hardware team. Problem being attacked: `color_analyzer.py` thresholds are absolute
HSV values that only hold under one specific light source/exposure.

## Decision inputs (agreed)

- Lighting regime: **controlled enclosure with our own LEDs**
- Hardware co-design: **yes** — we can propose diffuser, reference patch, locked WB/AE

## Approaches

### A — Reference-patch chromatic normalization (recommended core)

**Hardware ask (to hardware team):**
- Matte **white reference patch** in the camera FOV, where bananas can't cover it
- **Diffuser** over the LEDs for even, shadow-free illumination
- LEDs of **known color** (~5000K white), camera locked: manual WB + manual exposure

**Software:** sample patch pixels each frame → estimate illuminant color →
per-channel correction gains → apply before BGR→HSV. Existing thresholds in
`color_analyzer.py` become physically meaningful instead of light-dependent.

**Pros:** attacks the root cause; biggest accuracy win per dollar; gains code is ~40 lines.
**Cons:** patch must stay in FOV (co-design detail); depends on hardware team's build.

### B — Lighting-invariant color features (software-only backup)

Classify on **hue + relative saturation ratios** (green-vs-yellow pixel ratio)
instead of absolute V thresholds; drop hard `V ≥ 80` / `V ≤ 200` gates.

**Pros:** zero hardware cost; graceful degradation.
**Cons:** lower ceiling (uneven light, LED aging still shifts features);
gray-world auto-WB is NOT a tool here — a banana-dominated scene breaks it;
retunes thresholds calibrated by prior batches, so must re-measure.

### C — Two-illumination-snapshot solve

Toggle LEDs between two color temperatures, capture two frames per inference,
solve chromatic adaptation from the pair (real agricultural-imaging technique).

**Pros:** highest accuracy.
**Cons:** two captures per image, LED-toggle hardware, timing complexity —
overkill for Round 1 (70%). **Park as future upgrade.**

## Recommendation

**A as core, B folded in as a softening layer.** The patch makes the thresholds
*right*; ratio-based features make them *stable*. Hardware ask is modest:
white patch + diffuser + locked WB/AE.

## Verification

The 763-image Roboflow set was shot in **market lighting** — it cannot measure
this fix. Honest test: a small **lighting-stress set** photographed by us:
same bananas, same setup, at ~3500K / ~5000K / dim (~0.3× exposure).
Measure agreement before/after with the same tooling as `main/blind_batch.py`.

## Open choice

A+B (recommended) · A-only · B-only
