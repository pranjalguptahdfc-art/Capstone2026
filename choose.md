# Lighting-Robustness Approaches — Decision Sheet

> **Status (2026-10-28):** **B is done** — `main/vision/color_model.py`
> (exclusive color families + a relative brightness gate) is now the single
> source of color truth, and the mask + analyzer both consume it. Measured on
> the 154-image gate: ripeness agreement **20% → 57%**, detection unchanged.
> **A is pending — it is hardware-gated** (the white reference patch and the
> locked-WB camera don't exist yet; there is no camera code, the app runs on
> static uploads). **C is parked.** See `progress.md` for the numbers.
>
> **(Same evening, out of scope for this sheet):** a **gated slice-boost
> detection layer** (overlapping-tile inference, `detection/banana_detector.py`)
> fixed far-distance bunches — the 335×597 truck image went 4 → 9 detections
> while close-range images stay bit-identical. Plus analytics-nav crash fix +
> button centering. Details in `progress.md` / `learn/project-notes.md`.
>
> **(Next round, out of scope for this sheet):** **far-band rescue +
> containment dedup** in `detection/banana_detector.py`. The slice-boost
> gate proved inert on the far truck image (best conf 0.707 ≥ 0.35 gate),
> so its far crates were still missed: a far-band crop pass now fires when
> the top half of the frame has zero box centers (small images only) and
> added the missing far crate (9 → 10 boxes on `images.jfif`); a
> containment rule (drop a box fully inside a stronger kept box) removed
> the nested-crate/box duplicate pair on working images
> (`images (1)/(8).jfif` 14 → 9 each, 5 verified drops each). Working
> close-range images verified unchanged. Details in `progress.md` /
> `learn/project-notes.md`.

> **(Software freeze, 2026-10-29):** detection scope is **frozen for the
> Nov-4 demo** at commit `b66a0b6` (81/81 tests green). Two follow-up
> workstreams are documented in `progress.md` ("Hardware integration
> plan") and are NOT part of this sheet:
>
> 1. **Hardware handoff** — per-banana ripeness table → automatic CSV
>    download → gas-sensor calibration done in-repo against that CSV →
>    `analytics_view.py` update → one-line verdict summary rendered on the
>    ESP32 LCD (e.g. *"Acceptable, all banana bunches are mostly ripe.
>    Few bananas will take 1 day to ripen."*).
> 2. **Far-distance retraining round** — one YOLO model trained on a
>    combined close + far + far-crop dataset (labeling via auto-labeling
>    suggested). Decision recorded: **one combined model**, not two
>    (close + far) to merge — model weight-averaging is not a valid
>    operation across differently-shaped networks, and no off-the-shelf
>    far-banana YOLO exists to download. The frozen app pipeline prefers
>    `models/banana_detector_single.pt` over the COCO fallback, so a
>    retrained model drops in with zero app-code changes; it is validated
>    through the same regression harness before the swap is kept.

Context: controlled enclosure + own LEDs, and we may co-design the optics with the
hardware team. Problem being attacked: `color_analyzer.py` thresholds are absolute
HSV values that only hold under one specific light source/exposure.

## Decision inputs (agreed)

- Lighting regime: **controlled enclosure with our own LEDs**
- Hardware co-design: **yes** — we can propose diffuser, reference patch, locked WB/AE

## Approaches

### A — Reference-patch chromatic normalization (recommended core) ⏳ PENDING HARDWARE

> **Status (2026-10-28):** not started — blocked on hardware (patch +
> locked-WB camera don't exist; there's no camera code yet, the app runs on
> static uploads). The ~40-line software side (sample patch → per-channel
> gains → apply before BGR→HSV) can be built config-gated and unit-tested on
> synthetic patch images, but it ships untestable on real light until the
> enclosure lands. The user plans an **A+B round** after hardware exists.

**Hardware ask (to hardware team):**
- Matte **white reference patch** in the camera FOV, where bananas can't cover it
- **Diffuser** over the LEDs for even, shadow-free illumination
- LEDs of **known color** (~5000K white), camera locked: manual WB + manual exposure

**Software:** sample patch pixels each frame → estimate illuminant color →
per-channel correction gains → apply before BGR→HSV. Existing thresholds in
`color_analyzer.py` become physically meaningful instead of light-dependent.

**Pros:** attacks the root cause; biggest accuracy win per dollar; gains code is ~40 lines.
**Cons:** patch must stay in FOV (co-design detail); depends on hardware team's build.

### B — Lighting-invariant color features (software-only backup) ✅ SHIPPED

Classify on **hue + relative saturation ratios** (green-vs-yellow pixel ratio)
instead of absolute V thresholds; drop hard `V ≥ 80` / `V ≤ 200` gates.

**Shipped as `main/vision/color_model.py` (2026-10-28).** Exclusive families
(warm H0–34 / green H35–85, no hue overlap) plus a relative brown gate
(`brown = warm ∧ V ≤ 0.3 × p95(V of the crop)`). 0.30 is a measured optimum on
the full 154-image gate. `color_analyzer.py` + `banana_mask.py` both import it;
the old `HSV_RANGES` is deleted.

**Pros:** zero hardware cost; graceful degradation.
**Cons:** lower ceiling (uneven light, LED aging still shift features);
gray-world auto-WB is NOT a tool here — a banana-dominated scene breaks it;
a **uniformly dull** dark banana reads ripe (p95(V) = its own dark level, so
nothing is "dark relative to the crop") — that residual is exactly what **A**
closes, since a white patch gives an *absolute* illuminant to compare against.
This limit is why A is the later round, not B.

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

> **Chosen path (2026-10-28):** **B-only for now** — B shipped because it's
> software-only and verifiable today; **A deferred** pending hardware;
> **C parked** as a future upgrade. When the enclosure + locked camera exist,
> do the **A+B round** (A on top of the B model above).

## Verification

The 763-image Roboflow set was shot in **market lighting** — it cannot
measure the *lighting* axis, but it is the correct regression gate for "did
we break what worked." Used exactly that: `main/blind_batch.py --all` on the
154-image held-out set, before vs after.

**Honest lighting test:** a small **lighting-stress set** photographed by us:
same bananas, same setup, at ~3500K / ~5000K / dim (~0.3× exposure).
`main/tests/verify_images.py` is the tool for it — point it at any image
folder (incl. `.jfif`), it runs the real pipeline and writes per-image JSON
(boxes + per-banana G/Y/B% + ripeness class + bunch verdict) and annotated
PNGs. The B round's before/after on the user's root images lives in
`main/tests/verify_before_full11/` (old model) vs `main/tests/verify_after/`
(new model).

## Open choice (resolved)

**B-only ✅ (shipped 2026-10-28)** · A+B → later round (pending hardware) ·
A-only / C → not taken.

**Lighting-axis decision is closed as of the Nov-4 demo (2026-10-29):** B
stays the color path until the enclosure exists; A+B and C remain the
post-hardware rounds. The hardware handoff workflow (CSV table → in-repo
gas calibration → LCD one-liner) and the far-distance retraining decision
(one combined model, no merging) are recorded above under "Software
freeze" and in `progress.md`.
