# BananaRipeness — Fix Progress (as of 2026-07-07)

Two bugs being fixed, both **non-teachable** (OpenCV post-processing only, no retraining):

1. **Background bleed** — ripeness color analysis read ALL pixels in the
   detection box, so a green stand / yellow shelf next to the banana skewed
   the answer.
2. **Confidence picking** — one fixed threshold can't fit every image
   (0.20 misses occluded bananas; 0.10 creates duplicate boxes on small bunches).

---

## Git state

Clean tree. Rollback + work checkpoints:

| Tag | Commit | Meaning |
|-----|--------|---------|
| `baseline-pre-fixes` | `df03b79` | **Original pre-fix code** — full rollback target |
| `checkpoint/background-fix-wip` | `aee0571` | Latest state (all fixes in) |

Commit history (newest first):
```
8f42880 docs: add design spec + implementation plan for banana fixes
aee0571 fix: per-color-window center-component mask + center-core fallback
c6542cf docs: record background-mask and adaptive-detection fixes
26c03d5 feat: auto detection mode in UI (adaptive pipeline), manual as floor
6384a4e feat: adaptive detection (low-conf inference + relative score filter + NMS dedup)
8b985ca feat: background-resilient banana-mask ripeness analysis
df03b79  ← original
```

Design + plan live in `docs/superpowers/specs/` and `docs/superpowers/plans/`.
`learn/` is gitignored (local notes only).

---

## What's done (all committed)

**Fix 1 — background mask** (`main/vision/banana_mask.py`, new):
- Inside each box, per HSV color family (green/yellow/brown), keep only the
  connected component that touches the box **center**, then union them. A
  same-colored background forms a ring reaching the *edges*, not the center,
  so it's dropped.
- `color_analyzer.analyze_region` now counts green/yellow/brown over the
  banana body only. New keys: `background_suppressed`, `pixel_counts.banana_pixels`.
- Fallbacks: no clean mask → central 60% core; overexposed/white banana →
  low-sat/high-value HSV window; all-background crop → legacy whole-box.

**Fix 2 — adaptive detection** (`main/detection/banana_detector.py`):
- Pass 1 always infers at **0.10** (finds occluded bananas).
- Relative filter: keep `score >= max(best × 0.30, 0.08)`.
- Score-ordered NMS (IoU ≥ 0.50) removes duplicate boxes; `banana_id`
  renumbered by confidence.
- User threshold is now a **floor**, not a hard cutoff.
- UI (`main/app.py`): **Auto** (default, conf 0.10, no floor) / **Manual** (slider = floor).

**Tests** (all passing):
- `main/tests/test_background_resilience.py` — 6/6
- `main/tests/test_adaptive_detection.py` — 5/5
- `main/tests/test_pipeline.py` / `test_bunch_logic.py` — no regressions

**E2E on the 5 real images** (auto mode, YOLOv8n COCO fallback):
```
1.png: 11 boxes  semi-ripe×3, ripe×8    mask_active 2/11   (market, was overcounting/unripe)
2.png: 5 boxes   unripe×5               mask_active 2/5
3.png: 7 boxes   unripe×7               mask_active 1/7
4.png: 6 boxes   semi-ripe×1, unripe×5  mask_active 0/6
5.png: 1 box     unripe×1               mask_active 1/1    ⚠ still reads UNRIPE
```

---

## ⚠ Open issue — image 5 (yellow banana on green background)

The mask is now **active** on image 5, but it still reads **UNRIPE**
(green 99.8%, mean H≈53). Two possibilities to diagnose tomorrow:

1. **The banana in this shot is genuinely unripe/green** — in which case
   "UNRIPE" is *correct* and there's no bug. (Most likely — H=53 is in the
   green band.)
2. **It's a yellow banana and same-family background is still leaking
   through** — a green stand connected to the center through the banana.
   Color alone cannot separate two things in the *same* HSV family; that's
   the documented limit of the color-only approach.

This is the **hard case** (banana and background share a color family).
Different-family cases (yellow banana / green stand) are fixed.

---

## Tomorrow's plan

1. **Diagnose image 5.** Dump the actual pixel values inside the box.
   Decide: correct-unripe vs. same-family leak.
   - If same-family leak → tighten: use box *center* weighting + size
     prior (banana = largest center blob, background = larger edge ring),
     or drop to the center-core path for single-banana boxes.
2. **Validate the NMS/relative-filter constants** on image 1 (11 boxes —
   confirm that's the true count, not missed/dup bananas). The constants
   live as one-liners in `banana_detector.py`:
   `ADAPTIVE_INFERENCE_CONF / RELATIVE_KEEP_RATIO / ABSOLUTE_FLOOR / NMS_IOU_THRESHOLD`.
3. **Full-suite regression** before any release commit:
   ```
   cd main
   python tests/test_background_resilience.py
   python tests/test_adaptive_detection.py
   python tests/test_bunch_logic.py
   python tests/test_pipeline.py
   ```
4. **Manual UI check** — `streamlit run main/app.py`, confirm Auto/Manual
   sidebar + annotated output look right on the 5 images.
5. Only after that: consider `git tag checkpoint/final-verified`.

## How to resume

```
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"
git log --oneline -8          # see where we are
git status                   # should be clean
```
Read this file first. `baseline-pre-fixes` is your safety net if anything
regresses.
