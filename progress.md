# BananaRipeness — Progress (as of 2026-10-26)

## Reality check (what this repo actually is)

The three plan docs (`progress.md`, `NON_TEACHABLE_GUIDE.md`, `learn/project-notes.md`)
previously described a **341-image EfficientNet ripeness-classifier** workflow
(`train.py`, `model_config.yaml`, `models/current/`, 97%/3% split). **None of
that exists on disk or in git history.** The repo has since pivoted to a
**fully non-teachable architecture** ("No models to train." — `app.py` caption):

```
YOLOv8n COCO (class 46=banana, generic — the fallback IS the production path)
  → adaptive detection (0.10 infer → relative score filter → NMS)
  → banana_mask.py  (center-touching connected-component body isolation)
  → color_analyzer.py (HSV green/yellow/brown → unripe/ripe/semi-ripe/overripe)
  → fusion_engine.py (gas + color → bunch verdict)
```

No `.pt` training, no `.h5` files anywhere. The old `classification/ripeness_classifier.py`
is dead code (not imported by `app.py`; only poked by `test_pipeline.py` interface tests).

**The ground-truth data that exists:** `main/models/Training` = 763-image Roboflow
detection dataset ("Saba Banana Bunch"), split 609 train / 77 valid / 77 test,
YOLO **polygon** labels, 3 ripeness classes per labeled banana object:

| class index | name | labeled objects (all 763 images) |
|---|---|---|
| 0 | overripe | 287 |
| 1 | ripe | 508 |
| 2 | underripe | 757 |

Median 2 labeled bananas per image (0–25). `valid/` + `test/` = **154 images**
is the held-out set for verification (never used by anything, since nothing trains).

---

## Git state

Branch: `blind-batch-verify` (off `main` @ `9c117df`).

| What | Status |
|---|---|
| `main/blind_batch.py` (new) | Held-out batch runner: live pipeline vs. Training labels |
| `main/tests/test_blind_batch.py` (new) | 12 unit tests — all passing |
| `main/tests/blind_batch_report.md` | 20-image smoke-batch report |
| All existing suites | `test_background_resilience` 6/6, `test_adaptive_detection` 5/5, `test_pipeline` 6/6, `test_bunch_logic` standalone — no regressions |

---

## Blind-batch — 20-image smoke results (2026-10-26)

Ran the **live pipeline** (auto/adaptive mode, conf 0.10 no floor, IoU≥0.3 match)
on 20 held-out images (all 7 overripe-class images + 13 others, `valid/`+`test/`).
Mapping: `unripe→underripe`; `semi-ripe` = acceptable 50/50 band between ripe/underripe.

| metric | value |
|---|---|
| images | 20 ok / 0 errored |
| mean detection recall (19 labelled) | **66%** |
| images with zero detections | 0 |
| matched bananas | 26 |
| ripeness agreement | **11/26 (42%)** |
| spurious detections (no label overlap) | 63 |

### Findings (both are non-teachable threshold work, not retraining)

1. **Detector over-detects dense bunches** — `bunch_291`: 2 labels, 11 boxes;
   `bunch_647`: 4 labels, 10 boxes. The relative score filter + NMS keeps
   overlapping sub-bananas. 63 spurious boxes across 20 images.
2. **`brown > 20%` over-calls OVERRIPE** on ripe bananas with brown spotting —
   e.g. `bunch_101` label *ripe*: brown 99% → overripe; `bunch_432` label *ripe*:
   brown 81.9%. 10 of 15 mismatches are overripe over-calls.
3. **Yellow-heavy "underripe" labels read RIPE** — `bunch_79` (label underripe,
   yellow 99.6%), `bunch_45` (yellow 99%). These are early-ripe yellow bananas the
   labels call underripe: either label noise or the H-boundary (35) is too permissive.

Per-image + confusion tables: `main/tests/blind_batch_report.md`.

---

## Next

1. **Scale the batch** — `python main/blind_batch.py --all` (154 images, ~15 min
   on CPU; the smoke run was ~17 s total after the 8.7 s first-image torch warmup).
   Read the aggregates before tuning anything.
2. **Non-teachable threshold fixes** (rule-based, per `NON_TEACHABLE_GUIDE.md`):
   - overripe over-call: demote `brown > 20%` when `yellow ≥ ~50%` (spotted-ripe
     vs. rot-ripe distinction), or raise the brown floor;
   - over-detection: tighten `RELATIVE_KEEP_RATIO` / `NMS_IOU_THRESHOLD`
     against the 63 spurious boxes (measure first — full batch in step 1).
3. **Manual UI spot-check** — `streamlit run main/app.py` on 3–4 images.
4. Only after the above is clean: consider removing the dead
   `classification/ripeness_classifier.py` + duplicate `main/bunch_analyzer.py`
   (needs user sign-off — separate task).

## How to resume

```
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"
git log --oneline -6          # where we are
python main/blind_batch.py --all     # the 154-image batch
python main/tests/test_blind_batch.py  # 12 unit tests (or: python -m pytest main/tests/ -q)
```

Read this file first. `main` @ `9c117df` is the safety net.
