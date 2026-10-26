# BananaRipeness — Progress (as of 2026-10-26)

## Where this version is (`main` @ `d00ffad` = GitHub)

Original non-teachable pipeline + verification tooling. Behavior summary:

```
YOLOv8n COCO (class 46=banana, pre-built fallback — the production path)
  → adaptive detection (0.10 infer → relative score filter → NMS)
  → banana_mask.py  (center-touching connected-component body isolation)
  → color_analyzer.py (HSV; brown > 20% → overripe)
  → fusion_engine.py (gas + color → bunch verdict)
```

What this version does **not** include (measured fixes, kept on their branches):

| Branch | Tip | What it adds | Measured effect (154-image batch) |
|---|---|---|---|
| `fix/overripe-gate` | `1c375aa` | `decide_ripeness` gate: overripe needs brown > 15% AND yellow ≤ 35% AND brown leading | agreement 20% → 53% |
| `fix/green-background-mask` | `ec780d8` | gate + mask fix: no shrink-remask loop, coverage-qualified center rule, 2% floor (fixes yellow-banana-on-green-bg → UNRIPE) | 53% → 68% |
| `fix/nested-box-removal` | `dfdbace` | drop any box that lies inside a bigger box (overlap rule) | not yet measured |

Known issues in THIS version (what those branches fix):
- **Overripe over-calls** — `brown > 20%` fires on stained-ripe bananas (164/171 predicted-overripe wrong in the full batch).
- **Green-background misread** — yellow banana on a solid green background reads UNRIPE (center-touching rule keeps the background; the old "image 5" open issue).
- **Spurious/overlapping boxes** — 615 across the 154-image set; nested-box fragments survive NMS.
- **Requirements** — `requirements.txt` now includes `pandas` (the app's per-banana table) and `pyserial` (ESP32 path); `scikit-learn` was unused.

## Reality check (what this repo actually is)

Earlier plan docs described a 341-image EfficientNet training workflow — that never
existed on disk or in git history. The repo is a fully non-teachable architecture
("No models to train." — `app.py` caption). Ground truth for verification is
`main/models/Training`: 763-image Roboflow detection set (609/77/77 split), YOLO
polygon labels, 3 classes: 0=overripe (287 objects), 1=ripe (508), 2=underripe (757).
`valid/`+`test/` = 154 held-out images; `blind_batch.py` measures the live pipeline
against them. `classification/ripeness_classifier.py` is dead code (legacy, not
imported by `app.py`). Six uploaded test images live at the repo root.

## Git state

| Where | Commit |
|---|---|
| `main` (local **and** GitHub) | `d00ffad` — runner + smoke report + test images |
| `main`'s parent | `4aa88a1` — blind-batch runner + tests |
| GitHub `main` before this push | `df03b79` (pre-runner) |

Test suites on this version: `test_blind_batch` 12, `test_background_resilience` 6,
`test_adaptive_detection` 5, `test_pipeline` 6, `test_bunch_logic` standalone —
29/29 via pytest + standalone, all green.

## Blind-batch — 20-image smoke results (measured on this code)

Auto mode, IoU ≥ 0.3 matching, `unripe→underripe` mapping, `semi-ripe` = OK band:

| metric | value |
|---|---|
| mean detection recall | 66% |
| ripeness agreement | 11/26 (42%) |
| spurious boxes | 63 |

Full 154-image aggregates (measured before the gate/mask fixes): agreement 20%,
615 spurious boxes — that's the "before" column for the branch table above.
Per-image + confusion tables: `main/tests/blind_batch_report.md`.

## Next

1. **Decide which fixes to ship** — overripe gate, mask fix, and/or the
   nested-box rule, one at a time, each measured with `python main/blind_batch.py --all`.
2. **Manual UI spot-check** — `streamlit run main/app.py` with the 6 root test images
   (note: expect the known overripe/green-bg misreads at this version).
3. **Batch rerun after any fix** — regenerate the report, compare before/after.
4. Optional cleanup: remove dead `classification/ripeness_classifier.py` +
   duplicate `main/bunch_analyzer.py` (needs sign-off).

## How to resume

```
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"
git log --oneline -4            # d00ffad = main = GitHub
python main/blind_batch.py --all   # 154-image measurement (~4 min CPU)
python -m pytest main/tests/ -q    # 29 tests
```
