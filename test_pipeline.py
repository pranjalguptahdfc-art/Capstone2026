"""
test_pipeline.py
----------------
Unit and integration tests for the Banana Bunch Ripeness Detection pipeline.

Tests cover:
  1. Module imports
  2. BananaDetector loading (may use COCO fallback)
  3. RipenessClassifier interface (model file not required for interface tests)
  4. Image loading and cropping utilities
  5. BunchAnalyzer aggregation logic (no model required)
  6. End-to-end pipeline structure

Run:
    python test_pipeline.py

from inside:
    D:\\Pranjal Files\\Capstone2026\\Banana-Ripeness-Bunch\\
with the .venv activated.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print("-" * 60)


def run_test(name: str, fn) -> bool:
    try:
        result = fn()
        if result is False:
            print(f"{FAIL}  {name}")
            return False
        print(f"{PASS}  {name}")
        return True
    except Exception as exc:
        print(f"{FAIL}  {name}")
        print(f"         Exception: {exc}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Module imports
# ──────────────────────────────────────────────────────────────────────────────

def test_imports():
    section("TEST 1 — Module Imports")
    all_ok = True

    def import_detection():
        from detection.banana_detector import BananaDetector, COCO_BANANA_CLASS
        assert COCO_BANANA_CLASS == 46

    def import_classification():
        from classification.ripeness_classifier import (
            RipenessClassifier, CLASS_NAMES, INPUT_SIZE, NUM_CLASSES
        )
        assert NUM_CLASSES == 3
        assert CLASS_NAMES[0] == "Over Ripe"
        assert CLASS_NAMES[1] == "Ripe"
        assert CLASS_NAMES[2] == "Unripe"
        assert INPUT_SIZE == (224, 224)

    def import_processing():
        from processing.bunch_analyzer import BunchAnalyzer, RIPENESS_SCORE
        assert RIPENESS_SCORE["unripe"] == 0
        assert RIPENESS_SCORE["ripe"] == 1
        assert RIPENESS_SCORE["overripe"] == 2

    def import_utils():
        from utils.image_utils import (
            crop_banana, resize_for_classifier, annotate_image,
            load_image_cv2, cv2_to_pil, pil_to_cv2
        )

    all_ok &= run_test("detection module imports", import_detection)
    all_ok &= run_test("classification module imports (class map verified)", import_classification)
    all_ok &= run_test("processing module imports", import_processing)
    all_ok &= run_test("utils module imports", import_utils)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Detector loading
# ──────────────────────────────────────────────────────────────────────────────

def test_detector_loading():
    section("TEST 2 — BananaDetector Loading")
    from detection.banana_detector import BananaDetector

    def load_detector():
        det = BananaDetector(confidence_threshold=0.40)
        assert det._model is not None
        print(f"         → {det.model_label}")
        print(f"         → Fallback: {det.is_fallback}")
        return True

    return run_test("detector loads (fallback OK)", load_detector)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Classifier interface (no model required for interface tests)
# ──────────────────────────────────────────────────────────────────────────────

def test_classifier_interface():
    section("TEST 3 — RipenessClassifier Interface")
    from classification.ripeness_classifier import RipenessClassifier, CLASS_NAMES

    all_ok = True

    def class_map_correct():
        assert CLASS_NAMES[0] == "Over Ripe", f"Expected 'Over Ripe', got {CLASS_NAMES[0]}"
        assert CLASS_NAMES[1] == "Ripe",      f"Expected 'Ripe', got {CLASS_NAMES[1]}"
        assert CLASS_NAMES[2] == "Unripe",    f"Expected 'Unripe', got {CLASS_NAMES[2]}"
        return True

    def classifier_instantiates():
        clf = RipenessClassifier()
        assert not clf._loaded
        return True

    def model_availability_check():
        clf = RipenessClassifier()
        model_path = PROJECT_ROOT / "models" / "banana_ripeness.h5"
        available = clf.is_available()
        print(f"         → Model available: {available}")
        print(f"         → Expected path:   {model_path}")
        if not available:
            print(f"         → (Place EfficientnetBo.h5 at this path to enable classification)")
        return True  # not an error if model is not present yet

    all_ok &= run_test("class mapping: {0:OverRipe, 1:Ripe, 2:Unripe}", class_map_correct)
    all_ok &= run_test("classifier instantiates without loading model", classifier_instantiates)
    all_ok &= run_test("model availability check (may be absent)", model_availability_check)

    # If model exists, also test loading
    clf = RipenessClassifier()
    if clf.is_available():
        def load_classifier():
            clf2 = RipenessClassifier()
            clf2.load()
            info = clf2.inspect_model()
            print(f"         → Input shape:  {info['input_shape']}")
            print(f"         → Output shape: {info['output_shape']}")
            print(f"         → Num classes:  {info['num_classes']}")
            assert info["num_classes"] == 3, f"Expected 3 classes, got {info['num_classes']}"
            return True

        all_ok &= run_test("classifier loads and inspects OK", load_classifier)
    else:
        print(f"{SKIP}  classifier load test (model not yet provided)")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def test_image_utilities():
    section("TEST 4 — Image Utilities")
    import cv2
    from utils.image_utils import crop_banana, resize_for_classifier, annotate_image, cv2_to_pil, pil_to_cv2

    # Create a synthetic 640×480 green image
    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic[:, :] = (30, 150, 30)  # green-ish

    all_ok = True

    def test_crop_normal():
        crop = crop_banana(synthetic, [100, 100, 300, 350], padding=0.05)
        assert crop.ndim == 3
        assert crop.shape[2] == 3
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        print(f"         → Crop shape: {crop.shape}")
        return True

    def test_crop_clipping():
        # Box near image boundary — should not go out of bounds
        crop = crop_banana(synthetic, [0, 0, 50, 50], padding=0.10)
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        return True

    def test_resize():
        crop = crop_banana(synthetic, [100, 100, 300, 350])
        resized = resize_for_classifier(crop, size=(224, 224))
        assert resized.shape == (224, 224, 3)
        return True

    def test_annotate_no_preds():
        dets = [{"banana_id": 1, "bbox": [100, 100, 300, 350], "confidence": 0.90}]
        out = annotate_image(synthetic, dets, predictions=None)
        assert out.shape == synthetic.shape
        return True

    def test_annotate_with_preds():
        dets = [{"banana_id": 1, "bbox": [100, 100, 300, 350], "confidence": 0.90}]
        preds = [{"ripeness_class": "Ripe", "canonical": "ripe", "confidence": 0.88}]
        out = annotate_image(synthetic, dets, predictions=preds)
        assert out.shape == synthetic.shape
        return True

    def test_pil_conversion():
        pil = cv2_to_pil(synthetic)
        back = pil_to_cv2(pil)
        assert back.shape == synthetic.shape
        return True

    all_ok &= run_test("crop_banana — normal box",            test_crop_normal)
    all_ok &= run_test("crop_banana — boundary clipping",     test_crop_clipping)
    all_ok &= run_test("resize_for_classifier → (224,224,3)", test_resize)
    all_ok &= run_test("annotate_image — no predictions",     test_annotate_no_preds)
    all_ok &= run_test("annotate_image — with predictions",   test_annotate_with_preds)
    all_ok &= run_test("PIL ↔ OpenCV conversion roundtrip",   test_pil_conversion)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: BunchAnalyzer aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_bunch_analyzer():
    section("TEST 5 — BunchAnalyzer Aggregation")
    from processing.bunch_analyzer import BunchAnalyzer

    analyzer = BunchAnalyzer()
    all_ok = True

    def test_empty():
        r = analyzer.analyze([])
        assert r["total_bananas"] == 0
        assert r["final_result"] == "unknown"
        return True

    def test_all_ripe():
        bananas = [
            {"banana_id": i, "bbox": [0,0,100,100], "confidence": 0.9,
             "canonical": "ripe", "ripeness_class": "Ripe", "ripe_confidence": 0.9}
            for i in range(1, 6)
        ]
        r = analyzer.analyze(bananas)
        assert r["total_bananas"] == 5
        assert r["counts"]["ripe"] == 5
        assert r["majority_result"] == "ripe"
        assert r["final_result"] == "ripe"
        assert r["avg_ripeness_score"] == 1.0
        return True

    def test_mixed_majority_ripe():
        bananas = [
            {"banana_id": 1, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "unripe",   "ripeness_class": "Unripe",   "ripe_confidence": 0.8},
            {"banana_id": 2, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "ripe",     "ripeness_class": "Ripe",     "ripe_confidence": 0.9},
            {"banana_id": 3, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "ripe",     "ripeness_class": "Ripe",     "ripe_confidence": 0.9},
            {"banana_id": 4, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "ripe",     "ripeness_class": "Ripe",     "ripe_confidence": 0.9},
            {"banana_id": 5, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "overripe", "ripeness_class": "Over Ripe","ripe_confidence": 0.7},
        ]
        r = analyzer.analyze(bananas)
        assert r["total_bananas"] == 5
        assert r["counts"] == {"unripe": 1, "ripe": 3, "overripe": 1}
        assert r["majority_result"] == "ripe"
        assert r["percentages"]["ripe"] == 60.0
        print(f"         → Score: {r['avg_ripeness_score']:.3f}, Majority: {r['majority_result']}")
        return True

    def test_mostly_overripe():
        bananas = [
            {"banana_id": i, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "overripe",
             "ripeness_class": "Over Ripe", "ripe_confidence": 0.85}
            for i in range(1, 8)
        ] + [
            {"banana_id": 8, "bbox": [0,0,10,10], "confidence": 0.9, "canonical": "ripe",
             "ripeness_class": "Ripe", "ripe_confidence": 0.80}
        ]
        r = analyzer.analyze(bananas)
        assert r["majority_result"] == "overripe"
        assert r["avg_ripeness_score"] > 1.6
        return True

    def test_score_calculation():
        # 2 unripe (0) + 6 ripe (1) + 2 overripe (2)  → (0+6+4)/10 = 1.0
        bananas = (
            [{"banana_id": i, "bbox":[0,0,10,10], "confidence":0.9, "canonical":"unripe",   "ripeness_class":"Unripe",   "ripe_confidence":0.8} for i in range(1, 3)] +
            [{"banana_id": i, "bbox":[0,0,10,10], "confidence":0.9, "canonical":"ripe",     "ripeness_class":"Ripe",     "ripe_confidence":0.9} for i in range(3, 9)] +
            [{"banana_id": i, "bbox":[0,0,10,10], "confidence":0.9, "canonical":"overripe", "ripeness_class":"Over Ripe","ripe_confidence":0.7} for i in range(9, 11)]
        )
        r = analyzer.analyze(bananas)
        expected_score = (2*0 + 6*1 + 2*2) / 10   # = 1.0
        assert abs(r["avg_ripeness_score"] - expected_score) < 0.001, \
            f"Expected score {expected_score}, got {r['avg_ripeness_score']}"
        print(f"         → (2×0 + 6×1 + 2×2)/10 = {r['avg_ripeness_score']:.3f}  ✓")
        return True

    all_ok &= run_test("empty input returns unknown", test_empty)
    all_ok &= run_test("all ripe bunch",              test_all_ripe)
    all_ok &= run_test("mixed bunch — majority ripe", test_mixed_majority_ripe)
    all_ok &= run_test("mostly overripe bunch",       test_mostly_overripe)
    all_ok &= run_test("avg score = (2×0+6×1+2×2)/10 = 1.0", test_score_calculation)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: End-to-end pipeline structure (synthetic image, no real model)
# ──────────────────────────────────────────────────────────────────────────────

def test_pipeline_structure():
    section("TEST 6 — End-to-End Pipeline Structure (synthetic)")
    from detection.banana_detector import BananaDetector
    from classification.ripeness_classifier import RipenessClassifier
    from processing.bunch_analyzer import BunchAnalyzer
    from utils.image_utils import crop_banana, annotate_image, resize_for_classifier

    all_ok = True

    def full_pipeline():
        # Step 1: Create a synthetic image
        synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic[100:400, 100:500] = (30, 150, 30)

        # Step 2: Run detector
        det = BananaDetector(confidence_threshold=0.40)
        detections = det.detect(synthetic)
        print(f"         → Detections on synthetic image: {len(detections)}")
        # Synthetic image may get 0 detections — that's expected

        # Step 3: If detections exist, crop and prepare
        if detections:
            for d in detections:
                crop = crop_banana(synthetic, d["bbox"], padding=0.05)
                resized = resize_for_classifier(crop, size=(224, 224))
                assert resized.shape == (224, 224, 3)

        # Step 4: Simulate classification output
        simulated_banana_results = [
            {"banana_id": 1, "bbox": [100, 100, 300, 350], "confidence": 0.92,
             "ripeness_class": "Ripe",     "canonical": "ripe",     "ripe_confidence": 0.88},
            {"banana_id": 2, "bbox": [310, 100, 500, 350], "confidence": 0.87,
             "ripeness_class": "Unripe",   "canonical": "unripe",   "ripe_confidence": 0.79},
            {"banana_id": 3, "bbox": [100, 360, 300, 480], "confidence": 0.95,
             "ripeness_class": "Over Ripe","canonical": "overripe", "ripe_confidence": 0.83},
        ]

        # Step 5: Aggregation
        analyzer = BunchAnalyzer()
        bunch = analyzer.analyze(simulated_banana_results)
        assert bunch["total_bananas"] == 3
        assert bunch["final_result"] in ["unripe", "ripe", "overripe"]
        print(f"         → Majority result: {bunch['majority_result']}")
        print(f"         → Avg score:       {bunch['avg_ripeness_score']:.3f}")
        print(f"         → Final result:    {bunch['final_result']}")

        # Step 6: Annotate image
        annotated = annotate_image(synthetic, simulated_banana_results[:3], [])
        assert annotated.shape == synthetic.shape

        return True

    all_ok &= run_test("full pipeline structure (synthetic)", full_pipeline)
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  [BANANA] BANANA BUNCH RIPENESS - PIPELINE TESTS")
    print("=" * 60)

    results = []
    results.append(test_imports())
    results.append(test_detector_loading())
    results.append(test_classifier_interface())
    results.append(test_image_utilities())
    results.append(test_bunch_analyzer())
    results.append(test_pipeline_structure())

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    if passed == total:
        print(f"  [PASS] ALL {total} TEST GROUPS PASSED")
    else:
        print(f"  [WARN] {passed}/{total} test groups passed")
    print("=" * 60)

    # Reminder about classifier model
    from classification.ripeness_classifier import RipenessClassifier
    clf = RipenessClassifier()
    if not clf.is_available():
        print(
            "\n  [INFO] NEXT STEP:\n"
            "  Download EfficientnetBo.h5 from the vgry5 repository and place it at:\n"
            f"  {clf.model_path}\n"
            "  Then re-run this test to verify the classifier."
        )

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
