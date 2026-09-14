#!/usr/bin/env python3
"""
diagnose_classifier.py
----------------------
Standalone diagnostic for the banana ripeness classifier (EfficientnetBo.h5).

Verifies:
1. Model loads successfully with Keras 3 compatibility handling.
2. Preprocessing and inference on a synthetic 224x224x3 image.
3. Softmax probabilities sum to approximately 1.
4. Class mapping: 0 = Over Ripe, 1 = Ripe, 2 = Unripe.

Does not modify, retrain, overwrite, or convert the model file.
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.ripeness_classifier import RipenessClassifier, CLASS_NAMES

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"


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


def test_model_loads():
    """Test that the classifier loads the model from models/EfficientnetBo.h5."""
    clf = RipenessClassifier()
    if not clf.is_available():
        print(f"         → Model file not found at: {clf.model_path}")
        return False
    clf.load()
    if not clf._loaded:
        print("         → Model failed to load")
        return False
    print(f"         → Loaded from: {clf.model_path}")
    return True


def test_dummy_inference():
    """Run inference on a synthetic image and verify outputs."""
    clf = RipenessClassifier()
    if not clf.is_available():
        print("         → Model not available, skipping inference test")
        return False
    clf.load()

    # Create a synthetic 224x224x3 image (mid-gray)
    synthetic = np.full((224, 224, 3), 128, dtype=np.uint8)

    # Preprocess using the classifier's method
    processed = clf.preprocess(synthetic)
    # Expected shape: (1, 224, 224, 3)
    if processed.shape != (1, 224, 224, 3):
        print(f"         → Unexpected processed shape: {processed.shape}")
        return False

    # Get prediction (includes raw logits and probabilities)
    pred = clf.predict(synthetic)

    # Extract required information
    raw_logits = pred["raw_logits"]
    probabilities = pred["all_scores"]
    class_index = pred["class_index"]
    class_name = pred["ripeness_class"]
    confidence = pred["confidence"]

    # Verify probabilities sum to approximately 1
    prob_sum = sum(probabilities)
    if not abs(prob_sum - 1.0) < 1e-3:
        print(f"         → Probabilities sum to {prob_sum}, expected ~1.0")
        return False

    # Verify class mapping
    expected_names = ["Over Ripe", "Ripe", "Unripe"]
    if class_index < 0 or class_index >= len(expected_names):
        print(f"         → Invalid class index: {class_index}")
        return False
    if CLASS_NAMES[class_index] != expected_names[class_index]:
        print(f"         → Class name mismatch for index {class_index}")
        return False

    # Print the diagnostic information
    print(f"         → Raw logits: {raw_logits}")
    print(f"         → Probabilities: {probabilities}")
    print(f"         → Predicted class index: {class_index}")
    print(f"         → Predicted class name: {class_name}")
    print(f"         → Confidence: {confidence}")
    print(f"         → Probability sum: {prob_sum:.6f}")
    print(f"         → Class mapping verified: 0=Over Ripe, 1=Ripe, 2=Unripe")

    return True


def main():
    print("\n" + "=" * 60)
    print("  🍌 BANANA RIPENESS CLASSIFIER DIAGNOSTIC")
    print("=" * 60)

    results = []
    results.append(run_test("Model loads successfully", test_model_loads))
    results.append(run_test("Dummy inference and verification", test_dummy_inference))

    passed = sum(results)
    total = len(results)

    print("\n" + "═" * 60)
    if passed == total:
        print(f"  ✅ ALL {total} TESTS PASSED")
        print("  📋 Classifier is ready for YOLO integration.")
    else:
        print(f"  ⚠️  {passed}/{total} tests passed")
        print("  📋 Please check the errors above.")
    print("═" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()