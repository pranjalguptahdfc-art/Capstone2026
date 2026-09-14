#!/usr/bin/env python3
"""
diagnose_classifier_isolation.py
--------------------------------
Diagnostic script to test the banana ripeness classifier in isolation
on two clean images: a yellow/ripe banana and a green/unripe banana.

Usage:
    python diagnose_classifier_isolation.py

Expects the following files to exist:
    inputs/yellow_ripe_clean.jpg
    inputs/green_unripe_clean.jpg

The script:
1. Loads the existing RipenessClassifier (models/EfficientnetBo.h5)
2. For each image, runs the classifier's predict() method
3. Prints: filename, predicted class, Over Ripe probability, Ripe probability,
   Unripe probability, highest probability, and raw logits
4. Provides a clear diagnostic conclusion based on the results.

IMPORTANT: No modifications to the model, preprocessing, or classifier code.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.ripeness_classifier import RipenessClassifier

def main():
    print("=" * 70)
    print("CLASSIFIER ISOLATION DIAGNOSTIC")
    print("Testing RipenessClassifier on clean yellow and green banana images")
    print("=" * 70)

    # Define expected input paths
    yellow_path = PROJECT_ROOT / "inputs" / "yellow_ripe_clean.jpg"
    green_path = PROJECT_ROOT / "inputs" / "green_unripe_clean.jpg"

    # Check that files exist
    missing = []
    if not yellow_path.exists():
        missing.append(str(yellow_path))
    if not green_path.exists():
        missing.append(str(green_path))

    if missing:
        print("[ERROR] Missing required input files:")
        for m in missing:
            print(f"    {m}")
        print("\nPlease place the two clean banana images in the inputs/ directory.")
        print("Expected: inputs/yellow_ripe_clean.jpg and inputs/green_unripe_clean.jpg")
        sys.exit(1)

    # Load classifier (existing code, no changes)
    print("\n[INFO] Loading RipenessClassifier...")
    classifier = RipenessClassifier()
    if not classifier.is_available():
        print("[ERROR] Ripeness classifier model not found!")
        print("        Please place EfficientnetBo.h5 in models/ directory")
        sys.exit(1)

    try:
        classifier.load()
        print("[SUCCESS] Classifier loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")
        sys.exit(1)

    # Helper to process and print results for one image
    def process_image(image_path, label):
        print(f"\n{'-' * 50}")
        print(f"Processing {label}: {image_path.name}")
        print('-' * 50)

        # Load image using OpenCV (BGR)
        import cv2
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            return None

        # Run prediction via existing classifier
        try:
            pred = classifier.predict(bgr_image)
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None

        # Extract results
        class_index = pred["class_index"]          # 0,1,2
        ripeness_class = pred["ripeness_class"]    # string: "Over Ripe", "Ripe", "Unripe"
        confidence = pred["confidence"]            # max probability
        all_scores = pred["all_scores"]            # list [Over Ripe, Ripe, Unripe] probabilities
        raw_logits = pred["raw_logits"]            # list of raw logits

        # Print formatted results
        print(f"Filename:           {image_path.name}")
        print(f"Predicted class:    {ripeness_class} (index {class_index})")
        print(f"Confidence:         {confidence:.6f}")
        print(f"Probabilities:")
        print(f"  Over Ripe (0):    {all_scores[0]:.6f}")
        print(f"  Ripe (1):         {all_scores[1]:.6f}")
        print(f"  Unripe (2):       {all_scores[2]:.6f}")
        print(f"Highest probability: {max(all_scores):.6f} (class {np.argmax(all_scores)})")
        print(f"Raw logits:")
        print(f"  [0] Over Ripe:    {raw_logits[0]:.6f}")
        print(f"  [1] Ripe:         {raw_logits[1]:.6f}")
        print(f"  [2] Unripe:       {raw_logits[2]:.6f}")

        return {
            "filename": image_path.name,
            "predicted_class": ripeness_class,
            "class_index": class_index,
            "confidence": confidence,
            "probabilities": all_scores,
            "raw_logits": raw_logits,
        }

    # Process both images
    yellow_result = process_image(yellow_path, "YELLOW/Ripe")
    green_result = process_image(green_path, "GREEN/Unripe")

    # Print summary and diagnosis
    print("\n" + "=" * 70)
    print("CLASSIFIER ISOLATION TEST SUMMARY")
    print("=" * 70)

    if yellow_result is None or green_result is None:
        print("[ERROR] One or both images failed to process. Check above messages.")
        sys.exit(1)

    # Format for summary
    def fmt(res):
        return (f"Prediction: {res['predicted_class']:<10} "
                f"Over Ripe: {res['probabilities'][0]*100:5.2f}% "
                f"Ripe: {res['probabilities'][1]*100:5.2f}% "
                f"Unripe: {res['probabilities'][2]*100:5.2f}%")

    print("Yellow crop:")
    print(fmt(yellow_result))
    print("\nGreen crop:")
    print(fmt(green_result))

    # Determine diagnosis
    yellow_pred = yellow_result["predicted_class"]
    green_pred = green_result["predicted_class"]
    yellow_prob_ripe = yellow_result["probabilities"][1]   # index 1 = Ripe
    green_prob_ripe = green_result["probabilities"][1]

    print("\nDIAGNOSIS:")
    if yellow_pred == "Ripe" and green_pred == "Unripe":
        print("CLASSIFIER PASSES ISOLATION TEST — the classifier can distinguish clean ripe and unripe bananas.")
        print("The main problem is therefore likely in the fused-bunch detection/cropping pipeline.")
    elif yellow_pred == "Ripe" and green_pred == "Ripe" and green_prob_ripe > 0.8:
        print("CLASSIFIER FAILURE SUSPECTED — the classifier is predicting Ripe even for a clean green banana.")
        print("Further classifier/data/preprocessing investigation is required.")
    else:
        print("CLASSIFIER BEHAVIOR INCONCLUSIVE — inspect the crops and probabilities before changing anything.")

    print("\n" + "=" * 70)
    print("END OF DIAGNOSTIC")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())