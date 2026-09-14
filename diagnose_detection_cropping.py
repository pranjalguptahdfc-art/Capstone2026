#!/usr/bin/env python3
"""
diagnose_detection_cropping.py
------------------------------
Diagnostic script for YOLO detection & cropping stage on a mixed-ripeness bunch image.

Requirements:
- Accepts a JPG/JPEG/PNG image path from Jewelry command line.
- Uses existing BananaDetector and RipenessClassifier without modification.
- Saves each detected banana crop (with 5% padding) to outputs/detection_crops/
- Creates outputs/detection_annotated.jpg with bounding boxes, IDs, and YOLO confidence.
- For each crop, prints detection ID, bounding box, YOLO confidence, predicted ripeness,
  and probabilities [Over Ripe, Ripe, Unripe].
- Prints total number of detections.
- Diagnostic only: no changes to models, thresholds, or code.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import BananaDetector
from classification.ripeness_classifier import RipenessClassifier
from utils.image_utils import crop_banana

def create_output_dirs():
    outputs_dir = PROJECT_ROOT / "outputs"
    crops_dir = outputs_dir / "detection_crops"
    outputs_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    return outputs_dir, crops_dir

def get_image_path():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if image_path.exists():
            return image_path
        else:
            print(f"[ERROR] Image file not found: {image_path}")
            sys.exit(1)
    else:
        image_path = input("Enter path to the mixed-ripeness banana bunch image: ").strip()
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[ERROR] Image file not found: {image_path}")
            sys.exit(1)
        return image_path

def main():
    print("=" * 70)
    print("YOLO DETECTION & CROPPING DIAGNOSTIC")
    print("Visualizing each detection with crop, confidence, and classifier output")
    print("=" * 70)

    image_path = get_image_path()
    print(f"[INFO] Using image: {image_path}")

    outputs_dir, crops_dir = create_output_dirs()
    print(f"[INFO] Outputs -> {outputs_dir}")
    print(f"[INFO] Crops   -> {crops_dir}")

    # Load models exactly as existing
    print("\n[INFO] Loading YOLO detector (default conf=0.40) ...")
    detector = BananaDetector(confidence_threshold=0.40)
    print(f"      Model label: {detector.model_label}")
    print(f"      Using fallback? {detector.is_fallback}")

    print("[INFO] Loading ripeness classifier ...")
    classifier = RipenessClassifier()
    if not classifier.is_available():
        print("[ERROR] Ripeness classifier model not found! Place EfficientnetBo.h5 in models/")
        sys.exit(1)
    try:
        classifier.load()
        print("[SUCCESS] Classifier loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")
        sys.exit(1)

    # Load image
    print(f"\n[INFO] Loading image: {image_path}")
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)
    print(f"      Image shape: {bgr_image.shape} (H×W×C)")

    # Run detection
    print("\n[INFO] Running YOLO detection ...")
    detections = detector.detect(bgr_image, confidence_threshold=0.40)

    if not detections:
        print("[WARNING] No bananas detected!")
        print("        Try lowering confidence threshold or check image.")
        sys.exit(1)

    print(f"[SUCCESS] {len(detections)} banana(s) detected.")

    # Prepare annotated image (copy)
    annotated = bgr_image.copy()

    # Process each detection
    print("\n" + "=" * 70)
    print("PER-DETECTION RESULTS")
    print("=" * 70)

    for det in detections:
        banana_id = det["banana_id"]
        bbox = det["bbox"]  # [x1, y1, x2, y2] as floats
        yolo_conf = det["confidence"]

        # Convert bbox to int for drawing/cropping
        x1, y1, x2, y2 = map(int, bbox)

        print(f"\nBanana {banana_id:03d}:")
        print(f"  Bounding Box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  YOLO Confidence: {yolo_conf:.3f}")

        # Draw box on annotated image
        colour = (0, 255, 0)  # green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        label = f"ID:{banana_id:03d} {yolo_conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), colour, -1)
        cv2.putText(annotated, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Crop banana (with 5% padding)
        try:
            crop = crop_banana(bgr_image, bbox, padding=0.05)
        except Exception as e:
            print(f"  [ERROR] Cropping failed: {e}")
            continue

        # Save crop
        crop_filename = f"banana_{banana_id:03d}.jpg"
        crop_path = crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        print(f"  Crop saved: {crop_path}")
        print(f"  Crop shape: {crop.shape}")

        # Classify crop
        try:
            pred = classifier.predict(crop)
        except Exception as e:
            print(f"  [ERROR] Classification failed: {e}")
            pred = {
                "class_index": -1,
                "ripeness_class": "Error",
                "confidence": 0.0,
                "all_scores": [0.0, 0.0, 0.0],
                "raw_logits": [0.0, 0.0, 0.0],
            }

        class_idx = pred["class_index"]
        rip_class = pred["ripeness_class"]
        conf_pred = pred["confidence"]
        scores = pred["all_scores"]          # [Over Ripe, Ripe, Unripe]
        logits = pred["raw_logits"]

        print(f"  Predicted class:  {rip_class} (index {class_idx})")
        print(f"  Classification confidence: {conf_pred:.3f}")
        print(f"  Probabilities:")
        print(f"    Over Ripe (0): {scores[0]:.3f}")
        print(f"    Ripe (1):      {scores[1]:.3f}")
        print(f"    Unripe (2):    {scores[2]:.3f}")
        print(f"  Raw logits:")
        print(f"    [0] Over Ripe: {logits[0]:.3f}")
        print(f"    [1] Ripe:      {logits[1]:.3f}")
        print(f"    [2] Unripe:    {logits[2]:.3f}")

        # Optionally overlay classifier result on annotated image
        class_label = f"{rip_class[:4]} {conf_pred:.2f}"
        cv2.putText(annotated, class_label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save annotated image
    annotated_path = outputs_dir / "detection_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    print("\n" + "=" * 70)
    print("ANNOTATED IMAGE SAVED")
    print(f"  {annotated_path}")
    print("    Boxes: green, label = ID YOLOconf")
    print("    Below each box: ripeness prediction (red)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total detections processed: {len(detections)}")
    print(f"Crops saved to: {crops_dir}")

    print("\nNext steps:")
    print("1. Examine crops in outputs/detection_crops/ to see if green bananas are being cropped.")
    print("2. Check if any bananas are missing (fewer detections than expected).")
    print("3. Verify classifier probabilities on each crop to see if green crops are misclassified as ripe.")
    print("4. Annotated image shows all YOLO boxes with IDs and confidences.")

    print("\nNOTE: This is a diagnostic script. No changes were made to models, thresholds, or code.")

    return 0

if __name__ == "__main__":
    sys.exit(main())