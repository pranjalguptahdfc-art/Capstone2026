#!/usr/bin/env python3
"""
diagnose_failure_case_2.py
--------------------------
Diagnostic script for Failure Case 2: AI-generated mixed-ripeness image
showing 100% ripe when it should show ~50% ripe/50% unripe.

This script runs the EXISTING pipeline on a user-provided image and:
1. Saves each detected banana crop to outputs/diagnostic_crops/
2. Prints detailed detection/classification info for each banana
3. Creates a contact sheet of all crops with predictions
4. Prints final bunch analysis summary

Usage:
    python diagnose_failure_case_2.py <image_path>
    # or if no argument provided, will prompt for image path

IMPORTANT: Uses EXISTING models and code exactly as-is - no modifications.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing modules (UNCHANGED)
from detection.banana_detector import BananaDetector
from classification.ripeness_classifier import RipenessClassifier
from processing.bunch_analyzer import BunchAnalyzer
from utils.image_utils import crop_banana, annotate_image, cv2_to_pil, pil_to_cv2

def create_output_directories():
    """Create necessary output directories."""
    outputs_dir = PROJECT_ROOT / "outputs"
    crops_dir = outputs_dir / "diagnostic_crops"

    outputs_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    return outputs_dir, crops_dir

def get_image_path():
    """Get image path from command line argument or user input."""
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if image_path.exists():
            return image_path
        else:
            print(f"[ERROR] Image file not found: {image_path}")
            sys.exit(1)
    else:
        # Prompt user for image path
        image_path = input("Enter path to the AI-generated mixed-ripeness banana image: ").strip()
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[ERROR] Image file not found: {image_path}")
            sys.exit(1)
        return image_path

def main():
    print("=" * 70)
    print("DIAGNOSTIC SCRIPT FOR FAILURE CASE 2")
    print("AI-generated mixed-ripeness image showing 100% ripe")
    print("=" * 70)

    # Get image path
    image_path = get_image_path()
    print(f"[INFO] Using image: {image_path}")

    # Create output directories
    outputs_dir, crops_dir = create_output_directories()
    print(f"[INFO] Outputs will be saved to: {outputs_dir}")
    print(f"[INFO] Diagnostic crops will be saved to: {crops_dir}")

    # Load existing models (UNCHANGED)
    print("\n[INFO] Loading models...")
    detector = BananaDetector(confidence_threshold=0.40)  # Using default from app.py
    classifier = RipenessClassifier()
    analyzer = BunchAnalyzer()

    if not classifier.is_available():
        print("[ERROR] Ripeness classifier model not found!")
        print("        Please place EfficientnetBo.h5 in models/ directory")
        sys.exit(1)

    # Load the classifier
    try:
        classifier.load()
        print("[SUCCESS] Classifier loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")
        sys.exit(1)

    # Load image
    print(f"\n[INFO] Loading image: {image_path}")
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Image shape: {bgr_image.shape} (height x width x channels)")

    # Run detection
    print("\n[INFO] Running YOLO detection...")
    detections = detector.detect(bgr_image, confidence_threshold=0.40)

    if not detections:
        print("[WARNING] No bananas detected!")
        print("        This might be due to:")
        print("        - YOLO not detecting bananas in this image")
        print("        - Confidence threshold too high")
        print("        - Image quality/issues")
        sys.exit(1)

    print(f"[SUCCESS] Detected {len(detections)} banana(s)")

    # Process each detection
    banana_results = []
    crop_filenames = []

    print("\n" + "=" * 70)
    print("DETECTION & CLASSIFICATION RESULTS")
    print("=" * 70)

    for i, det in enumerate(detections):
        banana_id = det["banana_id"]
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        yolo_confidence = det["confidence"]

        print(f"\nBanana {banana_id:03d}:")
        print(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        print(f"  YOLO Confidence: {yolo_confidence:.3f}")

        # Crop the banana
        try:
            crop = crop_banana(bgr_image, bbox, padding=0.05)

            # Save crop
            crop_filename = f"banana_{banana_id:03d}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            crop_filenames.append((banana_id, crop_path, bbox))

            print(f"  Crop saved to: {crop_path}")
            print(f"  Crop shape: {crop.shape}")

        except Exception as e:
            print(f"  [ERROR] Failed to crop banana {banana_id}: {e}")
            continue

        # Classify the crop
        try:
            pred = classifier.predict(crop)

            class_index = pred["class_index"]
            ripeness_class = pred["ripeness_class"]
            confidence = pred["confidence"]
            all_scores = pred["all_scores"]  # [Over Ripe, Ripe, Unripe] probabilities
            raw_logits = pred["raw_logits"]

            # Store results for bunch analysis
            result_entry = {
                "banana_id": banana_id,
                "bbox": bbox,
                "confidence": yolo_confidence,  # detection confidence
                "ripeness_class": ripeness_class,
                "canonical": pred["canonical"],
                "ripe_confidence": confidence,  # classification confidence
            }
            banana_results.append(result_entry)

            # Print classification results
            print(f"  Predicted Class: {ripeness_class}")
            print(f"  Classification Confidence: {confidence:.3f}")
            print(f"  Probabilities:")
            print(f"    Over Ripe (Index 0): {all_scores[0]:.3f}")
            print(f"    Ripe (Index 1):      {all_scores[1]:.3f}")
            print(f"    Unripe (Index 2):    {all_scores[2]:.3f}")
            print(f"  Raw Logits:")
            print(f"    [0] Over Ripe: {raw_logits[0]:.3f}")
            print(f"    [1] Ripe:      {raw_logits[1]:.3f}")
            print(f"    [2] Unripe:    {raw_logits[2]:.3f}")

        except Exception as e:
            print(f"  [ERROR] Failed to classify banana {banana_id}: {e}")
            # Still add to results with error status for bunch analysis
            result_entry = {
                "banana_id": banana_id,
                "bbox": bbox,
                "confidence": yolo_confidence,
                "ripeness_class": "Error",
                "canonical": None,
                "ripe_confidence": 0.0,
            }
            banana_results.append(result_entry)
            continue

    print("\n" + "=" * 70)
    print("CREATING CONTACT SHEET")
    print("=" * 70)

    # Create contact sheet
    if crop_filenames:
        try:
            # Determine contact sheet dimensions
            thumbs_per_row = 5
            thumb_width, thumb_height = 120, 120

            rows_needed = (len(crop_filenames) + thumbs_per_row - 1) // thumbs_per_row
            contact_width = thumbs_per_row * thumb_width
            contact_height = rows_needed * thumb_height

            # Create blank contact sheet
            contact_sheet = np.full((contact_height, contact_width, 3), 255, dtype=np.uint8)  # White background

            for idx, (banana_id, crop_path, bbox) in enumerate(crop_filenames):
                # Load and resize crop for contact sheet
                crop_img = cv2.imread(str(crop_path))
                if crop_img is None:
                    continue

                # Resize to thumbnail size
                thumb = cv2.resize(crop_img, (thumb_width, thumb_height))

                # Calculate position in contact sheet
                row = idx // thumbs_per_row
                col = idx % thumbs_per_row
                y_start = row * thumb_height
                y_end = y_start + thumb_height
                x_start = col * thumb_width
                x_end = x_start + thumb_width

                # Place thumbnail
                contact_sheet[y_start:y_end, x_start:x_end] = thumb

                # Add label
                label = f"ID:{banana_id:03d}"
                # Find classification result for this banana
                class_label = "Unknown"
                conf_label = "0.000"
                for result in banana_results:
                    if result["banana_id"] == banana_id:
                        class_label = result["ripeness_class"][:4]  # Truncate for display
                        conf_label = f"{result['ripe_confidence']:.3f}"
                        break

                # Draw label background
                label_bg_y1 = y_start + 5
                label_bg_y2 = y_start + 25
                label_bg_x1 = x_start + 5
                label_bg_x2 = x_start + len(label) * 8 + 10
                cv2.rectangle(contact_sheet, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 0, 0), -1)

                # Draw label text
                cv2.putText(contact_sheet, label, (x_start + 10, y_start + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Add confidence label
                conf_label_text = f"C:{conf_label}"
                cv2.putText(contact_sheet, conf_label_text, (x_start + 10, y_start + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save contact sheet
            contact_sheet_path = outputs_dir / "diagnostic_contact_sheet.jpg"
            cv2.imwrite(str(contact_sheet_path), contact_sheet)
            print(f"[SUCCESS] Contact sheet saved to: {contact_sheet_path}")
            print(f"          Dimensions: {contact_width} x {contact_height} pixels")
            print(f"          Shows {len(crop_filenames)} banana crops with IDs and classification confidences")

        except Exception as e:
            print(f"[WARNING] Failed to create contact sheet: {e}")
    else:
        print("[WARNING] No crops to create contact sheet from")

    # Run bunch analysis
    print("\n" + "=" * 70)
    print("BUNCH ANALYSIS RESULTS")
    print("=" * 70)

    if banana_results:
        bunch = analyzer.analyze(banana_results)

        print(f"Total bananas detected: {bunch['total_bananas']}")
        print(f"Ripe count: {bunch['counts']['ripe']} ({bunch['percentages']['ripe']:.1f}%)")
        print(f"Unripe count: {bunch['counts']['unripe']} ({bunch['percentages']['unripe']:.1f}%)")
        print(f"Over Ripe count: {bunch['counts']['overripe']} ({bunch['percentages']['overripe']:.1f}%)")
        print(f"Problematic bananas (Unripe + Over Ripe): {bunch['problematic_count']} ({bunch['problematic_percentage']:.1f}%)")
        print(f"Bunch Status: {bunch['bunch_status']}")
        print(f"Dominant Problem: {bunch['dominant_problem']}")

        # Also show the supporting aggregation methods
        print(f"\nSupporting Information (Original Methods):")
        print(f"  Majority Vote: {bunch['majority_result']}")
        print(f"  Average Ripeness Score: {bunch['avg_ripeness_score']:.3f}")
        print(f"  Score Classification: {bunch['score_result']}")

    else:
        print("[ERROR] No valid classification results to analyze")
        bunch = None

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Summary of files created:")
    print(f"  - Individual crops: {len(crop_filenames)} files in {crops_dir}/")
    print(f"    Named: banana_001.jpg, banana_002.jpg, etc.")
    if 'contact_sheet_path' in locals():
        print(f"  - Contact sheet: {contact_sheet_path}")
    print(f"  - Output directory: {outputs_dir}")
    print("\nNext steps:")
    print("1. Visually inspect the saved crops in outputs/diagnostic_crops/")
    print("2. Check if crops actually contain good banana tissue")
    print("3. Verify if green bananas are being misclassified as ripe")
    print("4. Determine if issue is with detection/cropping or classification")

    return 0

if __name__ == "__main__":
    sys.exit(main())