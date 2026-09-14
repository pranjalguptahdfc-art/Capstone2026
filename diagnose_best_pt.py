import os
import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = r"D:\Pranjal Files\Capstone2026\Banana-Ripeness-Bunch"
MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "best.pt")

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "physical_banana_test"
)

CONFIDENCE = 0.40

# Detections with IoU above this value are considered
# likely to be duplicate detections of the same banana.
MERGE_IOU = 0.50


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.

    box format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    if intersection_area == 0:
        return 0.0

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# MERGE DUPLICATE DETECTIONS
# ============================================================

def merge_duplicate_detections(detections):
    """
    Groups detections that strongly overlap.

    The highest-confidence detection becomes the representative
    of the physical banana.
    """

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    groups = []

    for detection in detections:

        placed = False

        for group in groups:

            # Compare with the strongest detection in the group
            representative = group[0]

            iou = calculate_iou(
                detection["box"],
                representative["box"]
            )

            if iou >= MERGE_IOU:

                group.append(detection)
                placed = True
                break

        if not placed:
            groups.append([detection])

    # Select strongest detection from each group
    physical_bananas = []

    for group in groups:

        strongest = max(
            group,
            key=lambda x: x["confidence"]
        )

        strongest = strongest.copy()

        strongest["duplicates"] = len(group)

        strongest["group_detections"] = group

        physical_bananas.append(strongest)

    return physical_bananas


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("PHYSICAL BANANA DETECTION TESTER")
print("=" * 70)

print()
print("Choose an image from File Explorer...")
print()

# File picker
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Choose banana image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not image_path:

    print("No image selected.")
    raise SystemExit


print("Selected image:")
print(image_path)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading best.pt...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()

print("Classes:")
print(model.names)
print()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:

    print("ERROR: Could not read image.")
    raise SystemExit

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")
print()

print(
    f"Running detection "
    f"(confidence threshold = {CONFIDENCE})..."
)

print()


# ============================================================
# RUN YOLO
# ============================================================

results = model.predict(
    source=image_path,
    conf=CONFIDENCE,
    verbose=False
)

result = results[0]


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

detections = []

if result.boxes is not None:

    for box in result.boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(int, xyxy)

        confidence = float(
            box.conf[0].cpu().numpy()
        )

        class_id = int(
            box.cls[0].cpu().numpy()
        )

        class_name = model.names[class_id]

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })


# ============================================================
# RAW DETECTIONS
# ============================================================

print("=" * 70)
print("RAW YOLO DETECTIONS")
print("=" * 70)

if not detections:

    print("No bananas detected.")

else:

    for i, detection in enumerate(detections, start=1):

        print(
            f"Detection {i}: "
            f"{detection['class_name']} "
            f"({detection['confidence'] * 100:.2f}%) "
            f"Box={tuple(detection['box'])}"
        )

print()


# ============================================================
# MERGE DUPLICATES
# ============================================================

physical_bananas = merge_duplicate_detections(
    detections
)


# Sort from left to right
physical_bananas.sort(
    key=lambda x: x["box"][0]
)


# ============================================================
# PHYSICAL BANANA RESULTS
# ============================================================

print("=" * 70)
print("PHYSICAL BANANA CANDIDATES")
print("=" * 70)

print()
print(
    f"Raw detections: {len(detections)}"
)

print(
    f"After duplicate merging: "
    f"{len(physical_bananas)}"
)

print(
    f"Merge IoU threshold: {MERGE_IOU}"
)

print()


if not physical_bananas:

    print("No physical bananas detected.")

else:

    for i, banana in enumerate(
        physical_bananas,
        start=1
    ):

        print(
            f"Physical Banana {i}: "
            f"{banana['class_name']} "
            f"({banana['confidence'] * 100:.2f}%)"
        )

        print(
            f"  Box: {tuple(banana['box'])}"
        )

        print(
            f"  Overlapping detections merged: "
            f"{banana['duplicates']}"
        )

        print()


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = {}

for banana in physical_bananas:

    class_name = banana["class_name"]

    class_counts[class_name] = (
        class_counts.get(class_name, 0) + 1
    )


print("=" * 70)
print("PHYSICAL BANANA CLASS COUNTS")
print("=" * 70)

if class_counts:

    for class_name, count in class_counts.items():

        print(
            f"{class_name}: {count}"
        )

else:

    print("None")

print()

print(
    f"FINAL PHYSICAL BANANA COUNT: "
    f"{len(physical_bananas)}"
)

print()


# ============================================================
# DRAW RESULTS
# ============================================================

annotated = image.copy()

for i, banana in enumerate(
    physical_bananas,
    start=1
):

    x1, y1, x2, y2 = banana["box"]

    label = (
        f"Banana {i}: "
        f"{banana['class_name']} "
        f"{banana['confidence'] * 100:.0f}%"
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        5
    )

    cv2.putText(
        annotated,
        label,
        (x1, max(40, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

base_name = os.path.splitext(
    os.path.basename(image_path)
)[0]

output_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_physical_bananas.jpg"
)

cv2.imwrite(
    output_path,
    annotated
)

print("=" * 70)
print("RESULT SAVED")
print("=" * 70)

print(output_path)
print()


# ============================================================
# DISPLAY
# ============================================================

print("Opening physical banana result...")
print()
print("Press any key inside the image window to close it.")

cv2.imshow(
    "Physical Banana Candidates",
    annotated
)

cv2.waitKey(0)
cv2.destroyAllWindows()


print()
print("Test complete.")
print("No model files were modified.")
print("No application files were modified.")