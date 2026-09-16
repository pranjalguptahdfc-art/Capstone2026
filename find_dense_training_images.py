from pathlib import Path
from collections import defaultdict
import math

DATASET = Path("detector_training_data/train")

IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

results = []


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / union


for label_file in LABEL_DIR.glob("*.txt"):

    boxes = []

    try:
        lines = label_file.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                _, cx, cy, w, h = map(
                    float,
                    parts[:5]
                )

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                boxes.append(
                    (x1, y1, x2, y2)
                )

            except ValueError:
                continue

    except Exception:
        continue

    count = len(boxes)

    if count < 5:
        continue

    # --------------------------------------------------------
    # Calculate average overlap
    # --------------------------------------------------------

    overlaps = []

    for i in range(count):

        for j in range(i + 1, count):

            iou = box_iou(
                boxes[i],
                boxes[j]
            )

            if iou > 0:
                overlaps.append(iou)

    if overlaps:
        avg_overlap = sum(overlaps) / len(overlaps)
    else:
        avg_overlap = 0

    # --------------------------------------------------------
    # Calculate average box area
    # --------------------------------------------------------

    areas = []

    for x1, y1, x2, y2 in boxes:

        areas.append(
            max(0, x2 - x1) *
            max(0, y2 - y1)
        )

    avg_area = (
        sum(areas) / len(areas)
        if areas else 0
    )

    # --------------------------------------------------------
    # Dense score
    # --------------------------------------------------------

    # Object count is the strongest signal.
    #
    # Overlap is a secondary signal.
    #
    # This intentionally favors images containing
    # many individual annotated bananas.

    dense_score = (
        count * 1.0
        +
        avg_overlap * 20
    )

    image_name = label_file.stem

    results.append(
        (
            dense_score,
            count,
            avg_overlap,
            avg_area,
            image_name
        )
    )


# Highest score first
results.sort(
    reverse=True
)


print("=" * 90)
print("DENSE BANANA IMAGE RANKING")
print("=" * 90)

print(
    f"{'Rank':<6}"
    f"{'Bananas':<10}"
    f"{'Overlap':<12}"
    f"{'Score':<12}"
    f"Image"
)

print("-" * 90)


for rank, item in enumerate(
    results[:100],
    start=1
):

    score, count, overlap, area, name = item

    print(
        f"{rank:<6}"
        f"{count:<10}"
        f"{overlap:<12.3f}"
        f"{score:<12.2f}"
        f"{name}"
    )


# ------------------------------------------------------------
# Save names
# ------------------------------------------------------------

output = Path(
    "outputs/dense_training_candidates.txt"
)

output.parent.mkdir(
    exist_ok=True
)

with output.open(
    "w",
    encoding="utf-8"
) as f:

    for item in results[:100]:

        score, count, overlap, area, name = item

        f.write(
            f"{name} | "
            f"bananas={count} | "
            f"overlap={overlap:.3f} | "
            f"score={score:.2f}\n"
        )


print()
print("=" * 90)
print("Saved top 100 to:")
print(output)
print("=" * 90)