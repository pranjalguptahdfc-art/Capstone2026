import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "main"))
import cv2, torch
from detection.banana_detector import BananaDetector

img = cv2.imread(str(ROOT / "images.jfif"))
det = BananaDetector(device="cpu")
dets = det.detect(img)
canvas = img.copy()
for d in sorted(dets, key=lambda x: -x["confidence"]):
    x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
    is_far = (d["bbox"][1] + d["bbox"][3]) / 2 < img.shape[0] * 0.5
    color = (0, 200, 0) if not is_far else (0, 0, 255)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    label = f"#{d['banana_id']} {d['confidence']:.2f}{' FAR' if is_far else ''}"
    cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
out = ROOT / "main" / "tests" / "verify_ui" / "images_jfif_farband_rescue.png"
cv2.imwrite(str(out), canvas)
print(f"{len(dets)} boxes; green=kept existing, red=new far-band box(es)")
for d in sorted(dets, key=lambda x: -x["confidence"]):
    far = "FAR" if (d["bbox"][1]+d["bbox"][3])/2 < img.shape[0]*0.5 else "   "
    print(f"  #{d['banana_id']:>2} conf={d['confidence']:.3f} bbox={d['bbox']} {far}")
