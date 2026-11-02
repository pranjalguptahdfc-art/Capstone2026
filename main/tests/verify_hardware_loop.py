"""
verify_hardware_loop.py
----------------------
End-to-end proof of the ESP32 hardware loop with NO physical hardware:

    simulator (stands in for the ESP32 web server)
        → ESP32Client.fetch_image / fetch_telemetry
        → the REAL pipeline (BananaDetector → analyze_batch → FusionEngine)
        → build_lcd_lines
        → POST /verdict
        → assert the simulator captured exactly those two lines

Run:
    python main/tests/verify_hardware_loop.py

Exits 0 on success. Prints the verdict lines and the captured result so a
demo can show the closed loop without the enclosure on the bench.
"""

import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

# torch before pandas (Windows c10.dll ordering — same as conftest.py)
try:
    import torch  # noqa: F401
except Exception:
    pass

from hardware.simulator import ESP32Simulator          # noqa: E402
from hardware.esp32_client import ESP32Client          # noqa: E402
from hardware.lcd_line import build_lcd_lines          # noqa: E402

from detection.banana_detector import BananaDetector   # noqa: E402
from vision.color_analyzer import analyze_batch        # noqa: E402
from fusion.fusion_engine import FusionEngine          # noqa: E402

# A repo test image that reliably detects bananas (close-range bunch).
IMAGE = PROJECT_ROOT.parent / "images (1).jfif"
if not IMAGE.exists():
    IMAGE = PROJECT_ROOT.parent / "images.jfif"


def main():
    # ── 1. Start the ESP32 simulator (no hardware needed) ─────────────────
    sim = ESP32Simulator(port=0, image_bytes=IMAGE.read_bytes())
    sim.start()
    host, port = sim.address
    print(f"ESP32 simulator  : http://{host}:{port}  (stands in for the box)")
    print(f"camera frame     : {IMAGE.name}\n")

    try:
        client = ESP32Client(host, port=port)

        # ── 2. Fetch image + telemetry, exactly like app.py does ──────────
        img = client.fetch_image()
        if img is None:
            print("FAIL: fetch_image returned None")
            return 1
        telemetry = client.fetch_telemetry()
        gas = client.telemetry_to_gas_reading(telemetry)
        print(f"fetched image  : {img.shape[1]}x{img.shape[0]} BGR")
        print(f"telemetry      : ethylene {gas['ethylene_ppm']} ppm -> {gas['status']}\n")

        # ── 3. Run the REAL pipeline (same as the upload path) ────────────
        detector = BananaDetector(confidence_threshold=0.25)
        detections = detector.detect(img, confidence_threshold=0.10, floor=False)
        print(f"detections     : {len(detections)} bananas")
        if not detections:
            print("FAIL: no bananas detected on the reference image")
            return 1

        color_analyses = analyze_batch(img, detections)
        engine = FusionEngine()
        bunch = engine.evaluate(gas, color_analyses)
        print(f"bunch verdict  : {bunch['bunch_status']}  ({bunch['color_signal']})")
        print(f"reasoning      : {bunch['reasoning']}\n")

        # ── 4. Build the LCD lines and POST them to the simulator ────────
        lines = build_lcd_lines(bunch, telemetry)
        sent = client.post_verdict(lines)
        print(f"LCD lines      : {lines[0]!r} / {lines[1]!r}")
        print(f"verdict posted : {sent}")
        print(f"simulator saw  : {sim.last_verdict}\n")

        ok = sent and sim.last_verdict == tuple(lines)
        print("LOOP " + ("COMPLETED" if ok else "FAILED"))
        return 0 if ok else 1
    finally:
        sim.stop()


if __name__ == "__main__":
    sys.exit(main())
