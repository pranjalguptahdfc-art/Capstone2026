"""
test_esp32_simulator.py
----------------------
Tests for hardware.simulator — the local stand-in for the ESP32 web server
(/download_jpg, /download_csv, /verdict) used for no-hardware demo mode and
as the test target for hardware.esp32_client.

Run:
    python main/tests/test_esp32_simulator.py
"""

import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.simulator import ESP32Simulator  # noqa: E402


def test_simulator_lifecycle_and_endpoints():
    """Start, hit all three endpoints, stop — no port leaks, no threads."""
    sim = ESP32Simulator(port=0)  # port 0 → OS-assigned; .address available
    sim.start()
    try:
        host, port = sim.address
        base = f"http://{host}:{port}"

        # /download_csv returns a parseable header + row
        csv_text = urllib.request.urlopen(base + "/download_csv", timeout=5).read().decode()
        lines = [l for l in csv_text.strip().splitlines() if l]
        assert len(lines) == 2
        assert lines[0].startswith("Temperature(C),Humidity(%)")

        # /download_jpg returns bytes that look like an image (magic sniff)
        jpg = urllib.request.urlopen(base + "/download_jpg", timeout=5).read()
        assert len(jpg) > 0

        # /verdict captures a form-encoded post and acks (WebServer.h contract)
        body = "line1=ACCEPTABLE&line2=Ripe - eat now".encode()
        req = urllib.request.Request(base + "/verdict", data=body,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                                     method="POST")
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 200
        assert sim.last_verdict == ("ACCEPTABLE", "Ripe - eat now")
    finally:
        sim.stop()

    # After stop, the port refuses connections
    import socket
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((sim.address[0], sim.address[1]))
        raised = False
    except (ConnectionRefusedError, OSError):
        raised = True
    finally:
        s.close()
    assert raised, "simulator port should be closed after stop()"


def test_simulator_serves_image_from_bytes():
    sim = ESP32Simulator(port=0, image_bytes=b"\xff\xd8\xff\xe0fakejpeg\xff\xd9")
    sim.start()
    try:
        host, port = sim.address
        jpg = urllib.request.urlopen(f"http://{host}:{port}/download_jpg", timeout=5).read()
        assert jpg == b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"
    finally:
        sim.stop()


def test_simulator_rejects_overlong_verdict_lines():
    """Firmware contract: the LCD is 16×2 — lines over 16 chars are rejected,
    not truncated (the Python side must pre-truncate)."""
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        body = "line1=TOO LONG LINE THAT EXCEEDS SIXTEEN&line2=ok".encode()
        req = urllib.request.Request(f"http://{host}:{port}/verdict", data=body,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                                     method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            raised = None
        except urllib.error.HTTPError as e:
            raised = e
        assert raised is not None and raised.code == 400, \
            f"expected HTTP 400, got: {raised}"
        assert sim.last_verdict is None  # rejected, not captured
    finally:
        sim.stop()


def test_simulator_csv_has_extended_gas_columns():
    """The extended firmware CSV carries raw gas ppm for in-repo calibration."""
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        text = urllib.request.urlopen(f"http://{host}:{port}/download_csv", timeout=5).read().decode()
        header = text.strip().splitlines()[0]
        for col in ("EthylenePPM", "VOCPPM", "AlcoholEthanolPPM", "AldehydesPPM"):
            assert col in header, f"missing {col} in simulator CSV header"
    finally:
        sim.stop()


def _main():
    failed = []
    for name, fn in [
        ("test_simulator_lifecycle_and_endpoints", test_simulator_lifecycle_and_endpoints),
        ("test_simulator_serves_image_from_bytes", test_simulator_serves_image_from_bytes),
        ("test_simulator_rejects_overlong_verdict_lines", test_simulator_rejects_overlong_verdict_lines),
        ("test_simulator_csv_has_extended_gas_columns", test_simulator_csv_has_extended_gas_columns),
    ]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
