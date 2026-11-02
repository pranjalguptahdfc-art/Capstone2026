"""
simulator.py
------------
A local, no-hardware stand-in for the ESP32 web server.

It serves the same three endpoints the firmware exposes (see
``Hardware/ESP32_Fruit_Monitor.ino``):

- ``GET  /download_jpg`` — a JPEG (a real repo test image by default, or
  whatever ``image_bytes`` you pass) standing in for the last camera frame.
- ``GET  /download_csv`` — the telemetry CSV. Extended 13-column form when
  the firmware has the raw gas-ppm columns; 9-column form otherwise.
- ``POST /verdict``      — captures ``{"line1","line2"}`` in
  ``self.last_verdict`` (rejecting lines over 16 chars, matching the
  firmware's contract).

Run standalone for a no-hardware demo:

    python main/hardware/simulator.py            # serves on 127.0.0.1:8600
    python main/hardware/simulator.py --port 9000
    python main/hardware/simulator.py --image "images.jfif"

Then point the app's sidebar Hardware address at ``127.0.0.1:8600`` and the
full fetch→analyze→verdict loop runs with zero physical hardware.
"""

import argparse
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_CSV = (
    "Temperature(C),Humidity(%),EthyleneStatus,VOCStatus,AlcoholStatus,AldehydesStatus,"
    "RipenessScale,TimeToRipen,StorageCondition,EthylenePPM,VOCPPM,AlcoholEthanolPPM,AldehydesPPM\n"
    "25.0,70.0,LOW,LOW,LOW,LOW,0.00,3 to 5 Days,Store at room temp.,3.1,21.0,12.0,4.0\n"
)

# Sentinel so "unset" (serve a default image) is distinguishable from an
# explicit ``image_bytes=None`` (serve 404 — "no image captured yet").
_IMAGE_DEFAULT = object()


class ESP32Simulator:
    """In-process stand-in for the ESP32 web server.

    ``port=0`` lets the OS assign a free port (read it back via
    ``.address``) — the tests use this to run in parallel.
    """

    def __init__(self, port=0, host="127.0.0.1", image_bytes=_IMAGE_DEFAULT, csv_text=DEFAULT_CSV):
        self._requested_port = port
        self._host = host
        self._csv_text = csv_text
        # None means "no image captured yet" (mirrors the firmware's 404);
        # the sentinel default serves a real repo test image.
        self._image_bytes = self._default_image() if image_bytes is _IMAGE_DEFAULT else image_bytes
        self.last_verdict = None
        self._server = None
        self._thread = None
        self._bound_host = host
        self._bound_port = None

    @property
    def address(self):
        """(host, actual_port) — the port is resolved after bind and cached,
        so it stays readable after ``stop()`` (tests use it to confirm the
        port is closed)."""
        return self._bound_host, self._bound_port

    @staticmethod
    def _default_image() -> bytes:
        """A bundled reference test image when present, else a tiny valid JPEG."""
        fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
        candidate = fixtures / "images.jfif"
        if candidate.exists():
            return candidate.read_bytes()
        # Minimal 1×1 JPEG so the endpoint is still decodable in isolation.
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07"
            b"\xff\xd9"
        )

    def start(self):
        sim = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _send(self, code, body, ctype):
                if isinstance(body, str):
                    body = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/") == "/download_jpg":
                    if sim._image_bytes is None:
                        # mirrors the firmware: "No image captured yet"
                        self._send(404, "No image captured yet", "text/plain")
                    else:
                        self._send(200, sim._image_bytes, "image/jpeg")
                elif self.path.rstrip("/") == "/download_csv":
                    self._send(200, sim._csv_text, "text/csv")
                else:
                    self._send(404, "not found", "text/plain")

            def do_POST(self):
                if self.path.rstrip("/") != "/verdict":
                    self._send(404, "not found", "application/json")
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                # WebServer.h contract: form-encoded fields (line1, line2).
                fields = urllib.parse.parse_qs(raw.decode("utf-8", errors="replace"))
                l1 = (fields.get("line1", [""])[0] or "").strip()
                l2 = (fields.get("line2", [""])[0] or "").strip()
                if len(l1) > 16 or len(l2) > 16:
                    # Firmware contract: reject overlong lines (the Python side
                    # pre-truncates). 400, nothing captured.
                    self._send(400, "line too long", "text/plain")
                    return
                sim.last_verdict = (l1, l2)
                self._send(200, "ok", "text/plain")

        self._server = ThreadingHTTPServer((self._host, self._requested_port), _Handler)
        self._bound_host, self._bound_port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def _main(argv=None):
    parser = argparse.ArgumentParser(description="ESP32 web-server simulator")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--image", default=None, help="path to a JPEG to serve as the camera frame")
    args = parser.parse_args(argv)

    image_bytes = None
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"image not found: {p}", file=sys.stderr)
            sys.exit(1)
        image_bytes = p.read_bytes()

    # No --image → pass nothing so the default repo image is served (a working
    # demo). Explicit --image → serve exactly that file.
    sim = ESP32Simulator(port=args.port, host=args.host) if image_bytes is None \
        else ESP32Simulator(port=args.port, host=args.host, image_bytes=image_bytes)
    sim.start()
    host, port = sim.address
    print(f"ESP32 simulator serving on http://{host}:{port}")
    print(f"  GET  /download_jpg   GET /download_csv   POST /verdict")
    print("Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    _main()
