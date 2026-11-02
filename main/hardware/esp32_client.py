"""
esp32_client.py
---------------
Minimal client for the ESP32 Fruit Monitor web server (``Hardware/``).

Stdlib only (``urllib``) — no new dependencies to the app. Targets the
endpoints the firmware already exposes, plus the ``/verdict`` POST added
by this integration (see ``docs/esp32-hardware-integration.md``):

- ``GET  /download_jpg`` → last camera frame (BGR ndarray) or None
- ``GET  /download_csv`` → telemetry dict (9-col legacy + 13-col extended)
- ``POST /verdict``      → send the two ≤16-char LCD lines back to the box

All methods fail soft: an unreachable ESP32 returns ``None``/``False``
rather than raising, so the app can show a friendly "offline" state.

The telemetry→gas bridge (:meth:`ESP32Client.telemetry_to_gas_reading`)
maps the ESP32's row into the exact dict shape :class:`GasReader` returns,
so the existing :class:`FusionEngine` consumes it unchanged.
"""

import csv
import io
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The ESP32 firmware classifies ethylene on the SAME scale as GasReader
# (main/sensor/gas_reader.py) — keep the constants in lock-step.
ETHYLENE_LOW = 0.5
ETHYLENE_HIGH = 2.0
ETHYLENE_CRITICAL = 5.0

# Legacy firmware CSV (9 columns) and extended firmware CSV (13 columns,
# + raw gas ppm) from Hardware/ESP32_Fruit_Monitor.ino.
_CSV_HEADER = {
    "Temperature(C)": "temperature_c",
    "Humidity(%)": "humidity_pct",
    "EthyleneStatus": "eth_status",
    "VOCStatus": "voc_status",
    "AlcoholStatus": "alcohol_status",
    "AldehydesStatus": "aldehydes_status",
    "RipenessScale": "ripeness_scale",
    "TimeToRipen": "time_to_ripen",
    "StorageCondition": "storage_condition",
    "EthylenePPM": "ethylene_ppm",
    "VOCPPM": "voc_ppm",
    "AlcoholEthanolPPM": "alcohol_ethanol_ppm",
    "AldehydesPPM": "aldehydes_ppm",
}


def parse_address(address: str) -> Tuple[str, int]:
    """Split a user-entered ``host`` or ``host:port`` into ``(host, port)``.

    Bare host → port 80 (the firmware's WebServer default).
    """
    address = (address or "").strip()
    if ":" in address:
        host, port = address.rsplit(":", 1)
        return host.strip(), int(port)
    return address, 80


def _classify_ethylene(ppm: float) -> str:
    if ppm < ETHYLENE_LOW:
        return "LOW"
    if ppm < ETHYLENE_HIGH:
        return "MODERATE"
    if ppm < ETHYLENE_CRITICAL:
        return "HIGH"
    return "CRITICAL"


class ESP32Client:
    """HTTP client for one ESP32 Fruit Monitor box."""

    def __init__(self, host: str = "192.168.1.100", port: int = 80, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base = f"http://{host}:{port}"

    # ── primitives ────────────────────────────────────────────────────────
    def _get(self, path: str) -> Optional[bytes]:
        try:
            with urllib.request.urlopen(self.base + path, timeout=self.timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 — fail soft, app shows "offline"
            logger.warning("ESP32 GET %s failed: %s", path, e)
            return None

    # ── endpoints ─────────────────────────────────────────────────────────
    def fetch_image(self):
        """Last camera frame as a BGR ndarray, or None if unreachable/empty."""
        raw = self._get("/download_jpg")
        if raw is None or len(raw) < 2:
            return None
        import cv2  # local import keeps the module import fast
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def fetch_telemetry(self) -> Optional[dict]:
        """Current sensor row as a dict (None if the ESP32 is unreachable).

        Keys are named after the CSV header (see ``_CSV_HEADER``); absent
        columns (legacy 9-col firmware) come back as ``None`` — callers
        must tolerate that.
        """
        raw = self._get("/download_csv")
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        lines = [l for l in text.splitlines() if l]
        if len(lines) < 2:
            return None
        reader = csv.reader(io.StringIO("\n".join(lines)))
        header = next(reader, None)
        if not header:
            return None
        row = next(reader, None)
        if not row:
            return None
        mapping = dict(zip(header, row))
        out = {}
        for col, key in _CSV_HEADER.items():
            value = mapping.get(col)
            if value is None:
                out[key] = None
            elif key.endswith(("_c", "_pct", "_scale", "_ppm")) or key in ("temperature_c", "humidity_pct", "ripeness_scale"):
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = None
            else:
                out[key] = value
        return out

    def _post_form(self, path: str, fields: dict) -> bool:
        """POST form-encoded fields (the WebServer.h arg-parsing contract).

        The ESP32 sketch uses the Arduino WebServer library, whose
        ``hasArg()``/``arg()`` decode ``application/x-www-form-urlencoded``
        bodies (or the raw body under a ``"plain"`` arg) — NOT JSON. So the
        verdict is sent as form fields, keeping the firmware dependency-free.
        """
        from urllib.parse import urlencode
        data = urlencode(fields).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            logger.warning("ESP32 POST %s rejected: %s", path, e)
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning("ESP32 POST %s failed: %s", path, e)
            return False

    def post_verdict(self, lines: Tuple[str, str]) -> bool:
        """Send the two LCD lines to the ESP32 (form-encoded POST /verdict).

        Returns False (and does NOT send) when a line exceeds 16 chars —
        the firmware/LCD contract is fixed at 16×2, so pre-truncation is
        the caller's job and the client enforces it.
        """
        l1, l2 = (str(x) for x in lines)
        if len(l1) > 16 or len(l2) > 16:
            logger.warning("post_verdict refused: line longer than 16 chars")
            return False
        return self._post_form("/verdict", {"line1": l1, "line2": l2})

    def status(self) -> str:
        """Quick reachability probe: 'ok' or 'offline'."""
        return "ok" if self._get("/download_csv") is not None else "offline"

    # ── bridge into the app's existing sensor shape ───────────────────────
    def telemetry_to_gas_reading(self, telemetry: Optional[dict]) -> dict:
        """Map an ESP32 telemetry row into the :class:`GasReader` dict shape
        (``ethylene_ppm`` / ``raw_adc`` / ``status`` / ``timestamp``) so
        :class:`FusionEngine.evaluate` consumes it unchanged.

        - Extended firmware (raw ppm available): use real ppm.
        - Legacy firmware (flags only): the ethylene HIGH/LOW flag drives
          the status; ppm falls back to 0.0 (LOW).
        """
        now = datetime.now(timezone.utc).isoformat()
        if telemetry is None:
            return {"ethylene_ppm": 0.0, "raw_adc": 0, "status": "LOW",
                    "timestamp": now, "source": "esp32-none"}
        ppm = telemetry.get("ethylene_ppm")
        flag = (telemetry.get("eth_status") or "").upper()
        if ppm is not None:
            status = _classify_ethylene(float(ppm))
            # Calibration drift guard: a HIGH raw flag with a LOW ppm must not
            # be downgraded — the gas path saw something the raw column didn't.
            if flag == "HIGH" and status in ("LOW", "MODERATE"):
                status = "HIGH"
        elif flag == "HIGH":
            # Legacy firmware: flags only, no raw ppm. Represent as HIGH.
            status = "HIGH"
        else:
            status = "LOW"
        return {
            "ethylene_ppm": ppm if ppm is not None else 0.0,
            "raw_adc": 0,
            "status": status,
            "timestamp": now,
            "source": "esp32",
            "temperature_c": telemetry.get("temperature_c"),
            "humidity_pct": telemetry.get("humidity_pct"),
            "time_to_ripen": telemetry.get("time_to_ripen"),
            "storage_condition": telemetry.get("storage_condition"),
        }


def _is_ip_like(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", s or ""))


__all__ = ["ESP32Client", "parse_address"]
