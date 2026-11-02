"""
test_esp32_client.py
---------------------
Tests for hardware.esp32_client.ESP32Client against the real local
simulator (hardware.simulator): image fetch, telemetry CSV parse
(extended 13-column firmware CSV + legacy 9-column fallback), verdict
POST, and unreachable-host behavior.

Run:
    python main/tests/test_esp32_client.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

import hardware.esp32_client as ec  # noqa: E402
from hardware.simulator import ESP32Simulator  # noqa: E402


def test_fetch_image_returns_bgr_from_simulator():
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        client = ec.ESP32Client(host, port=port)
        img = client.fetch_image()
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
    finally:
        sim.stop()


def test_fetch_image_none_when_no_image_on_device():
    sim = ESP32Simulator(port=0, image_bytes=None)  # fresh ESP32, nothing captured yet
    sim.start()
    try:
        host, port = sim.address
        img = ec.ESP32Client(host, port=port).fetch_image()
        assert img is None
    finally:
        sim.stop()


def test_fetch_telemetry_extended_csv():
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        t = ec.ESP32Client(host, port=port).fetch_telemetry()
        assert t["temperature_c"] == 25.0
        assert t["humidity_pct"] == 70.0
        assert t["eth_status"] == "LOW"
        assert t["ripeness_scale"] == 0.0
        # extended gas columns present in the simulator CSV
        assert t["ethylene_ppm"] is not None
        assert t["voc_ppm"] is not None
    finally:        sim.stop()


def test_fetch_telemetry_legacy_9col_csv():
    """Older firmware without the raw-ppm columns: ppm fields are None,
    flags still parse."""
    sim = ESP32Simulator(port=0, csv_text=
        "Temperature(C),Humidity(%),EthyleneStatus,VOCStatus,AlcoholStatus,AldehydesStatus,RipenessScale,TimeToRipen,StorageCondition\n"
        "24.5,65.0,LOW,LOW,LOW,LOW,-0.20,3 to 5 Days,Store at room temp.\n")
    sim.start()
    try:
        host, port = sim.address
        t = ec.ESP32Client(host, port=port).fetch_telemetry()
        assert t["temperature_c"] == 24.5
        assert t["time_to_ripen"] == "3 to 5 Days"
        assert t["ethylene_ppm"] is None
    finally:
        sim.stop()


def test_post_verdict_delivers_lines():
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        ok = ec.ESP32Client(host, port=port).post_verdict(("ACCEPTABLE", "Ripe - eat now"))
        assert ok is True
        assert sim.last_verdict == ("ACCEPTABLE", "Ripe - eat now")
    finally:
        sim.stop()


def test_post_verdict_rejects_overlong_line():
    """The client refuses to send lines the firmware/LCD can't hold —
    pre-truncation is the caller's job, and the client enforces it."""
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        ok = ec.ESP32Client(host, port=port).post_verdict(("X" * 40, "ok"))
        assert ok is False
        assert sim.last_verdict is None
    finally:
        sim.stop()


def test_unreachable_host_returns_none_not_raises():
    client = ec.ESP32Client("127.0.0.1", port=1, timeout=0.5)  # nothing listens on :1
    assert client.fetch_image() is None
    assert client.fetch_telemetry() is None
    assert client.post_verdict(("A", "B")) is False


def test_telemetry_to_gas_reading_extended():
    """Maps the ESP32 telemetry dict to the GasReader shape the
    FusionEngine consumes: ethylene_ppm + status (+ raw_adc + timestamp)."""
    sim = ESP32Simulator(port=0)
    sim.start()
    try:
        host, port = sim.address
        t = ec.ESP32Client(host, port=port).fetch_telemetry()
        gas = ec.ESP32Client(host, port=port).telemetry_to_gas_reading(t)
        assert "ethylene_ppm" in gas and "status" in gas
        assert gas["ethylene_ppm"] is not None
        assert gas["status"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    finally:
        sim.stop()


def test_telemetry_to_gas_reading_legacy_flags_only():
    """Legacy CSV (no raw ppm): the ethylene flag drives the status —
    HIGH flag → HIGH, LOW flag → LOW — and ppm falls back to 0.0."""
    sim = ESP32Simulator(port=0, csv_text=
        "Temperature(C),Humidity(%),EthyleneStatus,VOCStatus,AlcoholStatus,AldehydesStatus,RipenessScale,TimeToRipen,StorageCondition\n"
        "25.0,70.0,HIGH,LOW,LOW,LOW,0.80,Overripe (Consume Immediately),Keep cool.\n")
    sim.start()
    try:
        host, port = sim.address
        client = ec.ESP32Client(host, port=port)
        gas = client.telemetry_to_gas_reading(client.fetch_telemetry())
        assert gas["status"] == "HIGH"
        assert gas["ethylene_ppm"] == 0.0
    finally:
        sim.stop()


def test_parse_address_defaults_to_port_80():
    assert ec.parse_address("192.168.1.100") == ("192.168.1.100", 80)
    assert ec.parse_address("127.0.0.1:8600") == ("127.0.0.1", 8600)
    assert ec.parse_address("  10.0.0.5 : 9000 ") == ("10.0.0.5", 9000)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
