# ESP32 Hardware Integration — Plan (2026-11-02)

Connects the `Hardware/` firmware (ESP32 + DHT-11 + MQ-135 + I2C LCD + web
server) to the frozen app pipeline. Decisions confirmed with the user
(poll, 3s LCD flash, include simulator).

## Loop

```
ESP32-CAM ─15s─▶ ESP32 buffers last JPEG + reads DHT11/MQ135 (web server :80)
                        │  GET /download_jpg + /download_csv
                        ▼
        app.py "Fetch from Hardware" → same detect→mask→color→fusion pipeline
                        │  POST /verdict {"line1","line2"} (≤16 chars/line)
                        ▼
        LCD flashes verdict 3s, reverts to T/H/ripeness
```

## Deliverables

1. **`main/hardware/lcd_line.py`** — `build_lcd_lines(bunch, telemetry=None) -> (line1, line2)`.
   Pure function, deterministic; every line ≤16 ASCII chars.
   - line1 = verdict: ACCEPTABLE / SEMI-RIPE / HIGH RISK / NO BANANAS
   - line2 = action: overripe → "Overripe - discard"; HIGH RISK → "High risk - act now";
     semi-ripe → "Ripe in ~1 day"; ripe → "Ripe - eat now"; unripe → telemetry
     time_to_ripen ("3 to 5 Days" → "Ripe in 3-5 days") or "Ripe in ~3 days".
2. **`main/hardware/esp32_client.py`** — `ESP32Client(host, port=80, timeout=5)`
   stdlib urllib only (no new dependencies):
   - `fetch_image() -> np.ndarray BGR | None` (GET /download_jpg, cv2.imdecode)
   - `fetch_telemetry() -> dict` (GET /download_csv; parses the firmware's 9-col
     CSV and the extended 13-col CSV — backward compatible)
   - `post_verdict(lines: tuple[str, str]) -> bool` (POST /verdict, JSON)
   - `telemetry_to_gas_reading(t) -> dict` — maps to the GasReader shape
     (`ethylene_ppm/raw_adc/status`) so FusionEngine consumes it unchanged
   - `status()` — quick reachability probe
3. **`main/hardware/simulator.py`** — stdlib http.server on 127.0.0.1:8600
   (overridable) emulating /download_jpg (serves a fixture or `--image`),
   /download_csv (extended 13-col sample telemetry), /verdict (captures posts).
   Runnable: `python main/hardware/simulator.py`.
4. **`main/app.py`** — sidebar "Hardware (ESP32)" block: checkbox + address
   input (default `192.168.1.100`, caption points to Serial Monitor + simulator
   tip `127.0.0.1:8600`). "📷 Fetch from Hardware" button next to the uploader:
   fetch image+telemetry → same pipeline as uploads → telemetry card +
   "Verdict sent to LCD ✓ / (offline)" confirmation. Per-banana breakdown
   gains a CSV download button (handoff-plan step 1).
5. **Firmware (`Hardware/ESP32_Fruit_Monitor.ino`)** —
   - `server.on("/verdict", ...)` POST JSON `{"line1","line2"}` → store +
     `lastVerdictTime`; loop() displays verdict lines for 3s then reverts.
   - `/download_csv` gains 4 raw gas-ppm columns (ethylene, VOC, alcohol,
     aldehydes) for the in-repo calibration step.
   - Boot prints the assigned WiFi IP to Serial.
   - `FruitAnalysis-0.cpp` is a duplicate of `FruitAnalysis.cpp` — keep one
     (flagged, not deleted; two identical `analyzeFruitData` in one sketch
     folder = compile error).
6. **Tests (TDD — written before implementation)** —
   - `main/tests/test_lcd_line.py` (~8): verdict mapping, telemetry time mapping,
     16-char truncation, ASCII-only, all four statuses
   - `main/tests/test_esp32_simulator.py` (~4): server start/stop, jpg decodes,
     csv columns, verdict capture
   - `main/tests/test_esp32_client.py` (~6): client against the live simulator —
     fetch_image/fetch_telemetry (9-col + 13-col)/post_verdict/unreachable host
   - `main/tests/test_app_hardware.py` (~2): AppTest — sidebar block renders,
     fetch button with a stubbed client shows the success + LCD-sent path
   - `main/tests/verify_hardware_loop.py`: end-to-end proof script — simulator
     up → client fetch → real pipeline on a repo test image → build_lcd_lines →
     POST verdict → assert simulator captured it
7. **Docs** — `progress.md` + `main/NON_TEACHABLE_GUIDE.md` "Remaining"
   table updated at the end.

## Out of scope (stays TODO)
- Approach A (white reference patch) — still hardware-gated, different round
- In-repo gas calibration / `analytics_view.py` update (handoff step 2)
- ESP32 WiFi/creds are still placeholders in the sketch — user's to fill
- Push mode (Flask sidecar) — rejected in favor of poll
