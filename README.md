# 🍌 Banana Ripeness & Quality Analytics

A **non-teachable** (no model training) banana-ripeness detector for import/export
fruit retailers. Point a camera at a bunch; the system tells you whether each banana
is unripe / semi-ripe / ripe / overripe, gives a bunch verdict, estimates days-to-ripe
from the environment, and (with the ESP-32 hardware box) flashes the verdict on an on-board LCD.

Everything in this README is the **complete** setup — Python app, test image, hardware
connection, and how to connect the code to the physical ESP-32.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Repo layout](#2-repo-layout)
3. [Prerequisites](#3-prerequisites)
4. [Software setup (Python)](#4-software-setup--python)
5. [Run the app](#5-run-the-app)
6. [Run the test suite](#6-run-the-test-suite)
7. [No-hardware demo mode (simulator)](#7-no-hardware-demo-mode--simulator)
8. [Hardware setup (ESP-32)](#8-hardware-setup--esp-32)
9. [Connect the ESP-32 to the app](#9-connect-the-esp-32-to-the-app)
10. [End-to-end hardware loop](#10-end-to-end-hardware-loop)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Architecture

```
        ┌─────────────────────────────  CAMERA  ────────────────────────────┐
        │  Upload a photo in the app   OR   ESP-32-CAM → ESP-32 (every 15s)│
        └────────────────────────────────────────────────────────────────────┘
                                              │
   ┌──────────────────────────────────────────▼──────────────────────────────────┐
   │                        main/app.py  (Streamlit)                             │
   │                                                                             │
   │   detection/banana_detector.py   pre-trained YOLOv8n (COCO) + adaptive      │
   │        filter + NMS + slice-boost + far-band rescue + containment dedup    │
   │   vision/banana_mask.py            isolate the banana body (ignore bg)      │
   │   vision/color_analyzer.py          per-banana G/Y/B % via color_model      │
   │   vision/color_model.py            lighting-invariant color rules (B)      │
   │   fusion/fusion_engine.py          gas + color → bunch verdict             │
   │   sensor/gas_reader.py             DHT/MQ gas reading (demo + serial)       │
   └──────────────────────────────────────────┬──────────────────────────────────┘
                                              │
        ┌─────────────────────────────────────▼──────────────────────────────────┐
        │  hardware/esp32_client.py   (when wired to the ESP-32)                │
        │     GET /download_jpg + /download_csv  →  run the SAME pipeline        │
        │     POST /verdict (2×16-char lines)   →  ESP-32 I2C LCD flashes 3s    │
        └────────────────────────────────────────────────────────────────────────┘
```

The pipeline is **frozen** (detection scope locked for the demo). The ripeness
verdict is pure color math — no learned classifier. The only "ML" is a
pre-built YOLOv8n used for *localization only*; it is never retrained.

---

## 2. Repo layout

```
BananaRipeness/
├── main/                          ← the Streamlit application
│   ├── app.py                     ← entry point (Streamlit UI)
│   ├── requirements.txt           ← app dependencies (pip install -r main/requirements.txt)
│   ├── detection/banana_detector.py
│   ├── vision/{color_model,color_analyzer,banana_mask}.py
│   ├── fusion/fusion_engine.py
│   ├── sensor/gas_reader.py
│   ├── hardware/                  ← NEW: ESP-32 integration
│   │   ├── esp32_client.py        stdlib HTTP client (fetch img/CSV, post verdict)
│   │   ├── lcd_line.py            build the 2×16-char LCD lines
│   │   └── simulator.py           no-hardware stand-in for the ESP-32 web server
│   ├── utils/…                    crop / annotate / analytics store
│   ├── models/
│   │   ├── yolov8n.pt            pre-trained detector (auto-downloads if missing)
│   │   └── Training/             763-image Roboflow verification set
│   └── tests/                     pytest suites + verification scripts
├── Hardware/                      ← ESP-32 firmware (Arduino)
│   ├── ESP32_Fruit_Monitor.ino    main sketch (WiFi + web server + LCD + DHT + MQ-135)
│   ├── Config.h                   pins + sensor structs
│   └── FruitAnalysis.cpp          on-device ripeness / time-to-ripen logic
├── docs/esp32-hardware-integration.md   ← the design + data contract
├── choose.md / progress.md        ← decision logs
└── choose.md / progress.md        ← decision logs
```

---

## 3. Prerequisites

- **Python 3.10** (tested on 3.10.4).
- **Windows, macOS, or Linux.**
- **~2 GB disk** for `torch` (CPU build).
- For the hardware box: an **ESP-32** with the parts below (§8) and **Arduino IDE**.
- No internet is required *at run time* except the one-time YOLOv8n download
  (`main/models/yolov8n.pt`) if it's missing.

> ⚠️ **Windows quirk (already handled for you):** on Windows the app must load
> **torch before pandas** or the two OpenMP runtimes clash and you get
> `OSError [WinError 1114] … c10.dll`. `main/app.py` and `main/tests/conftest.py`
> both preload torch first, so you never hit this. Don't import `pandas` before
> `torch` in any script you add.

---

## 4. Software setup (Python)

```bash
# 1. clone / open the project
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"

# 2. create + activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. install dependencies  (from the project root)
pip install -r main/requirements.txt
#    → opencv-python, numpy, pillow, streamlit, ultralytics, pandas, pyserial,
#      tensorflow.  `ultralytics` pulls in `torch` (the app preloads it, §3 note).
```

`main/requirements.txt` lists the app's runtime dependencies. If your environment
already has these (as this machine does), you can skip the install — but the
one-time `yolov8n.pt` download still happens on first run.

Verify the install:

```bash
python -c "import torch, cv2, streamlit, ultralytics, pandas; print('deps OK')"
```

> **Optional — browser verification only** (some tests can run the app in a real
> browser). Skip unless you want it:
>
> ```bash
> pip install playwright
> playwright install chromium
> ```

---

## 5. Run the app

From the project root:

```bash
streamlit run main/app.py
```

The browser opens `http://localhost:8501`.

**Use it:**
1. The **Ripeness Detector** page is the default.
2. Either **upload a banana bunch image** (JPG / JPEG / JFIF / PNG), **or** enable
   **Hardware (ESP32)** in the sidebar and press **📷 Fetch from ESP32** (§9).
3. Read the result cards: **DECISION** (Approved / Reject), **STATUS**
   (ACCEPTABLE / SEMI-RIPE / HIGH RISK), **BANANAS DETECTED**, **AVG RIPENESS**,
   plus the per-banana table, color breakdown, and the download buttons.

**Detection mode** (sidebar):
- **Auto** (default) — adaptive: detects all bananas, then removes low-confidence
  duplicates. Best for mixed/occluded shots.
- **Manual** — your confidence slider is a *floor*; the pipeline still adapts first.

**Gas sensor** (sidebar): **Demo** simulates a reading (default); **ESP32 serial**
reads a real serial port. For the connected hardware box, gas comes straight from
the ESP-32's `/download_csv` — you don't need the serial path in the app.

---

## 6. Run the test suite

```bash
python -m pytest main/tests/ -q
```

Expected: **113 passing** (81 pipeline/detection/color + 32 for the ESP-32
hardware round). The hardware tests (`test_esp32_client`, `test_esp32_simulator`,
`test_lcd_line`, `test_app_hardware`) run against the in-process simulator —
**no physical ESP-32 needed**.

Individual / verification scripts (all runnable standalone):

```bash
python main/tests/verify_hardware_loop.py    # full photo→app→LCD loop, no hardware
python main/tests/verify_images.py           # run the live pipeline on an image folder
python main/blind_batch.py                   # 20-image accuracy smoke vs. labels
python main/blind_batch.py --all             # full 154-image gate (~4 min)
```

---

## 7. No-hardware demo mode (simulator)

If the physical box is off the bench, the simulator stands in for it and the
*entire* fetch → analyze → verdict loop still works:

```bash
# terminal 1 — start the simulator (serves on 127.0.0.1:8600)
python main/hardware/simulator.py

# terminal 2 — run the app, then in the sidebar:
#   enable "Enable ESP32 hardware", set address to 127.0.0.1:8600, click Fetch
streamlit run main/app.py
```

The simulator serves a bundled banana image for `/download_jpg`, a sample
telemetry CSV for `/download_csv`, and captures your `POST /verdict`
(rejecting any line longer than 16 chars, exactly like the firmware).

You can point it at your own image: `python main/hardware/simulator.py --image "main/tests/fixtures/images (1).jfif"`.

---

## 8. Hardware setup (ESP-32)

### 8.1 Parts & wiring

| Part | Role | Notes |
|------|------|-------|
| **ESP-32** (DevKit / WROOM) | main controller + WiFi + web server | runs the sketch below |
| **ESP-32-CAM** (OV2640/OV5640) | camera | separate device on the network; IP set in `cam_ip` |
| **DHT-11** | temperature + humidity | pin **15** (data) — see `Config.h` |
| **MQ-135** | gas (VOC / ethylene proxy) | analog pin **34** |
| **I2C LCD** (PCF8574, 16×2) | displays the verdict | I2C addr **0x27**, SDA=**21**, SCL=**22** |
| **Reset button** | reboots the sketch | pin **4** (INPUT_PULLUP) |

> Pin mapping lives in `Hardware/Config.h` — change it there, not in the sketch.

### 8.2 Install the Arduino libraries

In **Arduino IDE → Tools → Manage Libraries**, install:

```
DHT sensor library (Adafruit)
LiquidCrystal_I2C
WebServer        (built-in)
HTTPClient       (built-in)
```

The sketch needs an ESP-32 board package (Tools → Board → *ESP32 Arduino*),
board set to **ESP32WROVER Dev Module** (or your actual ESP-32 model), and
USB port selected.

### 8.3 Flash the firmware

1. Open **`Hardware/ESP32_Fruit_Monitor.ino`** in the Arduino IDE
   (put `Config.h` and `FruitAnalysis.cpp` in the same sketch folder).

   > ⚠️ **Do NOT also include `FruitAnalysis-0.cpp`** — it is a byte-for-byte
   > duplicate of `FruitAnalysis.cpp` and will fail to compile
   > (duplicate `analyzeFruitData`). Keep one.
2. Set your credentials and the ESP-32-CAM's IP in the `.ino`:
   ```c
   const char* ssid     = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* cam_ip   = "192.168.1.50";   // your ESP-32-CAM's IP
   ```
   (The ESP-32 and the ESP-32-CAM must be on the **same network**.)
3. **Upload** to the ESP-32.
4. Open **Serial Monitor @ 115200**. After a brief "System Ready", you'll see:
   ```
   ESP32 Fruit Monitor — point the app at:
   http://192.168.1.73          ← use THIS address in the app (§9)
   ```

The box is now live. It polls the camera every 15 s, reads the DHT-11 + MQ-135
every second, and serves `/`, `/download_csv`, `/download_jpg`, and (new)
`POST /verdict` on port 80.

---

## 9. Connect the ESP-32 to the app

The app talks to the ESP-32 over **HTTP polling** (Streamlit can't receive webhooks,
so the app *pulls* rather than the box pushing). Three requests close the loop:

| # | Request | What it does |
|---|---------|--------------|
| 1 | `GET  /download_jpg`  | fetch the last camera frame |
| 2 | `GET  /download_csv`  | fetch DHT-11 / MQ-135 telemetry (+ raw gas ppm) |
| 3 | `POST /verdict`       | send two ≤16-char lines → LCD flashes 3 s |

**In the app:**
1. Start the app (§5).
2. Sidebar → **Hardware (ESP32)** → tick **Enable ESP32 hardware**.
3. Set **ESP32 address** to the IP the sketch printed (e.g. `192.168.1.73`).
   (No hardware? Use `127.0.0.1:8600` after starting the simulator, §7.)
4. Press **📷 Fetch from ESP32**.

The app runs the *exact same* detect → mask → color → fusion pipeline as an
upload, shows the full result view, and POSTs the verdict back. You'll see a
**"📟 Verdict sent to ESP32 LCD"** confirmation and the LCD light up for 3
seconds, then return to the T/H/ripeness display.

The per-banana table also has a **⬇️ CSV download** button (the handoff step for
calibrating the gas sensor in-repo).

### Verdict → LCD mapping (`hardware/lcd_line.py`)

| Pipeline result | LCD line 1 | LCD line 2 |
|-----------------|-----------|-----------|
| HIGH RISK, overripe | `HIGH RISK` | `Discard now` |
| HIGH RISK, other | `HIGH RISK` | `High risk, act!` |
| SEMI-RIPE | `SEMI-RIPE` | `Ripe in ~1 day` |
| ACCEPTABLE, ripe | `ACCEPTABLE` | `Ripe - eat now` |
| ACCEPTABLE, unripe | `ACCEPTABLE` | `Ripe in ~3 days` (or the ESP-32's own time estimate) |
| no bananas | `NO BANANAS` | `No bananas seen` |

---

## 10. End-to-end hardware loop (prove it works)

```bash
python main/tests/verify_hardware_loop.py
```

This spins up the simulator, fetches image + telemetry through the **real**
client, runs the **real** YOLO + color + fusion pipeline, builds the LCD lines,
POSTs the verdict, and asserts the simulator captured exactly those lines.
Exits `0` = the full loop is wired correctly — **no physical hardware required**.

To test against the *real* box instead, point `main/hardware/simulator.py`
at it by setting the app's address to the box IP (§9), or call
`hardware.esp32_client.ESP32Client("192.168.1.73")` directly.

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| App fails to start / `c10.dll` / `[WinError 1114]` on Windows | Ensure **torch is imported before pandas** (already done in `app.py`/`conftest.py`). Don't add a script that imports pandas first. |
| "YOLOv8n COCO model" warning | Expected. A pre-built COCO detector is the non-teachable fallback; it works out of the box. A custom `banana_detector.pt` in `main/models/` is picked up automatically if present. |
| ESP-32 **offline** in the app | The box must be on the same network as the app. Check the IP it prints on Serial. Try pinging it, or confirm port 80 is open. |
| No image from ESP-32 ("no captured image yet") | The first camera frame lands ~15 s after boot. Wait a moment and Fetch again. |
| LCD shows nothing / garbled | Verify I2C addr `0x27` and SDA=21/SCL=22. The box flashes a verdict only when `POST /verdict` succeeds (check the "📟 Verdict sent" line in the app). |
| `POST /verdict` returns 400 | A line exceeded 16 chars. The app pre-truncates via `lcd_line.truncate`; if you call the client directly, keep each line ≤ 16 ASCII chars. |
| Compile error: `analyzeFruitData` redefined | You have both `FruitAnalysis.cpp` **and** `FruitAnalysis-0.cpp` in the sketch folder. Delete the `-0` duplicate. |
| Camera image is black / not found | The ESP-32-CAM IP in `cam_ip` is wrong, or the CAM isn't powered / on the network. |

---

## One-line quick start

```bash
pip install -r main/requirements.txt && streamlit run main/app.py
# no hardware?  python main/hardware/simulator.py   (then use 127.0.0.1:8600 in the app)
```
