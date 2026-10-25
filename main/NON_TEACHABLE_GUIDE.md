# Non-Teachable Banana Ripeness Detector — Implementation Guide

## Project Context

> This is a **capstone competition** project. A working prototype must be demonstrated to the evaluator/teacher.

### Rounds

| Round | Target | Scope |
|-------|--------|-------|
| **Round 1** | **70% complete** | Hardware prototype + working software detection pipeline |
| Round 2 | TBD | Database integration for storing results over time |
| Round 3 | TBD | Mobile app for field use |

### Team Division

| Team | Responsibility |
|------|----------------|
| **Hardware team** | Building the ESP-32 prototype (camera, DHT-11, wiring, power, enclosure) |
| **Software team** (you) | Detection accuracy, color analysis, sensor fusion, UI — everything that runs on this machine |

### Current Priority

The hardware prototype is being built in parallel. Until it arrives, the **sole focus is the detection software** — making it as accurate as possible so it works reliably the moment the hardware is connected. Do not build UI for ESP-32 integration yet; keep the software working with image uploads and simulated sensor data for now.

### Multi-Fruit Support

> **IMPORTANT:** The system is not limited to bananas. The color segmentation, contour detection, and fusion engine must be designed and structured so that **other fruits can be added later** without rewriting the core pipeline. Each fruit type will have its own HSV thresholds, ripeness stages, and fusion rules. The architecture must support this extensibility from the start.

---

## Goal
Build a ripeness detection system that uses **zero machine learning models**. No `.pt` files, no `.h5` files, no training, no fine-tuning, no YOLO. Everything runs on **traditional computer vision** (color segmentation, contour detection, morphological operations) and **sensor data**.

---

## Hardware Architecture (ESP-32 Platform)

```
┌─────────────────────────────────────────────┐
│              ESP-32 Microcontroller           │
│                                               │
│  ┌─────────────┐    ┌─────────────────────┐  │
│  │  OV2640/    │    │   DHT-11 Sensor     │  │
│  │  OV5640     │    │  (Temp + Humidity)  │  │
│  │  Camera     │    │                     │  │
│  └──────┬──────┘    └──────────┬──────────┘  │
│         │                      │             │
│  ┌──────┴──────────────────────┴──────────┐  │
│  │     Image Capture + Sensor Read         │  │
│  │     (WiFi / Serial transmission)        │  │
│  └───────────────────┬────────────────────┘  │
└──────────────────────┼──────────────────────┘
                       │ WiFi / Serial
                       ▼
          ┌────────────────────────┐
          │  This Application      │
          │  (Streamlit + Python)  │
          │  - Receives image +    │
          │    sensor data         │
          │  - Traditional CV      │
          │    (color + contours)  │
          │  - Fusion verdict      │
          └────────────────────────┘
```

### Sensors

| Sensor | Measures | Range | Role in Detection |
|--------|----------|-------|-------------------|
| **DHT-11** | Temperature + Humidity | 0-50°C, 20-90% RH | Ambient conditions affect ripening rate; temp spike + humidity = faster ripening |
| **DHT-22** (optional upgrade) | Temperature + Humidity | -40-80°C, 0-100% RH | Better accuracy than DHT-11 |
| **MQ-135** (optional) | Air quality / gases | 10-300 ppm | Detects ethylene and CO2 — direct ripening signal |

> **Note:** The DHT-11 does NOT detect gas/ethylene. It measures temperature and humidity. These are **indirect** ripeness indicators (heat + humidity accelerate ripening). For direct ethylene detection, add an MQ-135 sensor.

### ESP-32 Data Format

The ESP-32 sends a JSON payload:

```json
{
  "image": "<base64_encoded_jpeg>",
  "sensor": {
    "temperature_c": 25.3,
    "humidity_pct": 72.1,
    "gas_ppm": null
  },
  "timestamp": "2026-01-15T10:30:00Z",
  "device_id": "esp32-cam-001"
}
```

---

## Current Progress

### Completed

| Component | Status | Notes |
|-----------|--------|-------|
| `app.py` | **Done** | Streamlit UI with sidebar, image upload, color breakdown, verdict display |
| `vision/color_analyzer.py` | **Done** | HSV color segmentation (green/yellow/brown), per-banana analysis, annotations |
| `fusion/fusion_engine.py` | **Done** | Gas + color fusion engine with semi-ripe classification |
| `sensor/gas_reader.py` | **Done** | MCP-9964 ethylene reader (demo mode + serial mode) |
| `utils/image_utils.py` | **Done** | Crop, annotate, color conversion utilities |
| `bunch_analyzer.py` | **Done** | Orchestrates detection → color analysis → fusion |
| Semi-ripe classification | **Done** | Detects mid-ripening when green + yellow both present |
| Detection fallback | **Done** | 3-pass detection (YOLO → CLAHE enhanced → all-class fallback) |
| Background-resilient ripeness | **Done** | `vision/banana_mask.py` — masks to the banana body (center-touching component) before HSV, so stand/crate/other-banana colors don't skew ripeness |
| Adaptive detection | **Done** | Infer at 0.10 → relative score filter (max(best×0.30, 0.08)) → score-ordered NMS (IoU 0.5). UI has Auto mode (no floor) / Manual mode (threshold = floor) |

### Remaining

| Component | Status | Notes |
|-----------|--------|-------|
| **Non-teachable detection** | **TODO** | Replace YOLO with contour-based detection |
| **DHT-11 sensor integration** | **TODO** | Update gas_reader.py for temp/humidity |
| **ESP-32 communication** | **TODO** | HTTP endpoint or serial reader for ESP-32 data |
| **Morphological cleanup** | **TODO** | Noise removal pipeline for contour detection |
| **Shape validation** | **TODO** | Aspect ratio and area filtering on contours |

---

## Scope: Non-Teachable Pipeline (No ML)

### Detection: Contour-Based Banana Localization

Since we cannot use YOLO or any trained model, detection uses **traditional CV**:

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: HSV Color Segmentation          │
│ - Convert BGR → HSV                      │
│ - Apply thresholds for banana yellowness │
│ - Binary mask of banana-colored pixels   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Step 2: Morphological Operations         │
│ - Closing (dilate then erode) — fills    │
│   gaps in banana shape                   │
│ - Opening (erode then dilate) — removes  │
│   small noise (leaves, specks)           │
│ - Gaussian blur to smooth edges          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Step 3: Contour Detection                │
│ - Find contours on cleaned mask          │
│ - For each contour:                      │
│   · Compute bounding rect & aspect ratio │
│   · Compute area relative to image       │
│   · Filter by:                           │
│     - Aspect ratio: 0.15 – 7.0          │
│     - Area: 2% – 50% of image           │
│     - Minimum solidity (>0.3)           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Step 4: Per-Contour Analysis             │
│ - Extract contour mask                   │
│ - Run HSV color analysis on mask only    │
│ - Classify: green / yellow / brown /     │
│   semi-ripe                              │
└────────────────┬────────────────────────┘
                 │
                 ▼
         Individual Banana Results
```

### HSV Color Thresholds

```
GREEN (unripe):   H: 35-85,   S: 40-255, V: 40-255
YELLOW (ripe):    H: 15-35,   S: 40-255, V: 40-255
BROWN (overripe): H: 0-15,    S: 20-100, V: 30-180
```

### Semi-Ripe Classification (Per Banana)

```
IF brown_pct > 20%     →  overripe
ELIF green_pct > 20% AND yellow_pct > 40%  →  semi-ripe
ELSE                   →  highest of green / yellow
```

### Bunch-Level Fusion

```
IF any banana is overripe    →  HIGH RISK
ELIF >=20% bananas semi-ripe  →  SEMI-RIPE
ELIF average yellow > 50%    →  ACCEPTABLE (ripe)
ELIF average green > 50%     →  ACCEPTABLE (unripe)
ELSE                         →  ACCEPTABLE (mixed)
```

### Ripening Time Estimation

> **IMPORTANT:** This feature is part of the planned scope and must be designed into the architecture from the start. The system should estimate how many days remain until a banana reaches each ripeness stage (ripe → overripe → spoiled).

Color analysis alone tells you **current state** — not how fast the banana is moving toward spoilage. A banana at 40% green in a cold room (15°C) may take 5 more days to fully ripen. The same-looking banana at 30°C may be overripe within 24 hours.

The DHT-11 temperature and humidity readings act as **rate modifiers** on the ripening process. The estimation uses:

```
Position on curve  →  from color analysis (how far along)
Speed of curve     →  from DHT-11 temperature + humidity
Time estimate      →  position divided by speed
```

#### Ripening Progress Score

Assign each banana a progress score from 0 to 100 based on its color breakdown:

```
progress = (yellow_pct * 0.5) + ((100 - green_pct) * 0.3) + (brown_pct * 2.0)

Example:
- Unripe:  90% green, 8% yellow, 0% brown → progress = (8*0.5) + (10*0.3) + (0*2.0) = 4 + 3 + 0 = 7
- Semi-ripe: 30% green, 60% yellow, 5% brown → progress = (60*0.5) + (70*0.3) + (5*2.0) = 30 + 21 + 10 = 61
- Overripe: 5% green, 75% yellow, 18% brown → progress = (75*0.5) + (95*0.3) + (18*2.0) = 37.5 + 28.5 + 36 = 102 (clamped to 100)
```

#### Temperature/Humidity Rate Modifier

Use the DHT-11 readings to estimate ripening speed relative to an ideal baseline (22°C, 65% RH):

```
temp_factor = 1.0 at 22°C
temp_factor ≈ 2.0 at 32°C (roughly doubles per +10°C Q10 rule)
temp_factor ≈ 0.5 at 12°C

humidity_factor = 1.0 at 65% RH
humidity_factor ≈ 1.3 at >80% RH (accelerates)
humidity_factor ≈ 0.8 at <50% RH (slows)

rate = temp_factor * humidity_factor
```

#### Time-to-Stage Estimation

```
days_to_ripe     = (75 - progress) / (25 * rate)    # 75 = midpoint score for "ripe"
days_to_overripe = (90 - progress) / (15 * rate)    # 90 = score for "approaching overripe"
days_to_spoiled  = (100 - progress) / (10 * rate)   # 100 = fully overripe
```

When rate < 1 (cold/dry), estimates extend. When rate > 1 (hot/humid), estimates shrink.

#### Display Output

```
Ripening Timeline for Bunch #1
  Now:          SEMI-RIPE (progress: 61/100)
  Ripe in:      ~0.6 days
  Overripe in:  ~2.0 days
  Spoiled in:   ~4.1 days
  Rate:         Normal (22°C, 70% RH)
```

---

## DHT-11 Sensor Integration

### Data Contract

```python
{
    "temperature_c": float,    # 0.0 – 50.0 °C
    "humidity_pct": float,     # 0.0 – 100.0 %
    "status": str,             # "LOW", "MODERATE", "HIGH", "CRITICAL"
    "timestamp": str           # ISO timestamp
}
```

### DHT-11 Thresholds (Temp + Humidity as Ripeness Proxy)

```
Temperature ripeness effect:
- < 15°C  →  ripening very slow (cold storage)
- 15-20°C  →  slow ripening
- 20-25°C  →  optimal ripening
- > 25°C  →  fast ripening, risk of spoilage

Humidity effect:
- < 60% RH  →  drying, slows ripening visually
- 60-80% RH  →  normal ripening
- > 80% RH  →  accelerated ripening + mold risk
```

### Combined Status Logic

```python
temp = reading["temperature_c"]
humidity = reading["humidity_pct"]

if temp > 28 and humidity > 80:
    status = "CRITICAL"   # Hot + humid = rapid ripening
elif temp > 25 or humidity > 75:
    status = "HIGH"       # Accelerating
elif 20 <= temp <= 25 and 60 <= humidity <= 75:
    status = "MODERATE"   # Normal
else:
    status = "LOW"        # Slow/stalled
```

### Future: Adding MQ-135 for Ethylene

When the MQ-135 gas sensor is added alongside DHT-11:

```python
{
    "temperature_c": float,
    "humidity_pct": float,
    "ethylene_ppm": float,   # from MQ-135
    "status": str,
    "timestamp": str
}
```

Thresholds:
- `LOW`:       < 0.5 ppm ethylene
- `MODERATE`:  0.5 – 2.0 ppm
- `HIGH`:      2.0 – 5.0 ppm
- `CRITICAL`:  > 5.0 ppm

---

## ESP-32 Integration

### Communication Method

**Option A: HTTP POST (Recommended)**
```
POST http://<server>:8501/upload
Content-Type: application/json

Body: { image: "<base64>", sensor: {...}, timestamp: "...", device_id: "..." }
```

**Option B: Serial (USB)**
```
ESP-32 → USB → serial port → Python reads line-by-line
Use: pyserial library
```

### ESP-32 Firmware Responsibilities

1. Capture image from OV2640 camera
2. Read DHT-11 (temp + humidity)
3. (Optional) Read MQ-135 (ethylene)
4. Encode image as base64 JPEG
5. Send JSON payload via WiFi HTTP POST
6. Handle reconnect if WiFi drops

---

## Files

```
app.py                            # Streamlit UI — entry point
bunch_analyzer.py                 # Orchestrator: detection → analysis → fusion
classification/ripeness_classifier.py  # Per-crop ripeness classification
detection/banana_detector.py      # YOLOv8-based banana detection (to be replaced with contours)
fusion/fusion_engine.py           # Sensor + color fusion → bunch verdict
processing/bunch_analyzer.py      # Processing layer
sensor/gas_reader.py              # DHT-11 reader (demo + serial modes)
tests/test_bunch_logic.py         # Unit tests — bunch fusion logic
tests/test_pipeline.py            # Integration tests — full pipeline
utils/image_utils.py              # Crop, annotate, color conversion
vision/color_analyzer.py          # HSV color segmentation + classification
```

---

## Dependencies

```
opencv-python
numpy
pillow
streamlit
pyserial          # For ESP-32 serial communication
```

**No PyTorch. No TensorFlow. No ultralytics. No ML dependencies.**

---

## Testing

### Unit Tests

| Module | Test Strategy |
|--------|---------------|
| `color_analyzer` | Create synthetic images with known green/yellow/brown regions, verify percentages |
| `gas_reader` | Demo mode returns values in expected range |
| `fusion_engine` | All rule combinations: gas LOW/MODERATE/HIGH/CRITICAL x color unripe/ripe/overripe/semi-ripe |
| `banana_detector` | Synthetic images with banana-shaped objects, verify contour detection |

### Integration Tests

- Upload a market photo → verify per-banana breakdown + bunch verdict
- Simulate DHT-11 readings at different temp/humidity → verify status changes
- Edge case: single banana, dark photo, overlapping bananas, fully rotten banana

---

## Confidence Scoring

Every detection result should include a confidence score so the user knows how sure the system is.

> **Default confidence threshold is 0.20.** This was chosen so that the detection pipeline is permissive enough to catch bananas in low-light or partially obscured images. The Streamlit UI slider also defaults to 0.20. Users can raise it for stricter filtering.

### Per-Banana Confidence

```
confidence = shape_match × color_gradient × size_consistency

shape_match (0-1):
  - Aspect ratio within banana range (0.2-5.0) → 1.0
  - Within 2x of expected banana length → 1.0
  - Outside expected range → 0.3

color_gradient (0-1):
  - Colors change along the long axis (green→yellow→brown) → 1.0
  - Uniform color (all one shade) → 0.4 (could be an apple)
  - Mixed with no gradient pattern → 0.5

size_consistency (0-1):
  - Within 2x size range of other detected objects → 1.0
  - Wildly different from others → 0.4 (could be a different fruit)
```

### Confidence Tiers

| Score | Label | Behavior |
|-------|-------|----------|
| > 0.7 | HIGH | Process normally, show result |
| 0.4 – 0.7 | MEDIUM | Process but flag "uncertain detection" |
| < 0.4 | LOW | Skip, show "no clear bananas detected" |

### Handling Non-Banana Objects

When the system detects objects that score LOW on confidence:
1. Do NOT classify them as bananas
2. Show "No clear bananas detected in this image"
3. If possible, indicate what was detected (e.g., "Found 1 round green object — possibly an apple")
4. Never give a false positive — it's better to say "I don't know" than be confidently wrong

### Edge Case Handling

| Scenario | Strategy |
|----------|----------|
| Single banana, fully rotten | 3-pass detection + all-class fallback with shape filter |
| Dark photo | Contrast enhancement pass (CLAHE) |
| Overlapping bananas | Contour splitting by shape analysis |
| Mixed fruit bowl | Low confidence on all → "no clear bananas detected" |
| No fruit at all | 3 passes all return empty → "no objects found" |

---

## Multi-Fruit Architecture

> **IMPORTANT:** This is part of the project scope. The system is not banana-only. The architecture must be designed so that other fruits can be added by changing configuration, not rewriting code.

### Why It Matters

The current implementation is banana-specific: HSV thresholds, ripeness stages, shape filters, and fusion rules are all hardcoded for bananas. Adding another fruit currently requires code changes. This must change.

### Fruit Profile System

Each fruit type gets a profile — a configuration object that describes everything the system needs to know:

```python
FRUIT_PROFILES = {
    "banana": {
        "name": "Banana",
        "colors": {
            "unripe":   {"hsv_range": [(35, 40, 40), (85, 255, 255)], "label": "green"},
            "ripe":     {"hsv_range": [(15, 40, 40), (35, 255, 255)], "label": "yellow"},
            "overripe": {"hsv_range": [(0, 20, 30),  (15, 100, 180)], "label": "brown"},
        },
        "shape": {
            "min_aspect_ratio": 0.2,
            "max_aspect_ratio": 5.0,
            "min_area_pct": 2.0,
            "max_area_pct": 50.0,
            "has_gradient": True,        # colors change along length
            "has_stem": True,            # stem-like structure expected
        },
        "ripeness_stages": ["unripe", "semi-ripe", "ripe", "overripe"],
        "semi_ripe_rule": "green > 20 AND yellow > 40",
        "ripening": {
            "progress_formula": "(yellow*0.5) + ((100-green)*0.3) + (brown*2.0)",
            "ripe_threshold": 75,
            "overripe_threshold": 90,
        }
    },
    # Future: "apple", "tomato", "orange", etc.
}
```

### How to Add a New Fruit

With this architecture, adding a new fruit is a config change, not a code change:

1. Define HSV color ranges for its ripeness stages
2. Define shape constraints (aspect ratio, area range)
3. Define ripeness stage names and thresholds
4. Define the semi-ripe classification rule
5. Done — the same detection + analysis + fusion pipeline works automatically

### Pipeline Remains Unchanged

The core pipeline (detect → analyze → fuse → display) does not need to know which fruit it's processing. It reads the active profile and applies the rules from it:

```
Input: Image + Active Fruit Profile
       ↓
Detect (uses profile's shape rules)
       ↓
Analyze colors (uses profile's HSV ranges)
       ↓
Classify (uses profile's ripeness stages + semi-ripe rule)
       ↓
Fuse with sensor (uses profile's ripening formula)
       ↓
Display (uses profile's stage names + colors)
```

### Multi-Fruit Output (Future)

When multiple fruit types are configured and the system detects multiple types in one image:

```
Results:
  Bananas (3):
    #1 — semi-ripe, eat in ~1 day
    #2 — ripe, eat today
    #3 — overripe, discard
  Apples (2):
    #4 — unripe, wait ~3 days
    #5 — ripe, ready
```

This is Round 2/3 scope. The architecture must support it, but full multi-fruit detection is not required for Round 1.

---

## What We're NOT Building

- No ML model training or fine-tuning at runtime
- No object-detection *training* — the YOLOv8 model ships pre-trained and is
  never retrained; "non-teachable" means no learning loop, not no detector
- No cloud inference — all processing is local
- No mobile app — web interface only (Streamlit)
- No database — runs on uploaded images and live sensor data

## Design Decisions & Trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Adaptive detection | Fixed confidence threshold | One threshold can't fit all images (occluded bananas vs. duplicate boxes). We infer at 0.10, then filter by the image's own best score + NMS. |
| Banana-mask color analysis | Whole-box pixel average | Background (green stand, yellow shelf) used to skew ripeness. Now we mask to the banana body (center-touching component) before HSV. |
| Detection as a floor, not a cutoff | Hard confidence cutoff | The user's threshold removes only boxes *below* the floor after relative filtering — weak-but-real bunches on clean images still survive. |
