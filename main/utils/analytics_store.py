"""
analytics_store.py
------------------
Persistent timestamped inspection logger and data provider for the Analytics page.
Maintains a 30-day historical log of banana bunch inspections, including:
  - Timestamp
  - Bunch / Image Identifier
  - Total Bananas Detected
  - Ripeness Stage & Average Ripeness Percentage
  - Quality Control Decision (Approved vs Denied)
  - Bunch Status (ACCEPTABLE, SEMI-RIPE, HIGH RISK)
  - Color Percentages (Green %, Yellow %, Brown %)
  - Gas Sensor Readings (Ethylene ppm)
  - Confidence & Dominant Problem
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default path for persistent logs
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_FILE = DATA_DIR / "inspection_logs.csv"

COLUMNS = [
    "timestamp",
    "bunch_id",
    "banana_count",
    "decision",
    "bunch_status",
    "ripeness_stage",
    "avg_ripeness_pct",
    "green_pct",
    "yellow_pct",
    "brown_pct",
    "gas_ppm",
    "confidence",
    "dominant_problem",
    "reasoning",
]


def ensure_data_dir() -> Path:
    """Ensure data directory and file exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        _create_empty_log_file()
        seed_demo_data(force=False)
    return LOG_FILE


def _create_empty_log_file() -> None:
    """Create CSV file with headers."""
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()


def compute_ripeness_metrics(
    avg_green: float,
    avg_yellow: float,
    avg_brown: float,
    bunch_status: str,
    color_signal: Optional[str] = None,
    color_analyses: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, float, str]:
    """
    Compute commercial ripeness stage, numerical ripeness score (0-100%),
    and the Quality Decision ('Approved' vs 'Denied').

    Quality Decision rules:
      - 'Approved': Bunch is ACCEPTABLE or SEMI-RIPE (fit for retail, market or controlled ripening).
      - 'Denied': Bunch is HIGH RISK or UNKNOWN (spoilage risk, overripe spots, ethylene hazard).

    Returns:
      (ripeness_stage, avg_ripeness_pct, decision)
    """
    # Decision
    if bunch_status in ("ACCEPTABLE", "SEMI-RIPE"):
        decision = "Approved"
    else:
        decision = "Denied"

    # Stage determination
    STAGE_MAP = {
        "unripe": "Unripe",
        "semi-ripe": "Semi-Ripe",
        "semi ripe": "Semi-Ripe",
        "ripe": "Ripe",
        "overripe": "Overripe",
        "over ripe": "Overripe",
    }
    if color_signal:
        stage = STAGE_MAP.get(color_signal.lower().strip(), color_signal.capitalize())
    elif avg_brown > 20:
        stage = "Overripe"
    elif avg_green > 20 and avg_yellow > 30:
        stage = "Semi-Ripe"
    elif avg_yellow >= 50:
        stage = "Ripe"
    elif avg_green >= 50:
        stage = "Unripe"
    else:
        stage = "Ripe" if avg_yellow >= avg_green else "Unripe"

    # Numerical ripeness score (0.0 to 100.0%)
    # Unripe/green contributes minimal ripeness (~10-25%),
    # Yellow contributes prime ripeness (~70-90%),
    # Brown indicates late-stage/overripeness (90-100%).
    # We calibrate a continuous ripeness index:
    # 0% = raw green, 50% = turning/semi-ripe, 85% = fully yellow/prime, 100% = dark overripe
    ripeness_pct = (avg_yellow * 0.85) + (avg_brown * 1.0) + (avg_green * 0.15)
    ripeness_pct = float(np.clip(ripeness_pct, 5.0, 100.0))
    ripeness_pct = round(ripeness_pct, 1)

    return stage, ripeness_pct, decision


def log_inspection(
    bunch_id: str,
    banana_count: int,
    avg_green: float,
    avg_yellow: float,
    avg_brown: float,
    bunch_status: str,
    confidence: str = "MEDIUM",
    dominant_problem: str = "None",
    reasoning: str = "",
    gas_ppm: float = 0.0,
    color_signal: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Log a single bunch inspection to the CSV file with a timestamp.
    """
    ensure_data_dir()

    if timestamp is None:
        timestamp = datetime.now()

    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    stage, ripeness_pct, decision = compute_ripeness_metrics(
        avg_green=avg_green,
        avg_yellow=avg_yellow,
        avg_brown=avg_brown,
        bunch_status=bunch_status,
        color_signal=color_signal,
    )

    record = {
        "timestamp": ts_str,
        "bunch_id": str(bunch_id),
        "banana_count": int(banana_count),
        "decision": decision,
        "bunch_status": bunch_status,
        "ripeness_stage": stage,
        "avg_ripeness_pct": float(ripeness_pct),
        "green_pct": round(float(avg_green), 1),
        "yellow_pct": round(float(avg_yellow), 1),
        "brown_pct": round(float(avg_brown), 1),
        "gas_ppm": round(float(gas_ppm), 2),
        "confidence": confidence,
        "dominant_problem": str(dominant_problem),
        "reasoning": str(reasoning),
    }

    # Append to file
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(record)

    logger.info(f"Logged inspection: {record['bunch_id']} -> {decision} ({stage})")
    return record


def load_inspections(days: Optional[int] = 30) -> pd.DataFrame:
    """
    Load inspection records, optionally filtered to the past `days` days.
    Returns a sorted pandas DataFrame (newest first).
    """
    ensure_data_dir()

    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        logger.error(f"Error reading inspection log file: {e}")
        return pd.DataFrame(columns=COLUMNS)

    if df.empty:
        return df

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Ensure numeric types
    for col in ["banana_count", "avg_ripeness_pct", "green_pct", "yellow_pct", "brown_pct", "gas_ppm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filter by days if requested
    if days is not None and days > 0 and not df.empty:
        # Use latest timestamp in data or today, whichever is more recent
        ref_date = max(datetime.now(), df["timestamp"].max())
        cutoff_date = ref_date - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff_date]

    # Sort descending
    df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    return df


def get_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics and KPIs from an inspections DataFrame.
    """
    if df.empty:
        return {
            "total_bunches": 0,
            "total_bananas": 0,
            "approved_count": 0,
            "denied_count": 0,
            "approval_rate": 0.0,
            "denial_rate": 0.0,
            "avg_ripeness": 0.0,
            "avg_bananas_per_bunch": 0.0,
            "dominant_stage": "N/A",
            "avg_green": 0.0,
            "avg_yellow": 0.0,
            "avg_brown": 0.0,
        }

    total_bunches = len(df)
    total_bananas = int(df["banana_count"].sum())
    approved_count = int((df["decision"] == "Approved").sum())
    denied_count = int((df["decision"] == "Denied").sum())

    approval_rate = round((approved_count / total_bunches) * 100, 1) if total_bunches > 0 else 0.0
    denial_rate = round((denied_count / total_bunches) * 100, 1) if total_bunches > 0 else 0.0

    avg_ripeness = round(float(df["avg_ripeness_pct"].mean()), 1)
    avg_bananas = round(float(df["banana_count"].mean()), 1)

    stage_counts = df["ripeness_stage"].value_counts()
    dominant_stage = stage_counts.index[0] if not stage_counts.empty else "N/A"

    return {
        "total_bunches": total_bunches,
        "total_bananas": total_bananas,
        "approved_count": approved_count,
        "denied_count": denied_count,
        "approval_rate": approval_rate,
        "denial_rate": denial_rate,
        "avg_ripeness": avg_ripeness,
        "avg_bananas_per_bunch": avg_bananas,
        "dominant_stage": dominant_stage,
        "avg_green": round(float(df["green_pct"].mean()), 1),
        "avg_yellow": round(float(df["yellow_pct"].mean()), 1),
        "avg_brown": round(float(df["brown_pct"].mean()), 1),
    }


def seed_demo_data(force: bool = False) -> None:
    """
    Seed realistic, high-quality historical inspection records for the past 30 days.
    This guarantees that the Analytics graphs, KPIs, and tables are immediately
    meaningful and visually rich.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing line count
    if not force and LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 20:
                    return  # already seeded
        except Exception:
            pass

    _create_empty_log_file()

    rng = np.random.RandomState(42)
    now = datetime.now()

    records = []
    # Generate ~3 to 7 inspections per day for the last 30 days
    for day_offset in range(30, -1, -1):
        target_date = now - timedelta(days=day_offset)
        n_inspections = rng.randint(3, 8)

        for i in range(n_inspections):
            hour = rng.randint(8, 18)
            minute = rng.randint(0, 60)
            second = rng.randint(0, 60)
            ts = target_date.replace(hour=hour, minute=minute, second=second)

            # Distribution of batch characteristics:
            # 60% Ripe, 22% Semi-Ripe, 10% Unripe, 8% Overripe/High Risk
            dice = rng.rand()

            if dice < 0.60:
                # Prime Ripe Bunch -> APPROVED
                status = "ACCEPTABLE"
                stage = "Ripe"
                yellow = rng.uniform(65.0, 92.0)
                green = rng.uniform(2.0, 15.0)
                brown = rng.uniform(0.5, 12.0)
                gas = rng.uniform(0.2, 1.2)
                confidence = "HIGH"
                problem = "No significant problems"
                reasoning = f"Mostly yellow ({yellow:.1f}%) - ripe and stable."
            elif dice < 0.82:
                # Semi-Ripe Bunch -> APPROVED
                status = "SEMI-RIPE"
                stage = "Semi-Ripe"
                yellow = rng.uniform(38.0, 58.0)
                green = rng.uniform(25.0, 48.0)
                brown = rng.uniform(1.0, 8.0)
                gas = rng.uniform(0.3, 1.8)
                confidence = "MEDIUM"
                problem = "Mid-ripening - green fading to yellow"
                reasoning = "Bananas mid-ripening - suitable for distribution."
            elif dice < 0.92:
                # Unripe Bunch -> APPROVED
                status = "ACCEPTABLE"
                stage = "Unripe"
                green = rng.uniform(65.0, 88.0)
                yellow = rng.uniform(8.0, 25.0)
                brown = rng.uniform(0.0, 5.0)
                gas = rng.uniform(0.1, 0.6)
                confidence = "HIGH"
                problem = "Mostly unripe"
                reasoning = f"Green bunch ({green:.1f}%) - optimal for long-distance transit."
            else:
                # Overripe or Gas Spike Bunch -> DENIED
                status = "HIGH RISK"
                stage = "Overripe"
                brown = rng.uniform(22.0, 45.0)
                yellow = rng.uniform(40.0, 65.0)
                green = rng.uniform(0.0, 8.0)
                gas = rng.uniform(3.5, 9.8)
                confidence = "HIGH" if rng.rand() > 0.3 else "MEDIUM"
                if brown > 25:
                    problem = f"High brown spotting ({brown:.1f}%)"
                    reasoning = f"Overripe bananas ({brown:.1f}% brown) - approaching spoilage."
                else:
                    problem = "Ethylene spike detected"
                    reasoning = f"Ethylene level ({gas:.1f} ppm) exceeds safe threshold."

            banana_count = rng.randint(4, 16)
            bunch_id = f"bunch_{ts.strftime('%Y%m%d')}_{i+1:03d}"

            stage_calc, ripeness_pct, decision = compute_ripeness_metrics(
                avg_green=green,
                avg_yellow=yellow,
                avg_brown=brown,
                bunch_status=status,
                color_signal=stage.lower(),
            )

            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "bunch_id": bunch_id,
                "banana_count": banana_count,
                "decision": decision,
                "bunch_status": status,
                "ripeness_stage": stage_calc,
                "avg_ripeness_pct": ripeness_pct,
                "green_pct": round(green, 1),
                "yellow_pct": round(yellow, 1),
                "brown_pct": round(brown, 1),
                "gas_ppm": round(gas, 2),
                "confidence": confidence,
                "dominant_problem": problem,
                "reasoning": reasoning,
            })

    # Sort chronological
    records.sort(key=lambda r: r["timestamp"])

    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Successfully seeded {len(records)} 30-day inspection records.")
