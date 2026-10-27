"""
test_analytics.py
-----------------
Unit tests for the analytics logging, 30-day querying, and metrics calculations.
"""

import sys
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.analytics_store import (
    compute_ripeness_metrics,
    log_inspection,
    load_inspections,
    get_summary_metrics,
    seed_demo_data,
    LOG_FILE,
)


class TestAnalyticsStore(unittest.TestCase):

    def test_compute_ripeness_metrics_approved_ripe(self):
        stage, score, decision = compute_ripeness_metrics(
            avg_green=5.0,
            avg_yellow=85.0,
            avg_brown=10.0,
            bunch_status="ACCEPTABLE",
            color_signal="ripe",
        )
        self.assertEqual(decision, "Approved")
        self.assertEqual(stage, "Ripe")
        self.assertGreaterEqual(score, 70.0)

    def test_compute_ripeness_metrics_approved_semiripe(self):
        stage, score, decision = compute_ripeness_metrics(
            avg_green=40.0,
            avg_yellow=50.0,
            avg_brown=5.0,
            bunch_status="SEMI-RIPE",
            color_signal="semi-ripe",
        )
        self.assertEqual(decision, "Approved")
        self.assertEqual(stage, "Semi-Ripe")
        self.assertTrue(40.0 <= score <= 75.0)

    def test_compute_ripeness_metrics_denied_overripe(self):
        stage, score, decision = compute_ripeness_metrics(
            avg_green=2.0,
            avg_yellow=45.0,
            avg_brown=40.0,
            bunch_status="HIGH RISK",
            color_signal="overripe",
        )
        self.assertEqual(decision, "Denied")
        self.assertEqual(stage, "Overripe")
        self.assertGreaterEqual(score, 75.0)

    def test_log_inspection_and_load(self):
        ts = datetime.now() - timedelta(days=2)
        record = log_inspection(
            bunch_id="test_unit_bunch_001",
            banana_count=8,
            avg_green=10.0,
            avg_yellow=80.0,
            avg_brown=10.0,
            bunch_status="ACCEPTABLE",
            confidence="HIGH",
            dominant_problem="None",
            reasoning="Unit test bunch",
            gas_ppm=0.5,
            color_signal="ripe",
            timestamp=ts,
        )
        self.assertEqual(record["decision"], "Approved")
        self.assertEqual(record["banana_count"], 8)

        df = load_inspections(days=30)
        self.assertFalse(df.empty)
        # Check that the test bunch is in the loaded dataframe
        matches = df[df["bunch_id"] == "test_unit_bunch_001"]
        self.assertGreaterEqual(len(matches), 1)

    def test_get_summary_metrics(self):
        df = load_inspections(days=30)
        metrics = get_summary_metrics(df)
        self.assertIn("total_bunches", metrics)
        self.assertIn("approved_count", metrics)
        self.assertIn("denied_count", metrics)
        self.assertIn("approval_rate", metrics)
        self.assertIn("avg_ripeness", metrics)
        self.assertGreater(metrics["total_bunches"], 0)
        self.assertEqual(metrics["approved_count"] + metrics["denied_count"], metrics["total_bunches"])
        self.assertAlmostEqual(metrics["approval_rate"] + metrics["denial_rate"], 100.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
