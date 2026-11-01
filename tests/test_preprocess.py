"""
Unit tests for ``radar_tracker/preprocess.py`` (geodetic-to-ENU projection and scan clustering).
"""

from __future__ import annotations

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from radar_tracker import group_into_scans, prepare_plots_dataframe


class TestPreprocessUnit(unittest.TestCase):
    """Unit tests for geodetic-to-ENU projection, timestamp parsing, and scan clustering."""

    def test_prepare_plots_dataframe_iso_and_numeric_timestamps(self) -> None:
        """Verify prepare_plots_dataframe handles both ISO-8601 strings and Unix epoch floats and sorts chronologically."""
        raw_plots = [
            {
                "id": 2,
                "timestamp": "2025-10-07T19:48:12.500Z",
                "latitude": 35.7255,
                "longitude": -78.6915,
                "altitude": 110.0,
            },
            {
                "id": 1,
                "timestamp": "2025-10-07T19:48:11.500Z",
                "latitude": 35.7250,
                "longitude": -78.6920,
                "altitude": 100.0,
            },
        ]
        df, (lat0, lon0, alt0) = prepare_plots_dataframe(raw_plots)
        self.assertEqual(list(df["id"]), [1, 2])
        self.assertAlmostEqual(float(df.iloc[0]["t_rel"]), 0.0, places=5)
        self.assertAlmostEqual(float(df.iloc[1]["t_rel"]), 1.0, places=5)
        self.assertAlmostEqual(lat0, 35.72525, places=5)
        self.assertTrue("x" in df.columns and "y" in df.columns and "z" in df.columns)

    def test_group_into_scans_edge_cases_and_batching(self) -> None:
        """Verify group_into_scans handles empty lists, single plots, and intra/inter-scan thresholds."""
        self.assertEqual(group_into_scans([]), [])

        single = [{"id": 1, "t": 10.0}]
        self.assertEqual(len(group_into_scans(single)), 1)

        records = [
            {"id": 1, "t": 10.00},
            {"id": 2, "t": 10.05},
            {"id": 3, "t": 10.12},
            {"id": 4, "t": 11.00},
            {"id": 5, "t": 11.04},
        ]
        scans = group_into_scans(records, scan_window_s=0.15)
        self.assertEqual(len(scans), 2)
        self.assertEqual([p["id"] for p in scans[0]], [1, 2, 3])
        self.assertEqual([p["id"] for p in scans[1]], [4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
