"""
Unit tests for external Fortem R20 validation utilities (``validations/``).
"""

from __future__ import annotations

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
VAL_DIR = os.path.join(BASE_DIR, "validations")
VAL_SCRIPTS = os.path.join(VAL_DIR, "scripts")
if VAL_DIR not in sys.path:
    sys.path.insert(0, VAL_DIR)
if VAL_SCRIPTS not in sys.path:
    sys.path.insert(0, VAL_SCRIPTS)

import numpy as np
import pymap3d as pm

from convert_fortem import distance_xy_m, rotate_lla, to_iso_utc
from geo_metrics import (
    compute_series_kinematics,
    evaluate_against_gps,
    parse_ts,
    to_enu_xy,
)


class TestValidationUtilitiesUnit(unittest.TestCase):
    """Unit tests for Fortem R20 conversion and GPS evaluation functions."""

    def test_timestamp_conversion_and_enu_rotation(self) -> None:
        """Verify ISO-8601 UTC conversion, parse_ts, and 2-degree yaw rotation."""
        epoch = 1759866491.25
        iso_str = to_iso_utc(epoch)
        self.assertTrue(iso_str.endswith("Z"))
        self.assertAlmostEqual(parse_ts(iso_str), epoch, places=3)
        self.assertAlmostEqual(parse_ts(epoch), epoch, places=6)

        lat_r, lon_r, alt_r = rotate_lla(35.7260, -78.6930, 50.0)
        self.assertAlmostEqual(alt_r, 50.0, places=2)
        d_m = distance_xy_m((35.7260, -78.6930), (lat_r, lon_r))
        self.assertGreater(d_m, 0.5)

    def test_compute_series_kinematics_and_evaluate_against_gps(self) -> None:
        """Verify reference series kinematics and linear GPS interpolation error calculation."""
        lat0, lon0, alt0 = 35.724976, -78.691933, 10.0
        gps_rows = []
        est_rows = []
        for step in range(10):
            t = 1000.0 + float(step)
            lat_g, lon_g, _ = pm.enu2geodetic(8.0 * step, 0.0, 40.0, lat0, lon0, alt0)
            gps_rows.append({"timestamp": t, "latitude": float(lat_g), "longitude": float(lon_g), "altitude_gps": 50.0})
            # Estimated trajectory has a constant +3m East offset and +4m North offset (5m 2D error)
            lat_e, lon_e, _ = pm.enu2geodetic(8.0 * step + 3.0, 4.0, 40.0, lat0, lon0, alt0)
            est_rows.append({"timestamp": t, "latitude": float(lat_e), "longitude": float(lon_e), "altitude": 50.0})

        gps_enriched = compute_series_kinematics(gps_rows)
        est_enriched = compute_series_kinematics(est_rows)
        gps_ts = np.array([r["epoch_s"] for r in gps_enriched], dtype=float)
        gps_xy = to_enu_xy(gps_enriched)
        gps_spds = np.array([float(r["speed_mps"]) for r in gps_enriched[1:]], dtype=float)

        metrics, errs = evaluate_against_gps(est_enriched, gps_ts, gps_xy, gps_spds)
        self.assertEqual(metrics["samples"], 10)
        self.assertAlmostEqual(metrics["rmse_xy_m"], 5.0, delta=0.05)
        self.assertAlmostEqual(metrics["mean_speed_mps"], 8.0, delta=0.05)
        self.assertAlmostEqual(metrics["speed_mae_vs_gps_mps"], 0.0, delta=0.05)
        self.assertEqual(len(errs), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
