"""
Synthetic multi-target stress tests for ``radar_tracker/tracker.py`` and ``radar_tracker/visualize_tracks.py``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pymap3d as pm

from radar_tracker import (
    MultiTargetTracker,
    TrackerConfig,
    create_certainty_visualization,
    create_interactive_html_map,
    create_visualization,
)


class TestMultiTargetTrackerSynthetic(unittest.TestCase):
    """Synthetic multi-target stress tests: coasting, 3D crossing, certainty lifecycle, exports, and visualizers."""

    def test_coasting_across_missed_detections_and_clutter_rejection(self) -> None:
        """Verify a track bridges a 3-second detection gap while rejecting isolated clutter."""
        lat0, lon0, alt0 = -29.32, 22.10, 150.0
        raw_plots = []
        plot_id = 1
        valid_times = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
        for t in valid_times:
            lat, lon, alt = pm.enu2geodetic(20.0 * t, 0.0, 0.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": plot_id,
                    "timestamp": 1700000000.0 + t,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude": float(alt),
                }
            )
            plot_id += 1

        for t_clutter, offset in [(1.5, 500.0), (3.5, -600.0), (5.5, 800.0)]:
            lat, lon, alt = pm.enu2geodetic(offset, offset, 40.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": plot_id,
                    "timestamp": 1700000000.0 + t_clutter,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude": float(alt),
                }
            )
            plot_id += 1

        tracks_out, confirmed, rejected, _ = MultiTargetTracker().reconstruct(raw_plots)
        self.assertEqual(len(tracks_out), 1)
        self.assertEqual(len(tracks_out[0]["plot_ids"]), len(valid_times))
        self.assertEqual(sum(len(t.plots) for t in rejected), 3)
        self.assertAlmostEqual(tracks_out[0]["average_speed_mps"], 20.0, delta=0.2)
        self.assertAlmostEqual(tracks_out[0]["average_heading_deg"], 90.0, delta=0.5)

    def test_3d_crossing_without_identity_switch(self) -> None:
        """Verify two targets crossing at the same (x, y) at different altitudes do not swap IDs."""
        lat0, lon0, alt0 = -29.32, 22.10, 150.0
        raw_plots = []
        target_a_ids = []
        target_b_ids = []
        pid = 1
        for step in range(8):
            t = float(step)
            lat_a, lon_a, alt_a = pm.enu2geodetic(0.0, 20.0 * (t - 4.0), 50.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": pid,
                    "timestamp": 1700000000.0 + t,
                    "latitude": float(lat_a),
                    "longitude": float(lon_a),
                    "altitude": float(alt_a),
                }
            )
            target_a_ids.append(pid)
            pid += 1

            lat_b, lon_b, alt_b = pm.enu2geodetic(20.0 * (t - 4.0), 0.0, -50.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": pid,
                    "timestamp": 1700000000.0 + t + 0.005,
                    "latitude": float(lat_b),
                    "longitude": float(lon_b),
                    "altitude": float(alt_b),
                }
            )
            target_b_ids.append(pid)
            pid += 1

        tracks_out, _, _, _ = MultiTargetTracker().reconstruct(raw_plots)
        self.assertEqual(len(tracks_out), 2)
        reconstructed_sets = [t["plot_ids"] for t in tracks_out]
        self.assertIn(target_a_ids, reconstructed_sets)
        self.assertIn(target_b_ids, reconstructed_sets)

    def test_certainty_score_lifecycle(self) -> None:
        """Verify track certainty starts at 30, adds +20 on hits, and deducts -7 on missed scans."""
        lat0, lon0, alt0 = -29.32, 22.10, 150.0
        raw_plots = []
        for pid, t in enumerate([0.0, 1.0, 3.0, 4.0], start=1):
            lat, lon, alt = pm.enu2geodetic(20.0 * t, 0.0, 0.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": pid,
                    "timestamp": 1700000000.0 + t,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude": float(alt),
                }
            )
        lat_c, lon_c, alt_c = pm.enu2geodetic(900.0, 900.0, 50.0, lat0, lon0, alt0)
        raw_plots.append(
            {
                "id": 99,
                "timestamp": 1700000002.0,
                "latitude": float(lat_c),
                "longitude": float(lon_c),
                "altitude": float(alt_c),
            }
        )

        tracks_out, confirmed, rejected, _ = MultiTargetTracker().reconstruct(raw_plots)
        self.assertEqual(len(tracks_out), 1)
        self.assertEqual(len(rejected), 1)
        certs = [p["certainty"] for p in tracks_out[0]["points"]]
        self.assertEqual(certs, [30.0, 50.0, 63.0, 83.0])

    def test_empty_input_and_csv_and_folder_exports(self) -> None:
        """Verify empty input handling, RTS coordinate export mode, CSV export, and per-track folder export."""
        self.assertEqual(MultiTargetTracker().reconstruct([]), ([], [], [], []))

        lat0, lon0, alt0 = -29.32, 22.10, 150.0
        raw_plots = []
        for pid in range(1, 6):
            lat, lon, alt = pm.enu2geodetic(15.0 * pid, 10.0 * pid, 0.0, lat0, lon0, alt0)
            raw_plots.append(
                {
                    "id": pid,
                    "timestamp": 1700000000.0 + pid,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude": float(alt),
                }
            )

        cfg_rts = TrackerConfig(
            kinematic_filter_method="hampel_kalman_rts",
        )
        tracks_out, confirmed, rejected, _ = MultiTargetTracker(cfg_rts).reconstruct(raw_plots)
        self.assertEqual(len(tracks_out), 1)
        self.assertIn("raw_latitude", tracks_out[0]["points"][0])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "tracks.csv")
            MultiTargetTracker.export_tracks_csv(tracks_out, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)

            out_folders = os.path.join(tmpdir, "tracks_dir")
            MultiTargetTracker.export_track_folders(tracks_out, out_folders)
            self.assertTrue(os.path.exists(os.path.join(out_folders, "T1", "T1_track.json")))
            self.assertTrue(os.path.exists(os.path.join(out_folders, "T1", "T1_points.csv")))

            # Also verify visualization generators run cleanly on synthetic output
            plots_file = Path(tmpdir) / "plots.json"
            tracks_file = Path(tmpdir) / "tracks.json"
            png_file = Path(tmpdir) / "vis.png"
            cert_file = Path(tmpdir) / "cert.png"
            html_file = Path(tmpdir) / "map.html"
            plots_file.write_text(json.dumps(raw_plots), encoding="utf-8")
            tracks_file.write_text(json.dumps(tracks_out), encoding="utf-8")

            create_visualization(plots_file, tracks_file, png_file, raw_plots=raw_plots, tracks=tracks_out)
            create_certainty_visualization(confirmed, rejected, cfg_rts, cert_file)
            create_interactive_html_map(raw_plots, tracks_out, html_file)
            self.assertTrue(png_file.exists())
            self.assertTrue(cert_file.exists())
            self.assertTrue(html_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
