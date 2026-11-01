"""
End-to-end integration tests on the benchmark ``plots.json`` dataset.
"""

from __future__ import annotations

import json
import os
import random
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from radar_tracker import (
    AVAILABLE_PIPELINES,
    MultiTargetTracker,
    evaluate_kinematic_pipelines,
)


class TestRadarTrackerIntegration(unittest.TestCase):
    """Integration tests on the benchmark dataset plots.json."""

    @classmethod
    def setUpClass(cls) -> None:
        plots_path = os.path.join(BASE_DIR, "plots.json")
        with open(plots_path, "r", encoding="utf-8") as f:
            cls.raw_plots = json.load(f)

    def test_plots_json_full_reconstruction_and_schema(self) -> None:
        """Verify 4 tracks of 178 plots and 71 rejected clutter plots on plots.json."""
        tracks_out, confirmed, rejected, _ = MultiTargetTracker().reconstruct(self.raw_plots)
        self.assertEqual(len(tracks_out), 4)
        self.assertEqual(sum(len(t.plots) for t in rejected), 71)

        seen_ids = set()
        for tr in tracks_out:
            self.assertEqual(len(tr["plot_ids"]), 178)
            self.assertEqual(len(tr["points"]), 178)
            self.assertIsNone(tr["points"][0]["speed_mps"])
            self.assertIsNone(tr["points"][0]["heading_deg"])
            self.assertTrue(0.0 <= tr["average_heading_deg"] < 360.0)
            self.assertTrue(10.0 <= tr["average_speed_mps"] <= 35.0)
            for pid in tr["plot_ids"]:
                self.assertNotIn(pid, seen_ids)
                seen_ids.add(pid)

        self.assertEqual(len(seen_ids), 712)

    def test_shuffled_input_invariance(self) -> None:
        """Verify that shuffling input plot order produces the exact same output tracks."""
        shuffled = list(self.raw_plots)
        random.Random(42).shuffle(shuffled)

        tracks_ordered, _, _, _ = MultiTargetTracker().reconstruct(self.raw_plots)
        tracks_shuffled, _, _, _ = MultiTargetTracker().reconstruct(shuffled)
        self.assertEqual(tracks_ordered, tracks_shuffled)

    def test_all_kinematic_filtering_pipelines(self) -> None:
        """Verify all 11 kinematic filtering pipelines execute and hampel_directional minimizes jumps."""
        _, confirmed, _, _ = MultiTargetTracker().reconstruct(self.raw_plots)
        df_bench = evaluate_kinematic_pipelines(confirmed)
        self.assertEqual(len(df_bench), len(AVAILABLE_PIPELINES))

        by_method = df_bench.set_index("pipeline")
        self.assertLess(
            by_method.loc["hampel_directional", "max_1s_speed_jump_mps"],
            by_method.loc["kalman_rts", "max_1s_speed_jump_mps"],
        )
        self.assertLess(
            by_method.loc["hampel_directional", "mean_speed_std_mps"],
            by_method.loc["raw", "mean_speed_std_mps"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
