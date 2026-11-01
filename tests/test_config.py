"""
Unit tests for ``radar_tracker/config.py`` (TrackerConfig defaults, immutability, and .properties loading).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from radar_tracker import TrackerConfig


class TestTrackerConfigUnit(unittest.TestCase):
    """Unit tests for TrackerConfig defaults, flat/sectioned .properties loading, and error handling."""

    def test_default_instantiation_and_immutability(self) -> None:
        """Verify default values and frozen dataclass immutability."""
        cfg = TrackerConfig()
        self.assertEqual(cfg.kinematic_filter_method, "hampel_directional")
        self.assertAlmostEqual(cfg.initial_certainty, 30.0)
        self.assertAlmostEqual(cfg.certainty_hit_increment, 20.0)
        self.assertAlmostEqual(cfg.certainty_miss_penalty, 7.0)
        self.assertAlmostEqual(cfg.confirmation_certainty_threshold, 50.0)
        self.assertAlmostEqual(cfg.sigma_pos_xy_m, 2.0)
        with self.assertRaises(Exception):
            cfg.min_confirmed_hits = 10  # type: ignore[misc]

    def test_load_from_default_and_custom_properties(self) -> None:
        """Verify loading default tracker.properties and custom flat/sectioned .properties files."""
        default_cfg = TrackerConfig.from_properties()
        self.assertEqual(default_cfg.kinematic_filter_method, "hampel_directional")
        self.assertAlmostEqual(default_cfg.initial_certainty, 30.0)

        with tempfile.NamedTemporaryFile("w", suffix=".properties", delete=False, encoding="utf-8") as tmp:
            tmp.write(
                "[custom_section]\n"
                "kinematic_filter_method = hampel_kalman_rts\n"
                "min_confirmed_hits = 5\n"
                "sigma_pos_xy_m = 15.0\n"
            )
            tmp_path = tmp.name

        try:
            custom_cfg = TrackerConfig.from_properties(tmp_path)
            self.assertEqual(custom_cfg.kinematic_filter_method, "hampel_kalman_rts")
            self.assertEqual(custom_cfg.min_confirmed_hits, 5)
            self.assertAlmostEqual(custom_cfg.sigma_pos_xy_m, 15.0)
        finally:
            os.remove(tmp_path)

    def test_invalid_or_missing_properties_raises_error(self) -> None:
        """Verify FileNotFoundError on missing file and ValueError on unknown property key."""
        with self.assertRaises(FileNotFoundError):
            TrackerConfig.from_properties("non_existent_config_12345.properties")

        with tempfile.NamedTemporaryFile("w", suffix=".properties", delete=False, encoding="utf-8") as tmp:
            tmp.write("unknown_parameter_xyz = 42\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                TrackerConfig.from_properties(tmp_path)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
