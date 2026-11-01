"""
Unit tests for ``radar_tracker/kinematics/`` (all 11 kinematic filtering pipelines and circular statistics).
"""

from __future__ import annotations

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from radar_tracker import angular_diff_deg, estimate_track_kinematics
from radar_tracker.kinematics import (
    apply_alpha_beta_filter,
    apply_directional_smoothing,
    apply_kalman_rts_positions,
    apply_moving_average,
    apply_savgol_filter,
    compute_raw_velocity_vectors,
    reject_waypoint_outliers,
    vectors_to_heading_deg,
)


class TestKinematicsModulesUnit(unittest.TestCase):
    """Unit tests for every kinematic filter and helper module in radar_tracker/kinematics/."""

    def test_angular_diff_and_vectors_to_heading(self) -> None:
        """Verify circular angular difference and compass bearing conversion (0° = North, 90° = East)."""
        self.assertAlmostEqual(angular_diff_deg(359.0, 1.0), 2.0, places=5)
        self.assertAlmostEqual(angular_diff_deg(1.0, 359.0), 2.0, places=5)
        self.assertAlmostEqual(angular_diff_deg(90.0, 270.0), 180.0, places=5)

        vx = np.array([0.0, 10.0, 0.0, -10.0])
        vy = np.array([10.0, 0.0, -10.0, 0.0])
        hdgs = vectors_to_heading_deg(vx, vy)
        np.testing.assert_allclose(hdgs, [0.0, 90.0, 180.0, 270.0], atol=1e-5)

    def test_compute_raw_velocity_vectors(self) -> None:
        """Verify step-by-step finite-difference velocity vectors on synthetic trajectory."""
        ts = np.array([0.0, 1.0, 3.0])
        xs = np.array([0.0, 15.0, 45.0])
        ys = np.array([0.0, 20.0, 60.0])
        dt, vx, vy = compute_raw_velocity_vectors(ts, xs, ys)
        np.testing.assert_allclose(dt, [1.0, 2.0])
        np.testing.assert_allclose(vx, [15.0, 15.0])
        np.testing.assert_allclose(vy, [20.0, 20.0])

    def test_alpha_beta_and_moving_average_and_savgol_filters(self) -> None:
        """Verify alpha-beta, moving average, and Savitzky-Golay filters reduce velocity noise."""
        rng = np.random.default_rng(42)
        ts = np.arange(26, dtype=float)
        xs = 20.0 * ts + rng.normal(0.0, 2.0, size=26)
        ys = 10.0 * ts + rng.normal(0.0, 2.0, size=26)
        _, vx_noisy, vy_noisy = compute_raw_velocity_vectors(ts, xs, ys)

        vx_ab, vy_ab = apply_alpha_beta_filter(ts, xs, ys, alpha=0.35, beta=0.08)
        vx_ma, vy_ma = apply_moving_average(vx_noisy, vy_noisy, window=5)
        vx_sg, vy_sg = apply_savgol_filter(vx_noisy, vy_noisy, window=7, polyorder=2)

        self.assertLess(float(np.std(vx_ab)), float(np.std(vx_noisy)))
        self.assertLess(float(np.std(vx_ma)), float(np.std(vx_noisy)))
        self.assertLess(float(np.std(vx_sg)), float(np.std(vx_noisy)))

        # Short array edge case for Savitzky-Golay (< 3 elements)
        vx_short, vy_short = apply_savgol_filter(np.array([10.0, 12.0]), np.array([5.0, 6.0]))
        self.assertEqual(len(vx_short), 2)

    def test_kalman_rts_positions_with_and_without_huber(self) -> None:
        """Verify 2D Kalman RTS smoother reduces position/velocity noise and clips outliers with Huber."""
        n = 30
        ts = np.arange(n, dtype=float)
        xs_true = 15.0 * ts
        ys_true = 10.0 * ts
        xs_obs = xs_true.copy()
        ys_obs = ys_true.copy()
        # Inject a 45m radar spike at index 15
        xs_obs[15] += 45.0

        vx_std, vy_std = apply_kalman_rts_positions(
            ts, xs_obs, ys_obs, pos_std_m=5.0, accel_spectral_density=0.5, huber_clip_m=None
        )
        vx_hub, vy_hub = apply_kalman_rts_positions(
            ts, xs_obs, ys_obs, pos_std_m=6.0, accel_spectral_density=0.5, huber_clip_m=10.0
        )
        # Huber-clipped RTS must have smaller velocity spike around index 15 than unclipped RTS
        self.assertLess(float(np.max(np.abs(vx_hub - 15.0))), float(np.max(np.abs(vx_std - 15.0))))

    def test_hampel_outlier_rejection_and_directional_smoothing(self) -> None:
        """Verify Hampel filter removes a 1-step waypoint speed drop while directional smoothing preserves 90° turns."""
        vx = np.array([20.0] * 10 + [0.0] + [20.0] * 10)
        vy = np.zeros(21)
        vx_clean, vy_clean = reject_waypoint_outliers(vx, vy, median_window_size=5, speed_outlier_rel_tol=0.35)
        self.assertAlmostEqual(float(vx_clean[10]), 20.0, places=4)

        # 90-degree sharp turn: 10 steps East (vx=20, vy=0) -> 10 steps North (vx=0, vy=20)
        vx_turn = np.array([20.0] * 10 + [0.0] * 10)
        vy_turn = np.array([0.0] * 10 + [20.0] * 10)
        vx_dir, vy_dir = apply_directional_smoothing(
            vx_turn, vy_turn, smooth_neighbor_radius=2, max_heading_diff_deg=25.0
        )
        # Across the 90-degree turn boundary (indices 9 and 10), directional smoothing must NOT blend East and North
        self.assertAlmostEqual(float(vx_dir[9]), 20.0, places=5)
        self.assertAlmostEqual(float(vy_dir[9]), 0.0, places=5)
        self.assertAlmostEqual(float(vx_dir[10]), 0.0, places=5)
        self.assertAlmostEqual(float(vy_dir[10]), 20.0, places=5)

    def test_estimate_track_kinematics_edge_cases_and_invalid_method(self) -> None:
        """Verify estimate_track_kinematics on 0, 1, and N points and ValueError on unknown pipeline."""
        self.assertEqual(estimate_track_kinematics([]), ([], [], 0.0, 0.0))
        self.assertEqual(
            estimate_track_kinematics([{"t": 0.0, "x": 0.0, "y": 0.0}]),
            ([None], [None], 0.0, 0.0),
        )
        pts = [
            {"t": 0.0, "x": 0.0, "y": 0.0},
            {"t": 1.0, "x": 10.0, "y": 0.0},
            {"t": 2.0, "x": 20.0, "y": 0.0},
        ]
        with self.assertRaises(ValueError):
            estimate_track_kinematics(pts, method="non_existent_filter")

    def test_circular_heading_near_north(self) -> None:
        """Verify circular average heading for a target moving North across 359° <-> 1°."""
        pts = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "z": 100.0},
            {"t": 1.0, "x": -0.35, "y": 20.0, "z": 100.0},
            {"t": 2.0, "x": 0.0, "y": 40.0, "z": 100.0},
            {"t": 3.0, "x": -0.35, "y": 60.0, "z": 100.0},
            {"t": 4.0, "x": 0.0, "y": 80.0, "z": 100.0},
        ]
        speeds, headings, avg_speed, avg_heading = estimate_track_kinematics(pts)
        self.assertIsNone(speeds[0])
        self.assertIsNone(headings[0])
        self.assertAlmostEqual(avg_speed, 20.0, delta=0.2)
        self.assertLess(angular_diff_deg(avg_heading, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
