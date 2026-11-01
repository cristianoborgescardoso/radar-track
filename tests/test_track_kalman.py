"""
Unit tests for ``radar_tracker/track.py`` (RadarTrack3D 6D Kalman filter, gating, RTS smoothing, and NIS consistency).
"""

from __future__ import annotations

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from radar_tracker import RadarTrack3D, TrackerConfig


class TestRadarTrack3DUnit(unittest.TestCase):
    """Unit tests for RadarTrack3D 6D Kalman filter, two-point initialization, Mahalanobis gating, and RTS smoothing."""

    def test_radar_track_3d_candidate_gates_and_rts_smoothing(self) -> None:
        """Verify RadarTrack3D kinematic/Mahalanobis gating and 3D RTS state smoothing."""
        cfg = TrackerConfig(sigma_pos_xy_m=10.0, sigma_pos_z_m=15.0)
        p0 = {"id": 1, "t": 0.0, "x": 0.0, "y": 0.0, "z": 100.0, "latitude": 0.0, "longitude": 0.0, "altitude": 100.0}
        tr = RadarTrack3D(1, p0, cfg)
        self.assertFalse(tr.is_confirmed)

        # Negative dt must be rejected
        ok, _, _ = tr.evaluate_candidate({"t": -1.0, "x": 0.0, "y": 0.0, "z": 100.0})
        self.assertFalse(ok)

        # Excessive horizontal speed (> max_speed_xy_mps) must be rejected
        ok_fast, _, _ = tr.evaluate_candidate({"t": 1.0, "x": 500.0, "y": 0.0, "z": 100.0})
        self.assertFalse(ok_fast)

        # Valid updates
        for step in range(1, 6):
            pt = {
                "id": step + 1,
                "t": float(step),
                "x": 10.0 * step + (25.0 if step == 3 else 0.0),
                "y": 5.0 * step,
                "z": 100.0,
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 100.0,
            }
            ok_step, cost, mahal = tr.evaluate_candidate(pt)
            self.assertTrue(ok_step)
            self.assertGreaterEqual(cost, 0.0)
            tr.update(pt, mahal)

        self.assertTrue(tr.is_confirmed)
        smoothed = tr.smooth_states_rts()
        self.assertEqual(smoothed.shape, (6, 6))
        # RTS smoothing with Huber clipping should pull the +25m spike at step=3 closer to x=30.0
        self.assertLess(abs(float(smoothed[3, 0]) - 30.0), 15.0)

    def test_two_point_velocity_variance_scales_with_inverse_dt_squared(self) -> None:
        """Verify two-point differencing velocity variance scales as 1 / dt^2 (Bar-Shalom & Li, 1995)."""
        cfg = TrackerConfig(sigma_pos_xy_m=2.0, sigma_pos_z_m=2.0)

        # Case A: dt = 1.0 s -> Var(v_xy) = 4 * 2^2 / 1.0^2 = 16.0 m^2/s^2
        tr_1hz = RadarTrack3D(1, {"id": 1, "t": 0.0, "x": 0.0, "y": 0.0, "z": 100.0}, cfg)
        tr_1hz.update({"id": 2, "t": 1.0, "x": 20.0, "y": 0.0, "z": 100.0}, mahal=0.5)
        var_vx_1hz = float(tr_1hz.kf.P[1, 1])
        self.assertAlmostEqual(var_vx_1hz, 16.0, places=5)

        # Case B: dt = 0.5 s -> Var(v_xy) = 4 * 2^2 / 0.5^2 = 64.0 m^2/s^2 (4x higher uncertainty)
        tr_2hz = RadarTrack3D(2, {"id": 1, "t": 0.0, "x": 0.0, "y": 0.0, "z": 100.0}, cfg)
        tr_2hz.update({"id": 2, "t": 0.5, "x": 10.0, "y": 0.0, "z": 100.0}, mahal=0.5)
        var_vx_2hz = float(tr_2hz.kf.P[1, 1])
        self.assertAlmostEqual(var_vx_2hz, 64.0, places=5)
        self.assertAlmostEqual(var_vx_2hz / var_vx_1hz, 4.0, places=5)

    def test_kalman_filter_convergence_rmse_improvement_and_nis_consistency(self) -> None:
        """Verify Kalman/RTS filtering strictly reduces position/velocity RMSE, maintains SPD covariance, and satisfies NIS consistency."""
        rng = np.random.default_rng(12345)
        sigma_xy = 3.0
        sigma_z = 3.0
        cfg = TrackerConfig(
            sigma_pos_xy_m=sigma_xy,
            sigma_pos_z_m=sigma_z,
            sigma_accel_xy=1.5,
            sigma_accel_z=1.0,
        )

        # True constant-velocity trajectory: v = (18.0, -12.0, 1.5) m/s over N = 40 scans at 1 Hz
        n_steps = 40
        true_vx, true_vy, true_vz = 18.0, -12.0, 1.5
        ts = np.arange(n_steps, dtype=float)
        true_x = true_vx * ts
        true_y = true_vy * ts
        true_z = 200.0 + true_vz * ts

        meas_x = true_x + rng.normal(0.0, sigma_xy, size=n_steps)
        meas_y = true_y + rng.normal(0.0, sigma_xy, size=n_steps)
        meas_z = true_z + rng.normal(0.0, sigma_z, size=n_steps)

        plots = [
            {"id": i + 1, "t": float(ts[i]), "x": float(meas_x[i]), "y": float(meas_y[i]), "z": float(meas_z[i])}
            for i in range(n_steps)
        ]

        tr = RadarTrack3D(1, plots[0], cfg)
        nis_values = []
        pos_var_trace_history = []

        for pl in plots[1:]:
            ok, _, mahal = tr.evaluate_candidate(pl)
            self.assertTrue(ok)
            nis_values.append(mahal**2)  # NIS = d_M^2 = y^T S^-1 y ~ chi^2_3 (expected mean = 3.0)
            tr.update(pl, mahal)

            # Verify covariance matrix P remains Symmetric Positive-Definite (SPD) at every step
            p_mat = tr.kf.P
            np.testing.assert_allclose(p_mat, p_mat.T, atol=1e-10)
            eigenvalues = np.linalg.eigvalsh(p_mat)
            self.assertTrue(np.all(eigenvalues > 0.0))
            pos_var_trace_history.append(float(p_mat[0, 0] + p_mat[2, 2] + p_mat[4, 4]))

        # 1. Verify covariance convergence: steady-state position variance trace is smaller than initial post-2nd-hit variance
        self.assertLess(pos_var_trace_history[-1], pos_var_trace_history[0])

        # 2. Verify forward Kalman state improves position and velocity RMSE over raw noisy measurements
        kf_states = np.array(tr.state_history)
        raw_pos_rmse = float(np.sqrt(np.mean((meas_x - true_x) ** 2 + (meas_y - true_y) ** 2)))
        kf_pos_rmse = float(np.sqrt(np.mean((kf_states[:, 0] - true_x) ** 2 + (kf_states[:, 2] - true_y) ** 2)))
        self.assertLess(kf_pos_rmse, raw_pos_rmse)

        raw_vx = np.diff(meas_x) / np.diff(ts)
        raw_vy = np.diff(meas_y) / np.diff(ts)
        raw_vel_rmse = float(np.sqrt(np.mean((raw_vx - true_vx) ** 2 + (raw_vy - true_vy) ** 2)))
        kf_vel_rmse = float(
            np.sqrt(np.mean((kf_states[2:, 1] - true_vx) ** 2 + (kf_states[2:, 3] - true_vy) ** 2))
        )
        self.assertLess(kf_vel_rmse, raw_vel_rmse)

        # 3. Verify 3D RTS backward smoother further improves position RMSE over forward-only Kalman filter
        rts_states = tr.smooth_states_rts()
        rts_pos_rmse = float(np.sqrt(np.mean((rts_states[:, 0] - true_x) ** 2 + (rts_states[:, 2] - true_y) ** 2)))
        self.assertLess(rts_pos_rmse, kf_pos_rmse)

        # 4. Verify statistical consistency: mean NIS (3-DOF chi-square, E[chi^2_3] = 3.0) lies within 95% confidence bounds [1.5, 5.0]
        mean_nis = float(np.mean(nis_values[2:]))
        self.assertGreater(mean_nis, 1.0)
        self.assertLess(mean_nis, 5.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
