"""
Single-target 3D Kalman Filter track entity using FilterPy and SciPy.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from filterpy.common import Q_continuous_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.spatial.distance import mahalanobis

from radar_tracker.config import TrackerConfig


# Rejection sentinel returned when a candidate plot violates physical speed or Mahalanobis gates.
GATE_REJECT_COST: float = 1e9

# Scales 1st-to-2nd hit displacement (~20-30 m) by 0.5 so it matches steady-state innovation cost (~10-15).
INITIATION_COST_SCALE: float = 0.5

# Converts dimensionless Mahalanobis distance (sigma units) into equivalent meters (~3.0 m/sigma) in composite cost.
MAHALANOBIS_COST_WEIGHT: float = 3.0

# Lower floor (1 micrometer) preventing division-by-zero in ellipsoidal aspect ratio (sigma_xy / sigma_z).
MIN_SIGMA_DENOMINATOR: float = 1e-6

# Initial position variance (3.0 m)^2 on 1st detection before velocity is observed.
INIT_PRIOR_POS_VAR_M2: float = 9.0

# Initial horizontal velocity variance (25.0 m/s)^2 covering typical UAV cruise speeds (3-sigma = 75 m/s).
INIT_PRIOR_VEL_XY_VAR_MPS2: float = 625.0

# Initial vertical velocity variance (5.0 m/s)^2 covering typical climb/descent rates (4-sigma = 20 m/s).
INIT_PRIOR_VEL_Z_VAR_MPS2: float = 25.0

# Two-point differencing velocity variance multiplier (2 * sigma_pos^2 / dt^2, doubled for initial turn curvature).
TWO_POINT_VEL_VAR_FACTOR: float = 4.0

# Minimum vertical velocity variance (2.0 m/s)^2 after 2nd hit to avoid over-confidence on level segments.
MIN_TWO_POINT_VEL_Z_VAR_MPS2: float = 4.0

# Maximum number of initial points used for linear OLS velocity initialization in the 3D RTS smoother.
RTS_INIT_WINDOW_POINTS: int = 12

# Initial velocity variance (3.0 m/s)^2 after multi-point linear OLS regression in the 3D RTS smoother.
RTS_INIT_VEL_VAR_MPS2: float = 9.0

# Classic Huber M-estimator threshold (1.5 sigma, ~95% Gaussian efficiency) for innovation clipping.
RTS_HUBER_CLIP_SIGMA: float = 1.5

# Minimum positive time step floor (1 ms) ensuring non-singular F(dt) and Q(dt) matrices during RTS smoothing.
MIN_RTS_DT_S: float = 1e-3


class RadarTrack3D:
    """3D Constant-Velocity Track managed by ``filterpy.kalman.KalmanFilter``.

    Each track instance maintains its own dedicated Kalman Filter state, covariance,
    and a bounded Certainty Score in ``[0, 100]`` (+30 per detection, -10 per missed scan).

    State vector ordering (block-diagonal per axis in metric ENU coordinates):
        x = [east, v_east, north, v_north, up, v_up]^T
    """

    def __init__(
        self, internal_id: int, plot: Dict[str, Any], config: TrackerConfig = TrackerConfig()
    ) -> None:
        self.internal_id: int = internal_id
        self.config: TrackerConfig = config
        self.plots: List[Dict[str, Any]] = [plot]
        self.last_t: float = plot["t"]
        self.hits: int = 1
        self.missed_scans: int = 0
        self.mahalanobis_history: List[float] = [0.0]
        self.last_penalized_t: float = plot["t"]

        # Track Certainty Score lifecycle (starts at 30, +30 on hit, -10 on missed scan, capped at 100)
        self.certainty: float = config.initial_certainty
        self.max_certainty_reached: float = config.initial_certainty
        self.certainty_history: List[float] = [config.initial_certainty]

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.array([plot["x"], 0.0, plot["y"], 0.0, plot["z"], 0.0], dtype=float)
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        # Initial state covariance P_0 before velocity is observed (1st hit):
        self.kf.P = np.diag(
            [
                INIT_PRIOR_POS_VAR_M2,
                INIT_PRIOR_VEL_XY_VAR_MPS2,
                INIT_PRIOR_POS_VAR_M2,
                INIT_PRIOR_VEL_XY_VAR_MPS2,
                INIT_PRIOR_POS_VAR_M2,
                INIT_PRIOR_VEL_Z_VAR_MPS2,
            ]
        )
        self.state_history = [self.kf.x.copy()]
        # Measurement uncertainty covariance R (supports sphere when sigma_xy == sigma_z, or 3D ellipsoid)
        self.kf.R = np.diag(
            [
                config.sigma_pos_xy_m**2,
                config.sigma_pos_xy_m**2,
                config.sigma_pos_z_m**2,
            ]
        )

    @property
    def is_confirmed(self) -> bool:
        """Return True if the track reached the confirmation certainty threshold and minimum hits."""
        return (
            self.max_certainty_reached >= self.config.confirmation_certainty_threshold
            and self.hits >= self.config.min_confirmed_hits
        )

    def _transition_and_noise(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build 3D state transition F(dt) and process covariance Q(dt) via FilterPy."""
        f_1d = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        f_mat = block_diag(f_1d, f_1d, f_1d)
        q_xy = Q_continuous_white_noise(
            dim=2, dt=dt, spectral_density=self.config.sigma_accel_xy**2
        )
        q_z = Q_continuous_white_noise(
            dim=2, dt=dt, spectral_density=self.config.sigma_accel_z**2
        )
        q_mat = block_diag(q_xy, q_xy, q_z)
        return f_mat, q_mat

    def evaluate_candidate(self, plot: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Gate candidate plot using physical speed bounds and 3D Mahalanobis distance."""
        dt = plot["t"] - self.last_t
        if dt <= self.config.scan_window_s or dt > self.config.max_coast_s:
            return False, GATE_REJECT_COST, GATE_REJECT_COST

        dx = plot["x"] - self.plots[-1]["x"]
        dy = plot["y"] - self.plots[-1]["y"]
        dz = plot["z"] - self.plots[-1]["z"]
        spd_xy = math.hypot(dx, dy) / dt
        spd_z = abs(dz) / dt

        if spd_xy > self.config.max_speed_xy_mps or spd_z > self.config.max_speed_z_mps:
            return False, GATE_REJECT_COST, GATE_REJECT_COST

        f_mat, q_mat = self._transition_and_noise(dt)
        x_pred = f_mat @ self.kf.x
        p_pred = f_mat @ self.kf.P @ f_mat.T + q_mat

        z_meas = np.array([plot["x"], plot["y"], plot["z"]], dtype=float)
        z_pred = self.kf.H @ x_pred
        s_mat = self.kf.H @ p_pred @ self.kf.H.T + self.kf.R
        s_inv = np.linalg.inv(s_mat)

        mahal = float(mahalanobis(z_meas, z_pred, s_inv))
        # Ellipsoidal metric error: scales vertical residual by (sigma_xy / sigma_z) so that
        # when sigma_xy == sigma_z it equals 3D Euclidean norm, and when sigma_z > sigma_xy
        # (coarse elevation beam) vertical jitter does not artificially inflate horizontal gate cost.
        err_vec = z_meas - z_pred
        z_scale = self.config.sigma_pos_xy_m / max(MIN_SIGMA_DENOMINATOR, self.config.sigma_pos_z_m)
        euclid_err = float(math.hypot(err_vec[0], err_vec[1], err_vec[2] * z_scale))

        if self.hits == 1:
            if spd_xy > self.config.init_speed_xy_mps or spd_z > self.config.init_speed_z_mps:
                return False, GATE_REJECT_COST, GATE_REJECT_COST
            # Scale 1st-to-2nd hit distance so displacement (~v*dt) has comparable magnitude to innovation cost
            return True, euclid_err * INITIATION_COST_SCALE, mahal

        max_pred_err = self.config.base_pred_err_m + self.config.pred_err_slope_mps * max(
            0.0, dt - self.config.nominal_scan_period_s
        )
        if euclid_err > max_pred_err or mahal > self.config.mahalanobis_gate:
            return False, GATE_REJECT_COST, GATE_REJECT_COST
        # Composite association cost: metric residual (m) + weighted dimensionless Mahalanobis distance
        return True, euclid_err + MAHALANOBIS_COST_WEIGHT * mahal, mahal

    def update(self, plot: Dict[str, Any], mahal: float) -> None:
        """Advance Kalman state with associated radar plot and increment certainty (+30, cap 100)."""
        dt = plot["t"] - self.last_t
        if self.hits == 1:
            # Bayesian-regularized two-point differencing initialization (Bar-Shalom & Li, 1995):
            # Raw two-point variance R_v = TWO_POINT_VEL_VAR_FACTOR * sigma_pos^2 / dt^2 explodes
            # at high radar cadence (e.g. dt = 0.256 s at 234 RPM -> R_v = 13,733 m^2/s^2 >> prior 625).
            # Wiener shrinkage w = P_v0 / (P_v0 + R_v) damps short-dt azimuth jitter while preserving
            # full responsiveness (w -> 1) when R_v << P_v0 (e.g. w = 0.975 at 1 Hz, sigma_pos = 2 m).
            dt_sq = max(MIN_RTS_DT_S, dt) ** 2
            s_xy2 = self.config.sigma_pos_xy_m**2
            s_z2 = self.config.sigma_pos_z_m**2
            raw_vel_xy_var = (TWO_POINT_VEL_VAR_FACTOR * s_xy2) / dt_sq
            raw_vel_z_var = max(MIN_TWO_POINT_VEL_Z_VAR_MPS2, s_z2 / dt_sq)
            if raw_vel_xy_var > INIT_PRIOR_VEL_XY_VAR_MPS2:
                w_xy = INIT_PRIOR_VEL_XY_VAR_MPS2 / (INIT_PRIOR_VEL_XY_VAR_MPS2 + raw_vel_xy_var)
                vel_xy_var = w_xy * raw_vel_xy_var
            else:
                w_xy = 1.0
                vel_xy_var = raw_vel_xy_var
            if raw_vel_z_var > INIT_PRIOR_VEL_Z_VAR_MPS2:
                w_z = INIT_PRIOR_VEL_Z_VAR_MPS2 / (INIT_PRIOR_VEL_Z_VAR_MPS2 + raw_vel_z_var)
                vel_z_var = max(MIN_TWO_POINT_VEL_Z_VAR_MPS2, w_z * raw_vel_z_var)
            else:
                w_z = 1.0
                vel_z_var = raw_vel_z_var
            vx = w_xy * ((plot["x"] - self.plots[-1]["x"]) / dt)
            vy = w_xy * ((plot["y"] - self.plots[-1]["y"]) / dt)
            vz = w_z * ((plot["z"] - self.plots[-1]["z"]) / dt)
            self.kf.x = np.array([plot["x"], vx, plot["y"], vy, plot["z"], vz], dtype=float)
            self.kf.P = np.diag(
                [
                    s_xy2,
                    vel_xy_var,
                    s_xy2,
                    vel_xy_var,
                    s_z2,
                    vel_z_var,
                ]
            )
        else:
            f_mat, q_mat = self._transition_and_noise(dt)
            self.kf.predict(F=f_mat, Q=q_mat)
            self.kf.update(np.array([plot["x"], plot["y"], plot["z"]], dtype=float))

        self.state_history.append(self.kf.x.copy())
        self.last_t = plot["t"]
        self.last_penalized_t = plot["t"]
        self.plots.append(plot)
        self.hits += 1
        self.mahalanobis_history.append(mahal)

        self.certainty = min(
            self.config.max_certainty,
            self.certainty + self.config.certainty_hit_increment,
        )
        self.max_certainty_reached = max(self.max_certainty_reached, self.certainty)
        self.certainty_history.append(self.certainty)

    def penalize_missed_scan(self) -> None:
        """Penalize track certainty (-10) when a radar scan passes without associated detection."""
        self.missed_scans += 1
        self.certainty = max(0.0, self.certainty - self.config.certainty_miss_penalty)

    def smooth_states_rts(self) -> np.ndarray:
        """Compute 3D Huber-Kalman + Rauch-Tung-Striebel (RTS) smoothed states ``[x, vx, y, vy, z, vz]``."""
        n = len(self.plots)
        if n == 0:
            return np.zeros((0, 6), dtype=float)
        if n == 1:
            return np.array(self.state_history, dtype=float)

        ts = np.array([p["t"] for p in self.plots], dtype=float)
        xs = np.array([p["x"] for p in self.plots], dtype=float)
        ys = np.array([p["y"] for p in self.plots], dtype=float)
        zs = np.array([p["z"] for p in self.plots], dtype=float)

        k_init = min(RTS_INIT_WINDOW_POINTS, n)
        dt_span = ts[:k_init] - ts[0]
        if k_init >= 2 and float(dt_span[-1]) > MIN_RTS_DT_S:
            vx0 = float(np.polyfit(dt_span, xs[:k_init], 1)[0])
            vy0 = float(np.polyfit(dt_span, ys[:k_init], 1)[0])
            vz0 = float(np.polyfit(dt_span, zs[:k_init], 1)[0])
        else:
            vx0, vy0, vz0 = 0.0, 0.0, 0.0

        s_xy2 = self.config.sigma_pos_xy_m**2
        s_z2 = self.config.sigma_pos_z_m**2
        clip_xy = RTS_HUBER_CLIP_SIGMA * self.config.sigma_pos_xy_m
        clip_z = RTS_HUBER_CLIP_SIGMA * self.config.sigma_pos_z_m

        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([xs[0], vx0, ys[0], vy0, zs[0], vz0], dtype=float)
        kf.H = self.kf.H.copy()
        kf.P = np.diag(
            [s_xy2, RTS_INIT_VEL_VAR_MPS2, s_xy2, RTS_INIT_VEL_VAR_MPS2, s_z2, RTS_INIT_VEL_VAR_MPS2]
        )
        kf.R = np.diag([s_xy2, s_xy2, s_z2])
        kf.update(np.array([xs[0], ys[0], zs[0]], dtype=float))

        xs_f: List[np.ndarray] = [kf.x.copy()]
        ps_f: List[np.ndarray] = [kf.P.copy()]
        fs_l: List[np.ndarray] = [np.eye(6)]
        qs_l: List[np.ndarray] = [np.zeros((6, 6))]

        for k in range(1, n):
            dt = max(MIN_RTS_DT_S, float(ts[k] - ts[k - 1]))
            f_mat, q_mat = self._transition_and_noise(dt)
            kf.predict(F=f_mat, Q=q_mat)
            z = np.array([xs[k], ys[k], zs[k]], dtype=float)
            z_pred = kf.H @ kf.x
            innov = z - z_pred
            norm_xy = float(np.hypot(innov[0], innov[1]))
            if norm_xy > clip_xy:
                z[0] = z_pred[0] + innov[0] * (clip_xy / norm_xy)
                z[1] = z_pred[1] + innov[1] * (clip_xy / norm_xy)
            if abs(float(innov[2])) > clip_z:
                z[2] = z_pred[2] + np.sign(innov[2]) * clip_z
            kf.update(z)
            xs_f.append(kf.x.copy())
            ps_f.append(kf.P.copy())
            fs_l.append(f_mat)
            qs_l.append(q_mat)

        xs_rts, _, _, _ = kf.rts_smoother(
            Xs=np.array(xs_f), Ps=np.array(ps_f), Fs=fs_l, Qs=qs_l
        )
        return np.asarray(xs_rts, dtype=float)
