"""
Rauch-Tung-Striebel (RTS) Forward-Backward Kalman Smoother and Huber Robust Kalman Smoother.

Used by Pipeline 05 (``kalman_rts``), Pipeline 06 (``huber_kalman_rts``), and Pipeline 10 (``hampel_kalman_rts``).
"""

from typing import List, Optional, Tuple

import numpy as np
from filterpy.common import Q_continuous_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

from radar_tracker.kinematics.kinematics_base import MIN_TIME_STEP_S

# Default horizontal position standard deviation (m) for the 2D RTS smoother (Rauch, Tung & Striebel, 1965).
RTS_POS_STD_M: float = 2.0

# Default continuous-time acceleration spectral density (m^2/s^3) for the 2D RTS smoother.
RTS_ACCEL_SPECTRAL_DENSITY: float = 2.5

# Default 3-sigma Huber innovation clipping bound (3 * RTS_POS_STD_M = 6.0 m; Huber, 1964).
HUBER_INNOVATION_CLIP_M: float = 6.0

# Position std threshold (m) above which multi-point linear OLS regression is used to initialize velocity.
HIGH_NOISE_POS_STD_THRESHOLD_M: float = 5.0

# Minimum track length required to fit an overdetermined linear OLS initial velocity trend.
OLS_INIT_MIN_POINTS: int = 4

# Maximum initial points used for linear OLS velocity regression.
OLS_INIT_MAX_POINTS: int = 12

# Initial velocity variance (3.0 m/s)^2 after multi-point linear OLS regression.
OLS_INIT_VEL_VAR_MPS2: float = 9.0

# Initial velocity variance (5.0 m/s)^2 for two-point finite-difference initialization.
TWO_POINT_INIT_VEL_VAR_MPS2: float = 25.0


def apply_kalman_rts_positions(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    pos_std_m: float = RTS_POS_STD_M,
    accel_spectral_density: float = RTS_ACCEL_SPECTRAL_DENSITY,
    huber_clip_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a 2D Constant-Velocity Kalman Filter + Rauch-Tung-Striebel (RTS) backward smoother."""
    n = len(ts)
    if pos_std_m > HIGH_NOISE_POS_STD_THRESHOLD_M and n >= OLS_INIT_MIN_POINTS:
        k_init = min(OLS_INIT_MAX_POINTS, n)
        dt_span = ts[:k_init] - ts[0]
        vx0 = float(np.polyfit(dt_span, xs[:k_init], 1)[0])
        vy0 = float(np.polyfit(dt_span, ys[:k_init], 1)[0])
        init_vel_var = OLS_INIT_VEL_VAR_MPS2
    else:
        dt0 = max(MIN_TIME_STEP_S, float(ts[1] - ts[0]))
        vx0 = float((xs[1] - xs[0]) / dt0)
        vy0 = float((ys[1] - ys[0]) / dt0)
        init_vel_var = TWO_POINT_INIT_VEL_VAR_MPS2

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([xs[0], vx0, ys[0], vy0], dtype=float)
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    kf.P = np.diag([pos_std_m**2, init_vel_var, pos_std_m**2, init_vel_var])
    kf.R = np.diag([pos_std_m**2, pos_std_m**2])

    xs_f: List[np.ndarray] = [kf.x.copy()]
    covs_f: List[np.ndarray] = [kf.P.copy()]
    fs_list: List[np.ndarray] = [np.eye(4)]
    qs_list: List[np.ndarray] = [np.zeros((4, 4))]

    for k in range(1, n):
        dt = max(MIN_TIME_STEP_S, float(ts[k] - ts[k - 1]))
        f_1d = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        f_mat = block_diag(f_1d, f_1d)
        q_1d = Q_continuous_white_noise(dim=2, dt=dt, spectral_density=accel_spectral_density)
        q_mat = block_diag(q_1d, q_1d)

        kf.predict(F=f_mat, Q=q_mat)
        z = np.array([xs[k], ys[k]], dtype=float)

        if huber_clip_m is not None:
            z_pred = kf.H @ kf.x
            innov = z - z_pred
            norm_innov = float(np.linalg.norm(innov))
            if norm_innov > huber_clip_m:
                z = z_pred + innov * (huber_clip_m / norm_innov)

        kf.update(z)
        xs_f.append(kf.x.copy())
        covs_f.append(kf.P.copy())
        fs_list.append(f_mat)
        qs_list.append(q_mat)

    xs_smooth, _, _, _ = kf.rts_smoother(
        Xs=np.array(xs_f), Ps=np.array(covs_f), Fs=fs_list, Qs=qs_list
    )
    return xs_smooth[1:, 1], xs_smooth[1:, 3]
