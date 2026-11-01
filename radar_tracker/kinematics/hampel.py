"""
Stage 1 Non-Linear Outlier Rejection: Median & Hampel (MAD) Filter for waypoint clamping and radar glint.

Shared by pipelines 07 through 11 (``median_only``, ``hampel_moving_avg``, ``hampel_savgol``,
``hampel_kalman_rts``, and ``hampel_directional``).
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter

from radar_tracker.kinematics.kinematics_base import MIN_MOVING_SPEED_MPS

# Default 9-scan median window spanning 6-10 s inter-waypoint segments (breakdown point = 4 outliers).
MEDIAN_WINDOW_SIZE: int = 9

# Relative speed deviation threshold (12%) separating normal radar jitter (~10%) from waypoint drops (>=15%).
SPEED_OUTLIER_REL_TOL: float = 0.12

# Standard normal reciprocal 1 / Phi^-1(0.75) scaling MAD to Gaussian sigma (Hampel, JASA 1974).
HAMPEL_MAD_SCALE: float = 1.4826

# Hampel outlier cutoff in Gaussian standard deviations (2.5 sigma ~ 98.8% coverage).
HAMPEL_N_SIGMA: float = 2.5


def reject_waypoint_outliers(
    vx_raw: np.ndarray,
    vy_raw: np.ndarray,
    median_window_size: int = MEDIAN_WINDOW_SIZE,
    speed_outlier_rel_tol: float = SPEED_OUTLIER_REL_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reject isolated 1-step speed drops (waypoint clamping / radar glint) via median/Hampel rule.

    Uses ``mode='mirror'`` at boundaries so endpoint clamping artifacts are not duplicated.
    """
    spd_raw = np.hypot(vx_raw, vy_raw)
    effective_win = min(median_window_size, len(spd_raw))
    half_win = median_window_size // 2
    med_spd = median_filter(spd_raw, size=effective_win, mode="mirror")

    # Compute local MAD (Median Absolute Deviation) for Hampel criterion alongside relative tolerance
    abs_dev = np.abs(spd_raw - med_spd)
    mad_spd = HAMPEL_MAD_SCALE * median_filter(abs_dev, size=effective_win, mode="mirror")

    vx_clean, vy_clean = vx_raw.copy(), vy_raw.copy()
    num_steps = len(spd_raw)
    for i in range(num_steps):
        rel_threshold = speed_outlier_rel_tol * med_spd[i]
        hampel_threshold = max(rel_threshold, HAMPEL_N_SIGMA * mad_spd[i])
        if med_spd[i] > MIN_MOVING_SPEED_MPS and abs_dev[i] > min(rel_threshold, hampel_threshold):
            search_window = range(max(0, i - half_win), min(num_steps, i + half_win + 1))
            candidates = [
                k for k in search_window if abs(spd_raw[k] - med_spd[i]) <= rel_threshold
            ]
            if candidates:
                best_k = min(candidates, key=lambda k: (abs(k - i), -1 if k > i else 1))
                vx_clean[i], vy_clean[i] = vx_raw[best_k], vy_raw[best_k]
    return vx_clean, vy_clean
