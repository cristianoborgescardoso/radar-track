"""
Classic Recursive Alpha-Beta Radar Tracking Filter.

Used by Pipeline 03 (``alpha_beta``).
"""

from typing import Tuple

import numpy as np

from radar_tracker.kinematics.kinematics_base import MIN_TIME_STEP_S

# Position smoothing gain for tracking index Lambda ~ 0.35 (Kalata, IEEE TAES 1984).
ALPHA_BETA_ALPHA: float = 0.45

# Critically damped velocity gain beta = alpha^2 / (2 - alpha) = 0.45^2 / 1.55 ~ 0.12 (Kalata, 1984).
ALPHA_BETA_BETA: float = 0.12


def apply_alpha_beta_filter(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    alpha: float = ALPHA_BETA_ALPHA,
    beta: float = ALPHA_BETA_BETA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a classic 2D Alpha-Beta radar tracking filter over position measurements."""
    n = len(ts)
    dt0 = max(MIN_TIME_STEP_S, float(ts[1] - ts[0]))
    x_est, y_est = float(xs[0]), float(ys[0])
    vx_est = float((xs[1] - xs[0]) / dt0)
    vy_est = float((ys[1] - ys[0]) / dt0)

    vx_out = np.zeros(n - 1, dtype=float)
    vy_out = np.zeros(n - 1, dtype=float)

    for k in range(1, n):
        dt = max(MIN_TIME_STEP_S, float(ts[k] - ts[k - 1]))
        x_pred = x_est + vx_est * dt
        y_pred = y_est + vy_est * dt

        rx = float(xs[k]) - x_pred
        ry = float(ys[k]) - y_pred

        x_est = x_pred + alpha * rx
        y_est = y_pred + alpha * ry
        vx_est = vx_est + (beta / dt) * rx
        vy_est = vy_est + (beta / dt) * ry

        vx_out[k - 1] = vx_est
        vy_out[k - 1] = vy_est

    return vx_out, vy_out
