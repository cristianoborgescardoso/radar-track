"""
Stage 2 Bilateral Direction-Compatible Vector Smoothing Filter.

Used by Pipeline 11 (``hampel_directional``, the default recommended pipeline).
"""

from typing import Tuple

import numpy as np

from radar_tracker.kinematics.kinematics_base import angular_diff_deg, vectors_to_heading_deg

# Neighbor radius of 1 scan (3-point window [i-1, i, i+1]), reducing white velocity variance by 3x without turn lag.
SMOOTH_NEIGHBOR_RADIUS: int = 1

# Bilateral directional gate (12.0 deg = 2-sigma angular radar noise), sitting between straight jitter (<=7.4 deg) and turns (>=18 deg).
MAX_COMPATIBLE_HEADING_DIFF_DEG: float = 12.0


def apply_directional_smoothing(
    vx: np.ndarray,
    vy: np.ndarray,
    smooth_neighbor_radius: int = SMOOTH_NEIGHBOR_RADIUS,
    max_heading_diff_deg: float = MAX_COMPATIBLE_HEADING_DIFF_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average velocity vectors within a local window only across direction-compatible steps."""
    num_steps = len(vx)
    vx_smooth, vy_smooth = vx.copy(), vy.copy()
    hdg = vectors_to_heading_deg(vx, vy)
    for i in range(num_steps):
        neighbor_window = range(
            max(0, i - smooth_neighbor_radius),
            min(num_steps, i + smooth_neighbor_radius + 1),
        )
        compatible_indices = [
            k
            for k in neighbor_window
            if angular_diff_deg(float(hdg[k]), float(hdg[i])) < max_heading_diff_deg
        ]
        vx_smooth[i] = float(np.mean(vx[compatible_indices]))
        vy_smooth[i] = float(np.mean(vy[compatible_indices]))
    return vx_smooth, vy_smooth
