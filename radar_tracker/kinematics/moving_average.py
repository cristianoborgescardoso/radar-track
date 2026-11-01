"""
Uniform Sliding-Window Moving Average Filter (Boxcar Filter).

Used by Pipeline 02 (``moving_average``) and Pipeline 08 (``hampel_moving_avg``).
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

# Symmetric 5-scan boxcar window ([t-2, t+2]); reduces noise by sqrt(5) while shorter than 8 s straight legs.
MOVING_AVG_WINDOW: int = 5


def apply_moving_average(
    vx: np.ndarray,
    vy: np.ndarray,
    window: int = MOVING_AVG_WINDOW,
    mode: str = "nearest",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a 1D uniform sliding-window average independently to East and North velocity vectors."""
    win = min(window, len(vx))
    vx_smooth = uniform_filter1d(vx, size=win, mode=mode)
    vy_smooth = uniform_filter1d(vy, size=win, mode=mode)
    return vx_smooth, vy_smooth
