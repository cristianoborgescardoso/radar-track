"""
Savitzky-Golay Local Polynomial Smoothing Filter.

Used by Pipeline 04 (``savgol``) and Pipeline 09 (``hampel_savgol``).
"""

from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter

# Odd 7-scan window ([t-3, t+3]) providing 4 degrees of freedom while shorter than 8 s straight legs.
SAVGOL_WINDOW: int = 7

# Quadratic polynomial order (degree 2) preserving local parabolic velocity trends (Savitzky & Golay, 1964).
SAVGOL_POLYORDER: int = 2


def apply_savgol_filter(
    vx: np.ndarray,
    vy: np.ndarray,
    window: int = SAVGOL_WINDOW,
    polyorder: int = SAVGOL_POLYORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a local polynomial Savitzky-Golay filter to East and North velocity vectors."""
    win = min(window, len(vx))
    if win % 2 == 0:
        win = max(1, win - 1)
    if win > polyorder:
        vx_smooth = savgol_filter(vx, window_length=win, polyorder=polyorder, mode="interp")
        vy_smooth = savgol_filter(vy, window_length=win, polyorder=polyorder, mode="interp")
        return vx_smooth, vy_smooth
    return vx.copy(), vy.copy()
