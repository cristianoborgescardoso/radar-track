"""
Shared constants, circular statistics, and raw finite-difference utilities for kinematic estimation.
"""

from typing import Tuple

import numpy as np

# Full compass circle in degrees (0=North, 90=East, 180=South, 270=West).
FULL_CIRCLE_DEG: float = 360.0

# Half compass circle in degrees for shortest-arc angular difference in [0, 180].
HALF_CIRCLE_DEG: float = 180.0

# Minimum positive time step (1 microsecond) preventing division-by-zero on identical timestamps.
MIN_TIME_STEP_S: float = 1e-6

# Minimum target speed (1 mm/s) avoiding relative-tolerance instability on stationary points.
MIN_MOVING_SPEED_MPS: float = 1e-3

# Default decimal rounding precision for exported speed (0.01 m/s) and heading (0.01 deg).
OUTPUT_DECIMALS: int = 2


def angular_diff_deg(a_deg: float, b_deg: float) -> float:
    """Compute shortest absolute angular difference in [0, 180] degrees across the 0/360 boundary."""
    return abs((a_deg - b_deg + HALF_CIRCLE_DEG) % FULL_CIRCLE_DEG - HALF_CIRCLE_DEG)


def vectors_to_heading_deg(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Convert East/North velocity components to compass heading in [0, 360) degrees."""
    return (np.degrees(np.arctan2(vx, vy)) + FULL_CIRCLE_DEG) % FULL_CIRCLE_DEG


def compute_raw_velocity_vectors(
    ts: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute step time deltas dt and raw finite-difference velocities (vx_raw, vy_raw)."""
    dt = np.maximum(MIN_TIME_STEP_S, np.diff(ts))
    vx_raw = np.diff(xs) / dt
    vy_raw = np.diff(ys) / dt
    return dt, vx_raw, vy_raw
