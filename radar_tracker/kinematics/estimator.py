"""
Kinematic Estimator (Default Pipeline: ``hampel_directional``).

When running ``./tracker.sh plots.json tracks.json``, the selected pipeline
(by default ``hampel_directional``: Stage 1 Hampel outlier rejection + Stage 2 Bilateral
Directional smoothing) is executed for each confirmed track. Alternative pipelines are only
loaded on demand if explicitly requested via ``method``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import circmean

from radar_tracker.kinematics.directional import (
    MAX_COMPATIBLE_HEADING_DIFF_DEG,
    SMOOTH_NEIGHBOR_RADIUS,
    apply_directional_smoothing,
)
from radar_tracker.kinematics.hampel import (
    MEDIAN_WINDOW_SIZE,
    SPEED_OUTLIER_REL_TOL,
    reject_waypoint_outliers,
)
from radar_tracker.kinematics.kinematics_base import (
    FULL_CIRCLE_DEG,
    OUTPUT_DECIMALS,
    compute_raw_velocity_vectors,
    vectors_to_heading_deg,
)

DEFAULT_PIPELINE_METHOD: str = "hampel_directional"


def estimate_track_kinematics(
    pts: List[Dict[str, Any]],
    method: str = DEFAULT_PIPELINE_METHOD,
    median_window_size: int = MEDIAN_WINDOW_SIZE,
    speed_outlier_rel_tol: float = SPEED_OUTLIER_REL_TOL,
    smooth_neighbor_radius: int = SMOOTH_NEIGHBOR_RADIUS,
    max_heading_diff_deg: float = MAX_COMPATIBLE_HEADING_DIFF_DEG,
    output_decimals: int = OUTPUT_DECIMALS,
    pos_std_m: Optional[float] = None,
    accel_spectral_density: Optional[float] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], float, float]:
    """Estimate per-point ground speed (m/s), compass heading (deg), and track averages.

    By default, executes the selected ``hampel_directional`` pipeline.
    If an alternative ``method`` is specified (e.g., via ``tracker.properties`` or during
    benchmark comparisons), dispatches to that specific filter without running the others.
    """
    n = len(pts)
    if n == 0:
        return [], [], 0.0, 0.0
    if n == 1:
        return [None], [None], 0.0, 0.0

    ts = np.array([p["t"] for p in pts], dtype=float)
    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)

    dt, vx_raw, vy_raw = compute_raw_velocity_vectors(ts, xs, ys)

    # Execute the selected pipeline (hampel_directional)
    if method == DEFAULT_PIPELINE_METHOD:
        vx_clean, vy_clean = reject_waypoint_outliers(
            vx_raw, vy_raw, median_window_size, speed_outlier_rel_tol
        )
        vx_final, vy_final = apply_directional_smoothing(
            vx_clean, vy_clean, smooth_neighbor_radius, max_heading_diff_deg
        )
    else:
        # On-demand dispatch for alternative pipelines (used when overridden or benchmarked)
        from radar_tracker.kinematics.pipeline_evaluator import run_alternative_pipeline

        vx_final, vy_final = run_alternative_pipeline(
            method=method,
            ts=ts,
            xs=xs,
            ys=ys,
            dt=dt,
            vx_raw=vx_raw,
            vy_raw=vy_raw,
            median_window_size=median_window_size,
            speed_outlier_rel_tol=speed_outlier_rel_tol,
            pos_std_m=pos_std_m,
            accel_spectral_density=accel_spectral_density,
        )

    spd_final = np.hypot(vx_final, vy_final)
    hdg_final = vectors_to_heading_deg(vx_final, vy_final)

    speeds: List[Optional[float]] = [None] + [
        round(float(s), output_decimals) for s in spd_final
    ]
    headings: List[Optional[float]] = [None] + [
        round(float(h), output_decimals) for h in hdg_final
    ]

    avg_speed = round(float(np.mean(spd_final)), output_decimals)
    avg_heading = round(
        float(circmean(hdg_final, high=FULL_CIRCLE_DEG, low=0.0)),
        output_decimals,
    )
    return speeds, headings, avg_speed, avg_heading
