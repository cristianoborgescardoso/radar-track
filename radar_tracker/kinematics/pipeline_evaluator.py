"""
Multi-Pipeline Benchmark Evaluator (``pipeline_evaluator.py``).

Dedicated exclusively to comparing and benchmarking all 11 candidate kinematic filtering
pipelines during offline validation (Jupyter Notebook Activity 5 and integration tests).
Not invoked during normal CLI execution of ``./tracker.sh plots.json tracks.json``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from radar_tracker.kinematics.alpha_beta import apply_alpha_beta_filter
from radar_tracker.kinematics.estimator import estimate_track_kinematics
from radar_tracker.kinematics.hampel import reject_waypoint_outliers
from radar_tracker.kinematics.kalman_rts import (
    HIGH_NOISE_POS_STD_THRESHOLD_M,
    HUBER_INNOVATION_CLIP_M,
    RTS_ACCEL_SPECTRAL_DENSITY,
    RTS_POS_STD_M,
    apply_kalman_rts_positions,
)
from radar_tracker.kinematics.kinematics_base import (
    angular_diff_deg,
    compute_raw_velocity_vectors,
)
from radar_tracker.kinematics.moving_average import (
    MOVING_AVG_WINDOW,
    apply_moving_average,
)
from radar_tracker.kinematics.savgol import (
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
    apply_savgol_filter,
)

# Huber M-estimator innovation bound multiplier (1.5 * pos_std_m; Huber, 1964).
HUBER_CLIP_SIGMA_MULTIPLIER: float = 1.5

# Residual position std dev (m) for Stage 2 RTS after Stage 1 Hampel outlier removal.
HAMPEL_RTS_CLEAN_POS_STD_M: float = 1.5

# Acceleration spectral density (m^2/s^3) for Stage 2 RTS on outlier-cleaned trajectories.
HAMPEL_RTS_CLEAN_ACCEL_SPEC: float = 8.0

# Baseline 5-scan median window for comparative 1 Hz pipeline benchmarking.
BENCH_DEFAULT_MEDIAN_WINDOW: int = 5

# Baseline 25% relative speed deviation tolerance for comparative pipeline benchmarking.
BENCH_DEFAULT_OUTLIER_REL_TOL: float = 0.25

# Baseline 1-scan neighbor radius (3-point local window) for directional smoothing in benchmark.
BENCH_DEFAULT_NEIGHBOR_RADIUS: int = 1

# Baseline 15.0 deg directional compatibility gate (< smallest 18 deg polygon turn) in benchmark.
BENCH_DEFAULT_MAX_HEADING_DIFF_DEG: float = 15.0

# Step heading change cutoff (< 10.0 deg) isolating straight segments from polygon turns (>= 18 deg).
BENCH_STRAIGHT_TURN_CUTOFF_DEG: float = 10.0

# 1-based track indices (T1 and T3) with constant 20.0 m/s cruise speed used for Speed MAE/Std reference.
BENCH_CRUISE_TARGET_INDICES: Tuple[int, int] = (1, 3)

# Internal floating-point rounding precision during pipeline evaluation.
BENCH_INTERNAL_DECIMALS: int = 4

# Decimal rounding precision for benchmark summary table metrics.
BENCH_METRIC_DECIMALS: int = 3

AVAILABLE_PIPELINES: Tuple[str, ...] = (
    "raw",
    "moving_average",
    "alpha_beta",
    "savgol",
    "kalman_rts",
    "huber_kalman_rts",
    "median_only",
    "hampel_moving_avg",
    "hampel_savgol",
    "hampel_kalman_rts",
    "hampel_directional",
)


def run_alternative_pipeline(
    method: str,
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    dt: np.ndarray,
    vx_raw: np.ndarray,
    vy_raw: np.ndarray,
    median_window_size: int,
    speed_outlier_rel_tol: float,
    pos_std_m: Optional[float] = None,
    accel_spectral_density: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute one of the alternative benchmark pipelines (01 through 10)."""
    p_std = float(pos_std_m) if pos_std_m is not None else RTS_POS_STD_M
    a_spec = (
        float(accel_spectral_density)
        if accel_spectral_density is not None
        else RTS_ACCEL_SPECTRAL_DENSITY
    )
    huber_clip = max(HUBER_INNOVATION_CLIP_M, HUBER_CLIP_SIGMA_MULTIPLIER * p_std)

    if method == "raw":
        return vx_raw, vy_raw
    if method == "moving_average":
        return apply_moving_average(vx_raw, vy_raw, MOVING_AVG_WINDOW, mode="nearest")
    if method == "alpha_beta":
        return apply_alpha_beta_filter(ts, xs, ys)
    if method == "savgol":
        return apply_savgol_filter(vx_raw, vy_raw, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    if method == "kalman_rts":
        return apply_kalman_rts_positions(
            ts, xs, ys, pos_std_m=p_std, accel_spectral_density=a_spec, huber_clip_m=None
        )
    if method == "huber_kalman_rts":
        return apply_kalman_rts_positions(
            ts, xs, ys, pos_std_m=p_std, accel_spectral_density=a_spec, huber_clip_m=huber_clip
        )
    if method == "median_only":
        return reject_waypoint_outliers(vx_raw, vy_raw, median_window_size, speed_outlier_rel_tol)
    if method == "hampel_moving_avg":
        vx_clean, vy_clean = reject_waypoint_outliers(
            vx_raw, vy_raw, median_window_size, speed_outlier_rel_tol
        )
        return apply_moving_average(vx_clean, vy_clean, MOVING_AVG_WINDOW, mode="mirror")
    if method == "hampel_savgol":
        vx_clean, vy_clean = reject_waypoint_outliers(
            vx_raw, vy_raw, median_window_size, speed_outlier_rel_tol
        )
        return apply_savgol_filter(vx_clean, vy_clean, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    if method == "hampel_kalman_rts":
        if pos_std_m is not None and pos_std_m > HIGH_NOISE_POS_STD_THRESHOLD_M:
            vx_rts, vy_rts = apply_kalman_rts_positions(
                ts, xs, ys, pos_std_m=p_std, accel_spectral_density=a_spec, huber_clip_m=huber_clip
            )
            return reject_waypoint_outliers(
                vx_rts, vy_rts, median_window_size, speed_outlier_rel_tol
            )
        vx_clean, vy_clean = reject_waypoint_outliers(
            vx_raw, vy_raw, median_window_size, speed_outlier_rel_tol
        )
        xs_clean = np.concatenate([[xs[0]], xs[0] + np.cumsum(vx_clean * dt)])
        ys_clean = np.concatenate([[ys[0]], ys[0] + np.cumsum(vy_clean * dt)])
        return apply_kalman_rts_positions(
            ts,
            xs_clean,
            ys_clean,
            pos_std_m=HAMPEL_RTS_CLEAN_POS_STD_M,
            accel_spectral_density=HAMPEL_RTS_CLEAN_ACCEL_SPEC,
        )
    raise ValueError(
        f"Unknown kinematic pipeline '{method}'. Valid options: {AVAILABLE_PIPELINES}"
    )


def evaluate_kinematic_pipelines(confirmed_tracks: List[Any]) -> pd.DataFrame:
    """Benchmark all 11 kinematic filtering pipelines across confirmed tracks."""
    rows: List[Dict[str, Any]] = []
    for method in AVAILABLE_PIPELINES:
        speed_maes: List[float] = []
        track_stds: List[float] = []
        max_jumps: List[float] = []
        mean_accels: List[float] = []
        heading_jitters: List[float] = []
        min_speeds_t1_t3: List[float] = []

        for idx, tr in enumerate(confirmed_tracks, start=1):
            pts = tr.plots
            ts = np.array([p["t"] for p in pts], dtype=float)
            xs = np.array([p["x"] for p in pts], dtype=float)
            ys = np.array([p["y"] for p in pts], dtype=float)
            dt_all, vx_raw, vy_raw = compute_raw_velocity_vectors(ts, xs, ys)
            dt_steps = dt_all[1:]

            cfg = getattr(tr, "config", None)
            high_noise_sensor = (
                cfg is not None
                and getattr(cfg, "sigma_pos_xy_m", RTS_POS_STD_M) > HIGH_NOISE_POS_STD_THRESHOLD_M
            )
            if high_noise_sensor:
                total_dt = max(1.0, float(ts[-1] - ts[0]))
                ref_speed = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) / total_dt)
                p_std = float(cfg.sigma_pos_xy_m)
                a_spec = float(cfg.sigma_accel_xy**2)
                m_win = int(cfg.median_window_size)
                s_tol = float(cfg.speed_outlier_rel_tol)
                s_rad = int(cfg.smooth_neighbor_radius)
                h_max = float(cfg.max_heading_diff_deg)
            else:
                raw_spd = np.hypot(vx_raw, vy_raw)
                ref_speed = float(np.median(raw_spd))
                p_std = None
                a_spec = None
                m_win = BENCH_DEFAULT_MEDIAN_WINDOW
                s_tol = BENCH_DEFAULT_OUTLIER_REL_TOL
                s_rad = BENCH_DEFAULT_NEIGHBOR_RADIUS
                h_max = BENCH_DEFAULT_MAX_HEADING_DIFF_DEG

            speeds, headings, _, _ = estimate_track_kinematics(
                pts,
                method=method,
                median_window_size=m_win,
                speed_outlier_rel_tol=s_tol,
                smooth_neighbor_radius=s_rad,
                max_heading_diff_deg=h_max,
                output_decimals=BENCH_INTERNAL_DECIMALS,
                pos_std_m=p_std,
                accel_spectral_density=a_spec,
            )
            spd_arr = np.array([s for s in speeds[1:] if s is not None], dtype=float)
            hdg_arr = np.array([h for h in headings[1:] if h is not None], dtype=float)

            dv = np.abs(np.diff(spd_arr))
            dtheta = np.array(
                [
                    angular_diff_deg(float(hdg_arr[k + 1]), float(hdg_arr[k]))
                    for k in range(len(hdg_arr) - 1)
                ]
            )

            speed_maes.append(float(np.mean(np.abs(spd_arr - ref_speed))))
            track_stds.append(float(np.std(spd_arr)))
            max_jumps.append(float(np.max(dv)))
            mean_accels.append(float(np.mean(dv / dt_steps)))
            straight_dtheta = dtheta[dtheta < BENCH_STRAIGHT_TURN_CUTOFF_DEG]
            heading_jitters.append(float(np.mean(straight_dtheta)) if len(straight_dtheta) else 0.0)
            if idx in BENCH_CRUISE_TARGET_INDICES:
                min_speeds_t1_t3.append(float(np.min(spd_arr)))

        rows.append(
            {
                "pipeline": method,
                "speed_mae_mps": round(float(np.mean(speed_maes)), BENCH_METRIC_DECIMALS),
                "mean_speed_std_mps": round(float(np.mean(track_stds)), BENCH_METRIC_DECIMALS),
                "max_1s_speed_jump_mps": round(float(np.max(max_jumps)), 2),
                "mean_accel_jitter_mps2": round(float(np.mean(mean_accels)), BENCH_METRIC_DECIMALS),
                "min_speed_on_20mps_targets": round(float(np.min(min_speeds_t1_t3)), 2),
                "straight_heading_jitter_deg": round(
                    float(np.mean(heading_jitters)), BENCH_METRIC_DECIMALS
                ),
            }
        )
    return pd.DataFrame(rows)
