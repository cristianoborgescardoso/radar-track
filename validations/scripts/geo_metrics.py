"""
Geodetic projections, timestamp parsing, and GPS ground-truth benchmarking metrics.

Responsibility: Pure mathematical/geodetic evaluation of radar trajectories
against RTK GPS ground truth on the WGS-84 ellipsoid and local ENU plane.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pymap3d as pm

# Surveyed WGS-84 coordinates (lat_deg, lon_deg, alt_m) of the Fortem R20 tripod (AERPAW Lake Wheeler Field Lab).
RADAR_ORIGIN: Tuple[float, float, float] = (35.724976, -78.691933, 10.0)

# Internal track ID (#919) assigned by the Fortem R20 onboard tracker during the 360-420 s flight corridor.
FORTEM_PRIMARY_TRACK_ID: int = 919

# Minimum positive time step floor (1 ms) preventing division-by-zero in finite-difference speed calculation.
MIN_TIME_STEP_S: float = 1e-3

# Full compass circle modulus (360.0 deg) for heading normalization in [0, 360).
FULL_CIRCLE_DEG: float = 360.0

# 90th-percentile (P90) tail error threshold (%).
PERCENTILE_90: int = 90

# Default 60-second duration fallback for the interactive HTML playback slider.
DEFAULT_PLAYBACK_DURATION_S: int = 60


def parse_ts(value: Any) -> float:
    """Parse either ISO-8601 UTC string or numeric Unix timestamp into float seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_enu_xy(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Project WGS-84 latitude/longitude rows to local 2D ENU (East, North) coordinates in meters."""
    lats = [float(r["latitude"]) for r in rows]
    lons = [float(r["longitude"]) for r in rows]
    e, n, _ = pm.geodetic2enu(lats, lons, RADAR_ORIGIN[2], *RADAR_ORIGIN)
    return np.column_stack([np.asarray(e, dtype=float), np.asarray(n, dtype=float)])


def compute_series_kinematics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute or preserve step-by-step speed (m/s) and compass heading (deg) for reference series."""
    if not rows:
        return []
    xy = to_enu_xy(rows)
    ts = np.array([parse_ts(r.get("epoch_s", r["timestamp"])) for r in rows], dtype=float)
    enriched: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        item = dict(r)
        item["epoch_s"] = float(ts[i])
        existing_spd = r.get("speed_mps")
        existing_hdg = r.get("heading_deg")
        if existing_spd not in (None, "") and existing_hdg not in (None, ""):
            item["speed_mps"] = round(float(existing_spd), 2)
            item["heading_deg"] = round(float(existing_hdg), 1)
        elif i == 0 or (ts[i] - ts[i - 1]) <= MIN_TIME_STEP_S:
            item["speed_mps"] = None
            item["heading_deg"] = None
        else:
            dt = float(ts[i] - ts[i - 1])
            dx = float(xy[i, 0] - xy[i - 1, 0])
            dy = float(xy[i, 1] - xy[i - 1, 1])
            item["speed_mps"] = round(math.hypot(dx, dy) / dt, 2)
            item["heading_deg"] = round((math.degrees(math.atan2(dx, dy)) + FULL_CIRCLE_DEG) % FULL_CIRCLE_DEG, 1)
        enriched.append(item)
    return enriched


def evaluate_against_gps(
    rows: List[Dict[str, Any]],
    gps_ts: np.ndarray,
    gps_xy: np.ndarray,
    gps_spds: np.ndarray,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Evaluate horizontal position and speed error against linearly interpolated GPS Ground Truth.

    Returns:
        tuple (metrics_dict, per_point_error_list)
    """
    ts = np.array([parse_ts(r.get("epoch_s", r["timestamp"])) for r in rows], dtype=float)
    xy = to_enu_xy(rows)
    per_point_err: List[float] = []
    valid_d2: List[float] = []
    valid_ts: List[float] = []
    speed_errs: List[float] = []
    valid_spds: List[float] = []

    for i, t in enumerate(ts):
        spd_val = rows[i].get("speed_mps")
        if spd_val not in (None, ""):
            valid_spds.append(float(spd_val))
        if gps_ts[0] <= t <= gps_ts[-1]:
            gx = float(np.interp(t, gps_ts, gps_xy[:, 0]))
            gy = float(np.interp(t, gps_ts, gps_xy[:, 1]))
            d2 = float((xy[i, 0] - gx) ** 2 + (xy[i, 1] - gy) ** 2)
            err_m = math.sqrt(d2)
            per_point_err.append(round(err_m, 2))
            valid_d2.append(d2)
            valid_ts.append(float(t))
            if spd_val not in (None, ""):
                g_spd = float(np.interp(t, gps_ts[1:], gps_spds))
                speed_errs.append(abs(float(spd_val) - g_spd))
        else:
            per_point_err.append(0.0)

    if not valid_d2:
        return {
            "samples": 0,
            "mse_xy_m2": None,
            "rmse_xy_m": None,
            "median_xy_m": None,
            "p90_xy_m": None,
            "mean_speed_mps": None,
            "speed_mae_vs_gps_mps": None,
            "speed_rmse_vs_gps_mps": None,
            "duration_s": 0.0,
        }, per_point_err

    d2_arr = np.asarray(valid_d2, dtype=float)
    d_arr = np.sqrt(d2_arr)
    spd_err_arr = np.asarray(speed_errs, dtype=float) if speed_errs else np.zeros(1)
    return {
        "samples": len(valid_d2),
        "mse_xy_m2": round(float(np.mean(d2_arr)), 3),
        "rmse_xy_m": round(float(np.sqrt(np.mean(d2_arr))), 3),
        "median_xy_m": round(float(np.median(d_arr)), 3),
        "p90_xy_m": round(float(np.percentile(d_arr, PERCENTILE_90)), 3),
        "mean_speed_mps": round(float(np.mean(valid_spds)), 2) if valid_spds else None,
        "speed_mae_vs_gps_mps": round(float(np.mean(spd_err_arr)), 2) if speed_errs else None,
        "speed_rmse_vs_gps_mps": round(float(np.sqrt(np.mean(spd_err_arr**2))), 2) if speed_errs else None,
        "duration_s": round(float(valid_ts[-1] - valid_ts[0]), 3),
    }, per_point_err
