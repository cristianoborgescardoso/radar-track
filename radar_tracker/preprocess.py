"""
Geodetic preprocessing and temporal scan clustering for radar plots.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pymap3d as pm

# Conversion factor from pandas datetime64[ns] nanoseconds to Unix UTC seconds (1 s = 10^9 ns).
NANOSECONDS_PER_SECOND: float = 1e9

# Default intra-sweep clustering window (150 ms): >6x above max intra-scan spread (22 ms in plots.json)
# while well below the 1.0 s inter-scan period. Overridden at runtime by TrackerConfig.scan_window_s.
DEFAULT_SCAN_WINDOW_S: float = 0.15


def prepare_plots_dataframe(raw_plots: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """Normalize timestamps, ensure chronological order, and project LLA to local ENU meters.

    Uses ``pymap3d.geodetic2enu`` over the WGS-84 ellipsoid centered at the dataset centroid.

    Args:
        raw_plots: List of raw radar plot dictionaries loaded from JSON.

    Returns:
        Tuple containing:
            - Enriched ``pd.DataFrame`` sorted by time with metric ``(x, y, z)`` ENU columns.
            - Reference origin tuple ``(lat0_deg, lon0_deg, alt0_m)``.
    """
    df = pd.DataFrame(raw_plots)
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["t"] = df["timestamp"].astype(float)
    else:
        df["t"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") / NANOSECONDS_PER_SECOND

    if not df["t"].is_monotonic_increasing:
        df = df.sort_values(["t", "id"], kind="mergesort").reset_index(drop=True)

    lat0 = float(df["latitude"].mean())
    lon0 = float(df["longitude"].mean())
    alt0 = float(df["altitude"].mean())

    df["x"], df["y"], df["z"] = pm.geodetic2enu(
        df["latitude"].to_numpy(dtype=float),
        df["longitude"].to_numpy(dtype=float),
        df["altitude"].to_numpy(dtype=float),
        lat0,
        lon0,
        alt0,
    )
    df["t_rel"] = df["t"] - float(df["t"].iloc[0])
    return df, (lat0, lon0, alt0)


def group_into_scans(
    plots: List[Dict[str, Any]], scan_window_s: float = DEFAULT_SCAN_WINDOW_S
) -> List[List[Dict[str, Any]]]:
    """Group chronologically sorted plots into radar scan frames within ``scan_window_s``.

    Args:
        plots: Chronologically sorted list of plot dictionaries.
        scan_window_s: Maximum time spread (seconds) of a single antenna sweep.

    Returns:
        List of scan frames, where each frame is a list of near-simultaneous plots.
    """
    if not plots:
        return []
    scans: List[List[Dict[str, Any]]] = []
    current_scan = [plots[0]]
    for plot in plots[1:]:
        if plot["t"] - current_scan[0]["t"] <= scan_window_s:
            current_scan.append(plot)
        else:
            scans.append(current_scan)
            current_scan = [plot]
    if current_scan:
        scans.append(current_scan)
    return scans
