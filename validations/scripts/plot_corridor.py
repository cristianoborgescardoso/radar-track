"""
3-Panel Corridor Comparison Plot Generator.

Responsibility: Render PNG charts comparing spatial
corridor trajectories, instantaneous horizontal errors, and speed profiles
between Our Tracker T1, Fortem R20 #919, and RTK GPS Ground Truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geo_metrics import parse_ts, to_enu_xy


def generate_corridor_comparison_png(
    track_t1: Dict[str, Any],
    t1_metrics: Dict[str, Any],
    t1_errors: List[float],
    raw_metrics: Dict[str, Any],
    r20_points: List[Dict[str, Any]],
    r20_metrics: Dict[str, Any],
    r20_errors: List[float],
    gps_points: List[Dict[str, Any]],
    t0_s: float,
    output_png: Path,
) -> None:
    """Render 3-panel spatial, positional error & speed comparison PNG from exported tracks.json."""
    g_xy = to_enu_xy(gps_points)
    r20_xy = to_enu_xy(r20_points)
    t1_xy = to_enu_xy(track_t1["points"])
    raw_rows = [
        {
            "latitude": p.get("raw_latitude", p["latitude"]),
            "longitude": p.get("raw_longitude", p["longitude"]),
        }
        for p in track_t1["points"]
    ]
    raw_xy = to_enu_xy(raw_rows)
    t_rel = [parse_ts(p["timestamp"]) - t0_s for p in track_t1["points"]]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), dpi=140)

    # Panel 1: Trajectory in horizontal ENU plane
    axes[0].scatter(
        raw_xy[:, 0],
        raw_xy[:, 1],
        color="#10b981",
        alpha=0.40,
        s=22,
        label=f"Associated Raw Radar Plots (RMSE = {raw_metrics['rmse_xy_m']:.2f} m)",
    )
    axes[0].plot(
        r20_xy[:, 0],
        r20_xy[:, 1],
        "--",
        color="#a855f7",
        lw=2.0,
        label=f"Fortem R20 Onboard #919 (RMSE = {r20_metrics['rmse_xy_m']:.2f} m)",
    )
    axes[0].plot(
        g_xy[:, 0],
        g_xy[:, 1],
        "k-",
        lw=3.0,
        label="RTK GPS Ground Truth (UAV)",
    )
    axes[0].plot(
        t1_xy[:, 0],
        t1_xy[:, 1],
        "-",
        color="#0284c7",
        lw=2.8,
        label=f"Our Tracker T1 — tracks.csv (RMSE = {t1_metrics['rmse_xy_m']:.2f} m)",
    )
    axes[0].set_title(
        "1. Flight Corridor Trajectory (ENU Relative to Radar)",
        fontweight="bold",
    )
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Instantaneous error vs. RTK GPS
    r20_t_rel = [parse_ts(p.get("epoch_s", p["timestamp"])) - t0_s for p in r20_points]
    axes[1].plot(
        r20_t_rel,
        r20_errors,
        color="#a855f7",
        lw=1.5,
        alpha=0.65,
        label=f"Fortem R20 #919 Error (RMSE = {r20_metrics['rmse_xy_m']:.2f} m, Med = {r20_metrics['median_xy_m']:.2f} m)",
    )
    axes[1].plot(
        t_rel,
        t1_errors,
        color="#0284c7",
        lw=2.2,
        marker="o",
        ms=3.2,
        label=f"Our Tracker T1 Error (RMSE = {t1_metrics['rmse_xy_m']:.2f} m, Med = {t1_metrics['median_xy_m']:.2f} m)",
    )
    axes[1].axhline(
        t1_metrics["rmse_xy_m"],
        color="#0284c7",
        ls="--",
        lw=1.5,
        label=f"Our Tracker T1 RMSE ({t1_metrics['rmse_xy_m']:.2f} m)",
    )
    axes[1].axhline(
        r20_metrics["rmse_xy_m"],
        color="#a855f7",
        ls=":",
        lw=1.8,
        label=f"Fortem R20 #919 RMSE ({r20_metrics['rmse_xy_m']:.2f} m)",
    )
    axes[1].set_title(
        "2. Instantaneous Horizontal Positional Error vs. RTK GPS",
        fontweight="bold",
    )
    axes[1].set_xlabel("Elapsed Time in Window (s)")
    axes[1].set_ylabel("2D Horizontal Error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8.5)

    # Panel 3: Speed comparison over time
    gps_t_spd = [parse_ts(p.get("epoch_s", p["timestamp"])) - t0_s for p in gps_points if p.get("speed_mps") is not None]
    gps_v_spd = [float(p["speed_mps"]) for p in gps_points if p.get("speed_mps") is not None]
    r20_t_spd = [parse_ts(p.get("epoch_s", p["timestamp"])) - t0_s for p in r20_points if p.get("speed_mps") is not None]
    r20_v_spd = [float(p["speed_mps"]) for p in r20_points if p.get("speed_mps") is not None]
    t1_t_spd = [parse_ts(p["timestamp"]) - t0_s for p in track_t1["points"] if p.get("speed_mps") is not None]
    t1_v_spd = [float(p["speed_mps"]) for p in track_t1["points"] if p.get("speed_mps") is not None]

    axes[2].plot(
        gps_t_spd,
        gps_v_spd,
        "k-",
        lw=2.5,
        alpha=0.85,
        label=f"RTK GPS Speed (Mean = {np.mean(gps_v_spd):.2f} m/s)",
    )
    axes[2].plot(
        r20_t_spd,
        r20_v_spd,
        "--",
        color="#a855f7",
        lw=1.8,
        alpha=0.85,
        label=f"Fortem R20 #919 Onboard Speed (Mean = {r20_metrics['mean_speed_mps']:.2f} m/s, MAE = {r20_metrics['speed_mae_vs_gps_mps']:.2f} m/s)",
    )
    axes[2].plot(
        t1_t_spd,
        t1_v_spd,
        "-",
        color="#0284c7",
        lw=2.4,
        marker="o",
        ms=3.0,
        label=f"Our Tracker T1 Speed (Mean = {t1_metrics['mean_speed_mps']:.2f} m/s, MAE = {t1_metrics['speed_mae_vs_gps_mps']:.2f} m/s)",
    )
    axes[2].set_title(
        "3. Estimated Horizontal Speed vs. R20 (#919) vs. RTK GPS",
        fontweight="bold",
    )
    axes[2].set_xlabel("Elapsed Time in Window (s)")
    axes[2].set_ylabel("Horizontal Speed (m/s)")
    axes[2].set_ylim(0, 16)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Real-World External Validation — Our Tracker (T1) vs. Fortem TrueView R20 Radar (#919) vs. RTK GPS",
        fontsize=14,
        fontweight="bold",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=140)
    plt.close(fig)
