"""
Orchestrator for Fortem TrueView R20 Real-World Radar Validation.

Responsibility: Load CLI output tracks and ground-truth references, invoke
geodetic error calculation, write metrics.json, and delegate visual reporting
to render_comparison_html and plot_corridor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geo_metrics import (
    DEFAULT_PLAYBACK_DURATION_S,
    FORTEM_PRIMARY_TRACK_ID,
    compute_series_kinematics,
    evaluate_against_gps,
    parse_ts,
    read_csv_rows,
    to_enu_xy,
)
from plot_corridor import generate_corridor_comparison_png
from render_comparison_html import generate_comparison_html

DEFAULT_DATA_DIR = HERE / "data" / "aerpaw_opt2_subset"
DEFAULT_OUT_DIR = HERE / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CLI reconstructed tracks against RTK GPS & Fortem R20 onboard tracker."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to input aerpaw_opt2_subset directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Path to CLI validation output directory")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_json_path = out_dir / "tracks.json"
    if not tracks_json_path.exists():
        raise FileNotFoundError(f"CLI output tracks not found: {tracks_json_path}. Run tracker.ps1/sh first.")

    tracks_data = json.loads(tracks_json_path.read_text(encoding="utf-8"))
    tracks = tracks_data if isinstance(tracks_data, list) else tracks_data.get("tracks", [])
    if not tracks:
        raise RuntimeError("No confirmed tracks found in tracks.json.")

    plots = json.loads((data_dir / "plots.json").read_text(encoding="utf-8"))
    gps_raw = read_csv_rows(data_dir / "ground_truth.csv")
    refs_raw = read_csv_rows(data_dir / "fortem_reference_tracks.csv")

    import numpy as np

    gps_points = compute_series_kinematics(gps_raw)
    gps_ts = np.array([p["epoch_s"] for p in gps_points], dtype=float)
    gps_xy = to_enu_xy(gps_points)
    gps_spds = np.array([float(p["speed_mps"]) for p in gps_points[1:]], dtype=float)

    r20_919_raw = [r for r in refs_raw if int(r["fortem_track_id"]) == FORTEM_PRIMARY_TRACK_ID]
    r20_points = compute_series_kinematics(r20_919_raw)

    t1 = tracks[0]
    t1_metrics, t1_errors = evaluate_against_gps(t1["points"], gps_ts, gps_xy, gps_spds)
    raw_pts_for_eval = [
        {
            "timestamp": p["timestamp"],
            "latitude": p.get("raw_latitude", p["latitude"]),
            "longitude": p.get("raw_longitude", p["longitude"]),
        }
        for p in t1["points"]
    ]
    raw_metrics, _ = evaluate_against_gps(raw_pts_for_eval, gps_ts, gps_xy, gps_spds)
    r20_metrics, r20_errors = evaluate_against_gps(r20_points, gps_ts, gps_xy, gps_spds)

    results = {
        "tracker_estimated": [
            {
                "track_id": t1["track_id"],
                "average_speed_mps": t1["average_speed_mps"],
                "average_heading_deg": t1["average_heading_deg"],
                "certainty": t1["certainty"],
                "confidence": t1["confidence"],
                **t1_metrics,
            }
        ],
        "tracker_raw_associated_plots": raw_metrics,
        "fortem_r20_onboard_919": {
            "fortem_track_id": FORTEM_PRIMARY_TRACK_ID,
            **r20_metrics,
        },
        "method": (
            "Horizontal error evaluated against linearly interpolated UAV RTK GPS "
            "at each estimate timestamp (no extrapolation)."
        ),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    all_ts = [parse_ts(p["timestamp"]) for p in plots]
    t0_s = min(all_ts) if all_ts else float(gps_ts[0])
    max_sec = int(math.ceil(max(all_ts) - t0_s)) if all_ts else DEFAULT_PLAYBACK_DURATION_S

    assigned_ids = set(t1.get("plot_ids", [int(p["plot_id"]) for p in t1["points"]]))
    clutter_plots = [p for p in plots if int(p["id"]) not in assigned_ids]

    comparison_html = out_dir / "comparison.html"
    generate_comparison_html(
        track_t1=t1,
        t1_metrics=t1_metrics,
        t1_errors=t1_errors,
        r20_points=r20_points,
        r20_metrics=r20_metrics,
        r20_errors=r20_errors,
        gps_points=gps_points,
        clutter_plots=clutter_plots,
        t0_s=t0_s,
        max_sec=max_sec,
        output_html=comparison_html,
    )

    corridor_png = out_dir / "fortem_gps_comparison_corridor.png"
    generate_corridor_comparison_png(
        track_t1=t1,
        t1_metrics=t1_metrics,
        t1_errors=t1_errors,
        raw_metrics=raw_metrics,
        r20_points=r20_points,
        r20_metrics=r20_metrics,
        r20_errors=r20_errors,
        gps_points=gps_points,
        t0_s=t0_s,
        output_png=corridor_png,
    )
    # Also provide alias gps_comparison_corridor.png for backwards-compatibility
    legacy_corridor_png = out_dir / "gps_comparison_corridor.png"
    if corridor_png.exists() and corridor_png != legacy_corridor_png:
        import shutil
        shutil.copyfile(corridor_png, legacy_corridor_png)

    print(
        f"[Evaluate Fortem R20] Comparison Summary against UAV RTK GPS:\n"
        f"  - Our Tracker T1 (tracks.csv) : Pos RMSE = {t1_metrics['rmse_xy_m']:.2f} m (Med = {t1_metrics['median_xy_m']:.2f} m) | "
        f"Mean Speed = {t1_metrics['mean_speed_mps']:.2f} m/s (Speed MAE = {t1_metrics['speed_mae_vs_gps_mps']:.2f} m/s)\n"
        f"  - Fortem R20 Onboard (#919)   : Pos RMSE = {r20_metrics['rmse_xy_m']:.2f} m (Med = {r20_metrics['median_xy_m']:.2f} m) | "
        f"Mean Speed = {r20_metrics['mean_speed_mps']:.2f} m/s (Speed MAE = {r20_metrics['speed_mae_vs_gps_mps']:.2f} m/s)\n"
        f"  - Raw Associated Radar Plots  : Pos RMSE = {raw_metrics['rmse_xy_m']:.2f} m | "
        f"MSE = {raw_metrics['mse_xy_m2']:.2f} m^2\n"
        f"[Evaluate Fortem R20] Generated comparison files:\n"
        f"  -> Metrics JSON          : {metrics_path}\n"
        f"  -> Comparison HTML (Play): {comparison_html}\n"
        f"  -> Corridor Plot PNG     : {corridor_png}"
    )


if __name__ == "__main__":
    main()
