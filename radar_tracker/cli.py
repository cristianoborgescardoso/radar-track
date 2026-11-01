#!/usr/bin/env python3
"""
CLI entrypoint for Radar Plot Track Reconstruction.

Loads default parameters from ``tracker.properties`` automatically, or accepts a custom
properties file via ``--config <file.properties>``.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_tracker import (
    MultiTargetTracker,
    RadarTrack3D,
    TrackerConfig,
    angular_diff_deg,
    estimate_track_kinematics,
    group_into_scans,
    prepare_plots_dataframe,
)
from radar_tracker.visualize_tracks import (
    create_certainty_visualization,
    create_interactive_html_map,
    create_visualization,
)


def reconstruct_tracks(
    raw_plots: List[Dict[str, Any]],
    scan_window_s: float = 0.15,
    max_coast_s: float = 6.0,
    min_confirmed_hits: int = 4,
    config_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[RadarTrack3D], List[RadarTrack3D], List[Dict[str, Any]]]:
    """Convenience function wrapping ``MultiTargetTracker.reconstruct``."""
    if config_path is not None:
        cfg = TrackerConfig.from_properties(config_path)
    else:
        cfg = TrackerConfig(
            scan_window_s=scan_window_s,
            max_coast_s=max_coast_s,
            min_confirmed_hits=min_confirmed_hits,
        )
    return MultiTargetTracker(cfg).reconstruct(raw_plots)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct radar target tracks from plots.json using tracker.properties"
    )
    parser.add_argument("input_file", nargs="?", default="plots.json", help="Path to input plots.json")
    parser.add_argument("output_file", nargs="?", default="tracks.json", help="Path to output tracks.json")
    parser.add_argument(
        "image_file",
        nargs="?",
        default=None,
        help="Optional path to output 4-panel visualization PNG (defaults to <output_stem>_visualization.png)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to custom .properties configuration file (defaults to tracker.properties)",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_plots = json.load(f)

    config = TrackerConfig.from_properties(args.config)
    tracks_out, confirmed, rejected, _ = MultiTargetTracker(config).reconstruct(raw_plots)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(tracks_out, f, indent=2)

    base_stem = (
        args.output_file[:-5]
        if args.output_file.lower().endswith(".json")
        else args.output_file
    )
    csv_file = f"{base_stem}.csv"
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_prefix = os.path.dirname(args.output_file)

    vis_png = args.image_file if args.image_file else f"{base_stem}_visualization.png"
    cert_png = f"{base_stem}_certainty_lifecycle.png"
    html_map = f"{base_stem}_interactive_map.html"

    if os.path.normcase(out_dir) == os.path.normcase(root_dir):
        tracks_folder = os.path.join(out_prefix, "diagnostics", "tracks")
    else:
        tracks_folder = os.path.join(out_prefix, "tracks")

    MultiTargetTracker.export_tracks_csv(tracks_out, csv_file)
    MultiTargetTracker.export_track_folders(tracks_out, base_folder=tracks_folder)
    create_visualization(
        Path(args.input_file),
        Path(args.output_file),
        Path(vis_png),
        raw_plots=raw_plots,
        tracks=tracks_out,
    )
    create_certainty_visualization(
        confirmed,
        rejected,
        config,
        Path(cert_png),
    )
    create_interactive_html_map(
        raw_plots,
        tracks_out,
        Path(html_map),
    )

    if os.path.normcase(out_dir) == os.path.normcase(root_dir):
        diag_dir = os.path.join(root_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        shutil.copy2(cert_png, os.path.join(diag_dir, "certainty_lifecycle.png"))
        shutil.copy2(html_map, os.path.join(diag_dir, "interactive_radar_map.html"))

    rejected_pts = sum(len(t.plots) for t in rejected)
    cfg_label = args.config if args.config else "tracker.properties (default)"
    track_ids_list = ", ".join(t["track_id"] for t in tracks_out)
    print(
        f"[Tracker] Config: {cfg_label} | Filter: {config.kinematic_filter_method}\n"
        f"[Tracker] Processed {len(raw_plots)} plots -> "
        f"Reconstructed {len(confirmed)} confirmed tracks "
        f"({sum(len(t['plot_ids']) for t in tracks_out)} plots assigned, "
        f"{rejected_pts} clutter plots rejected)."
    )
    for t in tracks_out:
        print(
            f"  - {t['track_id']}: {len(t['plot_ids'])} plots | "
            f"avg_speed={t['average_speed_mps']:.2f} m/s | "
            f"avg_heading={t['average_heading_deg']:.2f} deg | "
            f"certainty={t['certainty']:.0f} | confidence={t['confidence']:.3f}"
        )
    print(
        f"[Tracker] Generated files:\n"
        f"  -> JSON Tracks      : {args.output_file}\n"
        f"  -> CSV Tracks       : {csv_file}\n"
        f"  -> Track Logs       : {tracks_folder}{os.sep}[{track_ids_list}]\n"
        f"  -> PNG 4-Panel Plot : {vis_png}\n"
        f"  -> PNG Certainty    : {cert_png}\n"
        f"  -> Interactive HTML : {html_map}"
    )


if __name__ == "__main__":
    main()
