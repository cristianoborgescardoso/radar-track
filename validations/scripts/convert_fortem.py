"""
AERPAW / Fortem TrueView R20 Ingestion and Geodetic Normalization Utility.

Responsibility: Ingest raw AERPAW radar scans and GPS ground truth when provided,
apply bore-sight yaw calibration (+2.0 deg ENU), and format detections into the
standard challenge schema (plots.json with ISO-8601 UTC timestamps).

When running from the repository, uses the pre-converted benchmark subset in
data/aerpaw_opt2_subset/ directly without requiring multi-gigabyte raw downloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymap3d as pm

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DATA_OUT = HERE / "data" / "aerpaw_opt2_subset"

# Surveyed WGS-84 coordinates (lat, lon, alt_m) of the Fortem TrueView R20 radar at AERPAW Lake Wheeler Field Lab:
RADAR_ORIGIN: Tuple[float, float, float] = (35.724976, -78.691933, 10.0)

# +2.0 deg CCW ENU azimuth bore-sight alignment recommended by the AERPAW / Dryad dataset authors:
RADAR_YAW_ALIGN_DEG: float = 2.0
FULL_CIRCLE_DEG: float = 360.0
FORTEM_TARGET_IDS = (491, 919)

DEFAULT_START_OFFSET_S: float = 360.0
DEFAULT_DURATION_S: float = 60.0
GPS_INTERP_MARGIN_S: float = 1.0


def to_iso_utc(epoch_s: float) -> str:
    """Format a Unix UTC epoch timestamp (seconds) as ISO-8601 UTC string YYYY-MM-DDTHH:MM:SS.sssZ."""
    dt = datetime.fromtimestamp(float(epoch_s), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def rotate_lla(lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
    """Apply the authors' +2.0 degree ENU yaw bore-sight correction about the radar tripod origin."""
    e, n, u = pm.geodetic2enu(lat, lon, alt, *RADAR_ORIGIN)
    theta = math.radians(RADAR_YAW_ALIGN_DEG)
    er = math.cos(theta) * e - math.sin(theta) * n
    nr = math.sin(theta) * e + math.cos(theta) * n
    lat_r, lon_r, alt_r = pm.enu2geodetic(er, nr, u, *RADAR_ORIGIN)
    return float(lat_r), float(lon_r), float(alt_r)


def distance_xy_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Horizontal distance (meters) between two WGS-84 coordinates relative to the radar origin."""
    e1, n1, _ = pm.geodetic2enu(a[0], a[1], RADAR_ORIGIN[2], *RADAR_ORIGIN)
    e2, n2, _ = pm.geodetic2enu(b[0], b[1], RADAR_ORIGIN[2], *RADAR_ORIGIN)
    return float(math.hypot(e1 - e2, n1 - n2))


def parse_gps_log(gps_file: Path) -> List[Dict[str, Any]]:
    """Parse AERPAW RTK GPS vehicleOut text file into UTC timestamped records."""
    rows: List[Dict[str, Any]] = []
    edt = timezone(timedelta(hours=-4))
    with gps_file.open("r", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 10:
                continue
            local_time = datetime.strptime(row[7], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=edt)
            rows.append(
                {
                    "sequence": int(row[0]),
                    "longitude": float(row[1]),
                    "latitude": float(row[2]),
                    "altitude_gps": float(row[3]),
                    "timestamp": local_time.astimezone(timezone.utc).timestamp(),
                    "target_id": "UAV1",
                }
            )
    return rows


def iter_radar_file(radar_file: Path):
    """Yield radar JSON records containing antenna params and detections."""
    with radar_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                if "params" in rec and "data" in rec:
                    yield rec


def convert_raw_dataset(
    raw_dir: Path,
    output_dir: Path,
    start_offset_s: float = DEFAULT_START_OFFSET_S,
    duration_s: float = DEFAULT_DURATION_S,
    airborne_only: bool = True,
) -> None:
    """Convert raw radar and GPS logs into plots.json and ground_truth.csv."""
    radar_files = list(raw_dir.glob("radar_data_*.json"))
    gps_files = list(raw_dir.glob("*vehicleOut*.txt"))

    if not radar_files or not gps_files:
        raise FileNotFoundError(
            f"Raw radar or GPS logs not found in {raw_dir}. "
            f"Expected 'radar_data_*.json' and '*vehicleOut*.txt'."
        )

    radar_file = radar_files[0]
    gps_file = gps_files[0]

    gps_all = parse_gps_log(gps_file)
    if not gps_all:
        raise ValueError(f"No valid GPS records found in {gps_file}")

    start_epoch = gps_all[0]["timestamp"] + start_offset_s
    end_epoch = start_epoch + duration_s

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    total_detections = 0

    for scan in iter_radar_file(radar_file):
        scan_t = float(scan["params"]["globalTime"])
        if not (start_epoch <= scan_t <= end_epoch):
            continue

        for det in scan.get("data", []):
            total_detections += 1
            if airborne_only and det.get("onGround", False):
                continue
            lat, lon, alt = rotate_lla(*map(float, det["lla"][:3]))
            plots.append(
                {
                    "id": total_detections,
                    "timestamp": to_iso_utc(scan_t),
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                }
            )

        for track_rec in scan.get("trackData", []):
            tid = int(track_rec.get("id", -1))
            if tid in FORTEM_TARGET_IDS and int(track_rec.get("stepsSinceUpdated", 0)) == 0:
                ref_t = float(track_rec.get("globalTime", scan_t))
                lat, lon, alt = rotate_lla(*map(float, track_rec["lla"][:3]))
                v_ned = track_rec.get("velocityNed")
                if v_ned and len(v_ned) >= 2:
                    vn, ve = float(v_ned[0]), float(v_ned[1])
                    r20_spd = round(math.hypot(ve, vn), 2)
                    raw_hdg = (math.degrees(math.atan2(ve, vn)) - RADAR_YAW_ALIGN_DEG + FULL_CIRCLE_DEG) % FULL_CIRCLE_DEG
                    r20_hdg = round(raw_hdg, 2)
                else:
                    r20_spd, r20_hdg = None, None
                refs.append(
                    {
                        "fortem_track_id": tid,
                        "timestamp": to_iso_utc(ref_t),
                        "epoch_s": ref_t,
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": alt,
                        "speed_mps": r20_spd,
                        "heading_deg": r20_hdg,
                    }
                )

    (output_dir / "plots.json").write_text(json.dumps(plots, indent=2), encoding="utf-8")

    gps_window = [
        {
            "sequence": row["sequence"],
            "timestamp": to_iso_utc(row["timestamp"]),
            "epoch_s": row["timestamp"],
            "target_id": row["target_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "altitude_gps": row["altitude_gps"],
        }
        for row in gps_all
        if (start_epoch - GPS_INTERP_MARGIN_S) <= row["timestamp"] <= (end_epoch + GPS_INTERP_MARGIN_S)
    ]

    with (output_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence", "timestamp", "epoch_s", "target_id", "latitude", "longitude", "altitude_gps"],
        )
        writer.writeheader()
        writer.writerows(gps_window)

    with (output_dir / "fortem_reference_tracks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fortem_track_id", "timestamp", "epoch_s", "latitude", "longitude", "altitude", "speed_mps", "heading_deg"],
        )
        writer.writeheader()
        writer.writerows(refs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Fortem TrueView R20 radar and RTK GPS logs into challenge plots.json schema."
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Path to raw radar and vehicleOut directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_OUT, help="Output directory for plots.json")
    parser.add_argument("--start", type=float, default=DEFAULT_START_OFFSET_S, help="Start offset after GPS epoch (s)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Subset duration window (s)")
    parser.add_argument("--airborne-only", action="store_true", help="Filter out plots with onGround=True")
    args = parser.parse_args()

    out_dir: Path = args.out_dir

    if args.raw_dir is not None and args.raw_dir.exists():
        print(f"[convert_fortem] Converting raw radar data from {args.raw_dir} -> {out_dir}...")
        convert_raw_dataset(
            raw_dir=args.raw_dir,
            output_dir=out_dir,
            start_offset_s=args.start,
            duration_s=args.duration,
            airborne_only=args.airborne_only,
        )
        print(f"[convert_fortem] Successfully generated plots.json and ground_truth.csv in {out_dir}.")
        return

    # Default repo execution: check pre-converted benchmark dataset
    plots_file = out_dir / "plots.json"
    gt_file = out_dir / "ground_truth.csv"
    if plots_file.exists() and gt_file.exists():
        print(f"[convert_fortem] Pre-converted benchmark dataset verified in {out_dir} ({plots_file.name}, {gt_file.name}).")
        return

    raise FileNotFoundError(
        f"Pre-converted dataset not found in {out_dir} and no --raw-dir specified."
    )


if __name__ == "__main__":
    main()
