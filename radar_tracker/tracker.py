"""
Multi-Target Tracker orchestrator (Global Nearest Neighbor + Track Certainty Lifecycle).
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.optimize import linear_sum_assignment

from radar_tracker.config import TrackerConfig
from radar_tracker.kinematics.estimator import estimate_track_kinematics
from radar_tracker.preprocess import group_into_scans, prepare_plots_dataframe
from radar_tracker.track import GATE_REJECT_COST, RadarTrack3D

# Sentinel cost (>> unassigned_cost_threshold = 500.0) blocking out-of-gate pairs in Hungarian assignment.
INVALID_ASSIGNMENT_SENTINEL: float = 1e6

# Minimum hits required to verify 3-point collinearity (Blackman, Ch. 6) and waive tentative_cost_penalty.
MIN_ESTABLISHED_HITS: int = 3

# Upper cap on hit count when sorting active tracks so all confirmed tracks (hits >= 5) share equal priority.
MAX_PRIORITY_HITS_CAP: int = 5

# Chi-square innovation decay factor in exp(-0.08 * d_M^2), mapping d_M = 1..9 sigma onto [0.92, 0.001].
MAHALANOBIS_CONFIDENCE_DECAY: float = 0.08

# Equal 50/50 convex weight between temporal scan density and mean spatial innovation confidence.
TRACK_CONFIDENCE_WEIGHT: float = 0.5

# Decimal rounding precision for normalized confidence scores in [0, 1].
CONFIDENCE_DECIMALS: int = 3

# Decimal rounding precision for Wald SPRT certainty scores in [0, 100].
CERTAINTY_DECIMALS: int = 1


class MultiTargetTracker:
    """Orchestrates scan batching, Hungarian global assignment, and Track Certainty (0-100) lifecycle."""

    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config: TrackerConfig = config

    def _associate_scan(
        self, active_tracks: List[RadarTrack3D], scan: List[Dict[str, Any]], next_id: int
    ) -> int:
        """Solve global one-to-one plot-to-track assignment for a single radar scan."""
        if not active_tracks:
            for pl in scan:
                active_tracks.append(RadarTrack3D(next_id, pl, self.config))
                next_id += 1
            return next_id

        n_tr, n_pl = len(active_tracks), len(scan)
        cost_matrix = np.full((n_tr, n_pl), GATE_REJECT_COST, dtype=float)
        mahal_matrix = np.zeros((n_tr, n_pl), dtype=float)
        valid_mask = np.zeros((n_tr, n_pl), dtype=bool)

        for i, tr in enumerate(active_tracks):
            for j, pl in enumerate(scan):
                ok, cost, mahal = tr.evaluate_candidate(pl)
                if ok:
                    established = tr.is_confirmed or tr.hits >= max(
                        MIN_ESTABLISHED_HITS, self.config.min_confirmed_hits - 1
                    )
                    penalty = 0.0 if established else self.config.tentative_cost_penalty
                    cost_matrix[i, j] = cost + penalty
                    mahal_matrix[i, j] = mahal
                    valid_mask[i, j] = True

        aug_cost = np.full((n_tr, n_pl + n_tr), INVALID_ASSIGNMENT_SENTINEL, dtype=float)
        aug_cost[:, :n_pl] = np.where(valid_mask, cost_matrix, INVALID_ASSIGNMENT_SENTINEL)
        for i in range(n_tr):
            aug_cost[i, n_pl + i] = self.config.unassigned_cost_threshold

        row_ind, col_ind = linear_sum_assignment(aug_cost)
        assigned_plots = set()
        matched_tracks = set()
        for r, c in zip(row_ind, col_ind):
            if c < n_pl and valid_mask[r, c]:
                active_tracks[r].update(scan[c], mahal_matrix[r, c])
                assigned_plots.add(c)
                matched_tracks.add(r)

        # Penalize certainty for existing tracks that received no detection in this scan.
        # Physics guard: Only apply penalty if at least ~70% of a nominal antenna rotation period
        # has elapsed since the track's last associated hit or previous penalty, preventing
        # spurious clutter-only micro-clusters from inducing premature penalty cascades.
        t_scan = scan[0]["t"]
        for r, tr in enumerate(active_tracks):
            if r not in matched_tracks:
                last_event_t = getattr(tr, "last_penalized_t", tr.last_t)
                if t_scan - last_event_t >= 0.70 * self.config.nominal_scan_period_s:
                    tr.penalize_missed_scan()
                    tr.last_penalized_t = t_scan

        for j, pl in enumerate(scan):
            if j not in assigned_plots:
                active_tracks.append(RadarTrack3D(next_id, pl, self.config))
                next_id += 1
        return next_id

    def _format_confirmed_track(self, idx: int, tr: RadarTrack3D) -> Dict[str, Any]:
        """Serialize a confirmed RadarTrack3D into the required output dictionary structure."""
        track_id = f"T{idx}"
        use_rts_coords = self.config.kinematic_filter_method in (
            "kalman_rts",
            "huber_kalman_rts",
            "hampel_kalman_rts",
        )
        if use_rts_coords:
            smoothed_states = tr.smooth_states_rts()
            pts: List[Dict[str, Any]] = []
            for raw, state in zip(tr.plots, smoothed_states):
                est = dict(raw, x=float(state[0]), y=float(state[2]), z=float(state[4]))
                lat, lon, alt = pm.enu2geodetic(state[0], state[2], state[4], *self._origin)
                est.update(latitude=float(lat), longitude=float(lon), altitude=float(alt))
                pts.append(est)
        else:
            pts = [dict(raw) for raw in tr.plots]

        speeds, headings, avg_speed, avg_heading = estimate_track_kinematics(
            pts,
            method=self.config.kinematic_filter_method,
            median_window_size=self.config.median_window_size,
            speed_outlier_rel_tol=self.config.speed_outlier_rel_tol,
            smooth_neighbor_radius=self.config.smooth_neighbor_radius,
            max_heading_diff_deg=self.config.max_heading_diff_deg,
            output_decimals=self.config.output_decimals,
        )

        point_records: List[Dict[str, Any]] = []
        point_confidences: List[float] = []
        for p_idx, p in enumerate(pts):
            raw_p = tr.plots[p_idx]
            mahal_val = tr.mahalanobis_history[p_idx]
            cert_val = tr.certainty_history[p_idx]
            # Plot-level association confidence in (0, 1]: Gaussian likelihood decay exp(-0.08 * d_M^2).
            # On the 1st hit (initiation), spatial innovation is undefined, so confidence defaults to neutral (0.500).
            if p_idx == 0:
                p_conf = round(0.500, CONFIDENCE_DECIMALS)
            else:
                p_conf = round(
                    float(math.exp(-MAHALANOBIS_CONFIDENCE_DECAY * mahal_val * mahal_val)),
                    CONFIDENCE_DECIMALS,
                )
            point_confidences.append(p_conf)
            p["track_id"] = track_id
            p["speed_mps"] = speeds[p_idx]
            p["heading_deg"] = headings[p_idx]
            p["confidence"] = p_conf
            p["certainty"] = cert_val
            point_records.append(
                {
                    "plot_id": int(p["id"]),
                    "timestamp": p["timestamp"],
                    "latitude": float(p["latitude"]),
                    "longitude": float(p["longitude"]),
                    "altitude": float(p["altitude"]),
                    "raw_latitude": float(raw_p["latitude"]),
                    "raw_longitude": float(raw_p["longitude"]),
                    "raw_altitude": float(raw_p["altitude"]),
                    "speed_mps": speeds[p_idx],
                    "heading_deg": headings[p_idx],
                    "confidence": p_conf,
                    "certainty": cert_val,
                }
            )

        duration = max(1.0, pts[-1]["t"] - pts[0]["t"])
        expected_scans = max(1.0, duration / max(1e-3, self.config.nominal_scan_period_s))
        density_score = min(1.0, len(pts) / (expected_scans + 1.0))
        # Exclude the initiation plot (p_idx=0) from mean innovation confidence evaluation
        mean_spatial_conf = float(np.mean(point_confidences[1:])) if len(point_confidences) > 1 else 0.500
        track_conf = round(
            TRACK_CONFIDENCE_WEIGHT * density_score
            + TRACK_CONFIDENCE_WEIGHT * mean_spatial_conf,
            CONFIDENCE_DECIMALS,
        )

        return {
            "track_id": track_id,
            "confidence": track_conf,
            "certainty": round(float(tr.certainty), CERTAINTY_DECIMALS),
            "plot_ids": [int(p["id"]) for p in pts],
            "average_speed_mps": avg_speed,
            "average_heading_deg": avg_heading,
            "points": point_records,
        }

    def reconstruct(
        self, raw_plots: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[RadarTrack3D], List[RadarTrack3D], List[Dict[str, Any]]]:
        """Run end-to-end track reconstruction on raw radar plots."""
        if not raw_plots:
            return [], [], [], []

        df, self._origin = prepare_plots_dataframe(raw_plots)
        plots: List[Dict[str, Any]] = df.to_dict("records")
        scans = group_into_scans(plots, scan_window_s=self.config.scan_window_s)

        active_tracks: List[RadarTrack3D] = []
        finished_tracks: List[RadarTrack3D] = []
        next_id = 1

        for scan in scans:
            t_scan = scan[0]["t"]
            still_active: List[RadarTrack3D] = []
            for tr in active_tracks:
                if t_scan - tr.last_t > self.config.max_coast_s or tr.certainty <= 0.0:
                    finished_tracks.append(tr)
                else:
                    still_active.append(tr)
            active_tracks = still_active
            active_tracks.sort(key=lambda tr: (-min(tr.hits, MAX_PRIORITY_HITS_CAP), tr.internal_id))
            next_id = self._associate_scan(active_tracks, scan, next_id)

        all_tracks = finished_tracks + active_tracks
        confirmed = [tr for tr in all_tracks if tr.is_confirmed]
        rejected = [tr for tr in all_tracks if not tr.is_confirmed]
        confirmed.sort(key=lambda tr: (tr.plots[0]["t"], tr.plots[0]["id"]))

        output_tracks = [
            self._format_confirmed_track(idx, tr) for idx, tr in enumerate(confirmed, start=1)
        ]

        for tr in rejected:
            for p in tr.plots:
                p["track_id"] = "CLUTTER"
                p["speed_mps"] = None
                p["heading_deg"] = None
                p["confidence"] = 0.0
                p["certainty"] = tr.certainty

        return output_tracks, confirmed, rejected, plots

    @staticmethod
    def export_track_folders(
        output_tracks: List[Dict[str, Any]], base_folder: str = "diagnostics/tracks"
    ) -> None:
        """Create one dedicated folder per confirmed track containing its JSON and CSV state log."""
        os.makedirs(base_folder, exist_ok=True)
        for tr_dict in output_tracks:
            tid = tr_dict["track_id"]
            track_dir = os.path.join(base_folder, tid)
            os.makedirs(track_dir, exist_ok=True)
            with open(os.path.join(track_dir, f"{tid}_track.json"), "w", encoding="utf-8") as f:
                json.dump(tr_dict, f, indent=2)
            pd.DataFrame(tr_dict["points"]).to_csv(
                os.path.join(track_dir, f"{tid}_points.csv"), index=False
            )

    @staticmethod
    def export_tracks_csv(output_tracks: List[Dict[str, Any]], csv_path: str = "tracks.csv") -> None:
        """Export a flat CSV version of all reconstructed tracks and their assigned points (Section 9 extension)."""
        rows: List[Dict[str, Any]] = []
        for tr_dict in output_tracks:
            for pt in tr_dict["points"]:
                rows.append(
                    {
                        "track_id": tr_dict["track_id"],
                        "track_confidence": tr_dict["confidence"],
                        "track_certainty": tr_dict["certainty"],
                        "average_speed_mps": tr_dict["average_speed_mps"],
                        "average_heading_deg": tr_dict["average_heading_deg"],
                        **pt,
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)


if __name__ == "__main__":
    from radar_tracker.cli import main

    main()
