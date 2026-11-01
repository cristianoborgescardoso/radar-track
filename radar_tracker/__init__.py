"""
Modular 3D Multi-Target Radar Tracker package.
"""

from radar_tracker.config import TrackerConfig
from radar_tracker.kinematics import (
    AVAILABLE_PIPELINES,
    angular_diff_deg,
    estimate_track_kinematics,
    evaluate_kinematic_pipelines,
)
from radar_tracker.preprocess import group_into_scans, prepare_plots_dataframe
from radar_tracker.track import RadarTrack3D
from radar_tracker.tracker import MultiTargetTracker
from radar_tracker.visualize_tracks import (
    create_certainty_visualization,
    create_interactive_html_map,
    create_visualization,
)

__all__ = [
    "AVAILABLE_PIPELINES",
    "TrackerConfig",
    "RadarTrack3D",
    "MultiTargetTracker",
    "prepare_plots_dataframe",
    "group_into_scans",
    "estimate_track_kinematics",
    "evaluate_kinematic_pipelines",
    "angular_diff_deg",
    "create_visualization",
    "create_certainty_visualization",
    "create_interactive_html_map",
]
