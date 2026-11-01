"""
Modular Kinematic Estimation Subpackage (``radar_tracker.kinematics``).

Contains the runtime estimator alongside benchmark filters:
- ``kinematics_base.py``: Shared angular geometry, circular statistics, and raw finite differences.
- ``hampel.py``: Stage 1 non-linear Median & Hampel (MAD) outlier rejection.
- ``directional.py``: Stage 2 bilateral direction-compatible vector smoothing with 12-deg gate.
- ``estimator.py``: Runtime estimator (``estimate_track_kinematics``) — executes the default
  ``hampel_directional`` pipeline when ``./tracker.sh plots.json tracks.json`` runs.
- ``pipeline_evaluator.py``: Benchmark evaluator (``evaluate_kinematic_pipelines``) used in the Jupyter
  Notebook and test suite to compare 11 classic and hybrid candidate pipelines.
- Alternative benchmark filters: ``moving_average.py``, ``alpha_beta.py``, ``savgol.py``, ``kalman_rts.py``.
"""

from radar_tracker.kinematics.alpha_beta import (
    ALPHA_BETA_ALPHA,
    ALPHA_BETA_BETA,
    apply_alpha_beta_filter,
)
from radar_tracker.kinematics.directional import (
    MAX_COMPATIBLE_HEADING_DIFF_DEG,
    SMOOTH_NEIGHBOR_RADIUS,
    apply_directional_smoothing,
)
from radar_tracker.kinematics.estimator import (
    DEFAULT_PIPELINE_METHOD,
    estimate_track_kinematics,
)
from radar_tracker.kinematics.hampel import (
    HAMPEL_MAD_SCALE,
    HAMPEL_N_SIGMA,
    MEDIAN_WINDOW_SIZE,
    SPEED_OUTLIER_REL_TOL,
    reject_waypoint_outliers,
)
from radar_tracker.kinematics.kalman_rts import (
    HUBER_INNOVATION_CLIP_M,
    RTS_ACCEL_SPECTRAL_DENSITY,
    RTS_POS_STD_M,
    apply_kalman_rts_positions,
)
from radar_tracker.kinematics.kinematics_base import (
    FULL_CIRCLE_DEG,
    HALF_CIRCLE_DEG,
    MIN_MOVING_SPEED_MPS,
    MIN_TIME_STEP_S,
    OUTPUT_DECIMALS,
    angular_diff_deg,
    compute_raw_velocity_vectors,
    vectors_to_heading_deg,
)
from radar_tracker.kinematics.moving_average import (
    MOVING_AVG_WINDOW,
    apply_moving_average,
)
from radar_tracker.kinematics.pipeline_evaluator import (
    AVAILABLE_PIPELINES,
    evaluate_kinematic_pipelines,
)
from radar_tracker.kinematics.savgol import (
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
    apply_savgol_filter,
)

# Backward-compatible aliases for internal helpers
_vectors_to_heading_deg = vectors_to_heading_deg
_reject_waypoint_outliers = reject_waypoint_outliers
_apply_directional_smoothing = apply_directional_smoothing
_apply_alpha_beta_filter = apply_alpha_beta_filter
_apply_kalman_rts_positions = apply_kalman_rts_positions

__all__ = [
    "AVAILABLE_PIPELINES",
    "DEFAULT_PIPELINE_METHOD",
    "FULL_CIRCLE_DEG",
    "HALF_CIRCLE_DEG",
    "MIN_TIME_STEP_S",
    "MIN_MOVING_SPEED_MPS",
    "OUTPUT_DECIMALS",
    "MEDIAN_WINDOW_SIZE",
    "SPEED_OUTLIER_REL_TOL",
    "HAMPEL_MAD_SCALE",
    "HAMPEL_N_SIGMA",
    "SMOOTH_NEIGHBOR_RADIUS",
    "MAX_COMPATIBLE_HEADING_DIFF_DEG",
    "MOVING_AVG_WINDOW",
    "ALPHA_BETA_ALPHA",
    "ALPHA_BETA_BETA",
    "SAVGOL_WINDOW",
    "SAVGOL_POLYORDER",
    "RTS_POS_STD_M",
    "RTS_ACCEL_SPECTRAL_DENSITY",
    "HUBER_INNOVATION_CLIP_M",
    "angular_diff_deg",
    "vectors_to_heading_deg",
    "compute_raw_velocity_vectors",
    "reject_waypoint_outliers",
    "apply_moving_average",
    "apply_alpha_beta_filter",
    "apply_savgol_filter",
    "apply_kalman_rts_positions",
    "apply_directional_smoothing",
    "estimate_track_kinematics",
    "evaluate_kinematic_pipelines",
]
