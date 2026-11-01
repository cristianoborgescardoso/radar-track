"""
Configuration parameters and .properties loader for the Multi-Target Radar Tracker.
"""

import configparser
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrackerConfig:
    """Immutable configuration parameters for 3D Multi-Target Tracking.

    Can be instantiated directly in Python or loaded from a documented ``.properties``
    file via :meth:`TrackerConfig.from_properties`.
    """

    # Intra-scan dwell grouping window (s): >6x above max intra-sweep spread (22 ms), well below 1.0 s scan period.
    scan_window_s: float = 0.15

    # Nominal radar update period (s): 1.0 s for a 1 Hz (60 RPM) surveillance/C-UAS radar.
    nominal_scan_period_s: float = 1.0

    # Maximum coasting duration (s) before track termination; bridges 2-4 s detection dropouts without fragmentation.
    max_coast_s: float = 6.0

    # Minimum associated hits for confirmation (M-out-of-N rule, Blackman Ch. 6; 4 hits verify 3 collinear segments).
    min_confirmed_hits: int = 4

    # Initial Wald SPRT certainty score (0-100 scale) assigned on 1st plot; below confirmation threshold (50.0).
    initial_certainty: float = 30.0

    # Certainty increment (+20.0) per associated scan; crosses Bayes MAP threshold (50.0) on 2nd hit.
    certainty_hit_increment: float = 20.0

    # Certainty penalty (-7.0) per missed scan (SPRT 20:7 ratio for Pd=0.85, Pfa=0.0037).
    certainty_miss_penalty: float = 7.0

    # Upper saturation bound on the track certainty score.
    max_certainty: float = 100.0

    # Minimum certainty score required to promote a Tentative track to Confirmed status (Bayes MAP threshold = 50.0).
    confirmation_certainty_threshold: float = 50.0

    # Maximum physical horizontal speed (m/s) for tactical UAVs (Groups 1-3) and helicopters (~288 km/h).
    max_speed_xy_mps: float = 80.0

    # Maximum physical vertical climb/descent speed (m/s) (~3,900 ft/min).
    max_speed_z_mps: float = 20.0

    # Tighter horizontal speed gate (m/s) on 1st-to-2nd hit transition to suppress false-alarm pair initiations.
    init_speed_xy_mps: float = 55.0

    # Tighter vertical speed gate (m/s) on 1st-to-2nd hit transition.
    init_speed_z_mps: float = 10.0

    # 3D Mahalanobis innovation gate (sigma); expanded from 3.76 (99.7% chi^2_3) to 9.0 to absorb 90-deg turn lag.
    mahalanobis_gate: float = 9.0

    # Base 3D Euclidean prediction error gate (m); covers worst-case 90-deg turn chord at 30 m/s in 1 s (42.4 m).
    base_pred_err_m: float = 45.0

    # Linear expansion rate (m/s) of the Euclidean prediction gate per second of coasting.
    pred_err_slope_mps: float = 12.0

    # Association cost penalty added to unestablished tracks (hits < 3) so confirmed tracks have priority.
    tentative_cost_penalty: float = 15.0

    # Dummy assignment cost in the augmented Hungarian matrix (>> valid gate cost ~75, << invalid sentinel 1e6).
    unassigned_cost_threshold: float = 500.0

    # Horizontal measurement position standard deviation (m) in Kalman matrix R.
    sigma_pos_xy_m: float = 2.0

    # Vertical measurement altitude standard deviation (m) in Kalman matrix R.
    sigma_pos_z_m: float = 2.0

    # Horizontal continuous white-noise acceleration std dev (m/s^2, ~1.2 G) in process matrix Q(dt).
    sigma_accel_xy: float = 12.0

    # Vertical continuous white-noise acceleration std dev (m/s^2, ~0.6 G) in process matrix Q(dt).
    sigma_accel_z: float = 6.0

    # Post-processing kinematic estimator pipeline ("hampel_directional" is the default pipeline).
    kinematic_filter_method: str = "hampel_directional"

    # Stage 1 Hampel/median window length (scans); 9 scans span full 6-10 s inter-waypoint segments.
    median_window_size: int = 9

    # Stage 1 relative speed deviation tolerance (12%); separates normal radar jitter from waypoint drops (>=15%).
    speed_outlier_rel_tol: float = 0.12

    # Stage 2 bilateral smoothing radius (1 scan = 3-point local window [i-1, i, i+1]).
    smooth_neighbor_radius: int = 1

    # Stage 2 directional compatibility gate (12.0 deg = 2-sigma angular radar noise, below smallest 18 deg turn).
    max_heading_diff_deg: float = 12.0

    # Decimal rounding precision for exported speed (m/s) and heading (deg) values.
    output_decimals: int = 2

    @classmethod
    def from_properties(cls, path: Optional[str] = None) -> "TrackerConfig":
        """Load TrackerConfig from a ``.properties`` file (supports both sectioned and flat keys).

        If ``path`` is None, looks for ``tracker.properties`` in the project root directory
        and returns default values if the file does not exist.

        Args:
            path: Path to a custom ``.properties`` file, or None to use ``tracker.properties``.

        Returns:
            Populated :class:`TrackerConfig` instance.
        """
        if path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tracker.properties",
            )
            if not os.path.isfile(default_path):
                return cls()
            target_path = default_path
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Properties file not found: {path}")
            target_path = path

        with open(target_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Support both INI-style [section] properties and flat Java-style key=value properties
        has_section = any(
            line.strip().startswith("[") and line.strip().endswith("]")
            for line in raw_text.splitlines()
            if line.strip() and not line.strip().startswith(("#", ";"))
        )
        parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        if has_section:
            parser.read_string(raw_text)
            raw_items: Dict[str, str] = {}
            for section in parser.sections():
                for k, v in parser.items(section):
                    raw_items[k.strip()] = v.strip()
        else:
            parser.read_string("[default]\n" + raw_text)
            raw_items = {k.strip(): v.strip() for k, v in parser.items("default")}

        field_types = {f.name: f.type for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, str_val in raw_items.items():
            if key not in field_types:
                raise ValueError(
                    f"Unknown property '{key}' in {target_path}. "
                    f"Valid properties: {sorted(field_types.keys())}"
                )
            ftype = field_types[key]
            if ftype in (int, "int"):
                kwargs[key] = int(str_val)
            elif ftype in (float, "float"):
                kwargs[key] = float(str_val)
            else:
                kwargs[key] = str_val.strip("\"'")

        return cls(**kwargs)
