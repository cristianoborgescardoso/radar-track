"""
Modular unit, synthetic, and integration test suite for the ``radar_tracker`` package.

Test modules:
- ``test_config.py``: ``TrackerConfig`` immutability and ``.properties`` parser tests.
- ``test_preprocess.py``: Geodetic-to-ENU projection and temporal scan clustering tests.
- ``test_kinematics.py``: Unit tests for the 11 kinematic filtering pipelines and circular statistics.
- ``test_track_kalman.py``: ``RadarTrack3D`` 6D Kalman filter, Mahalanobis gating, RTS smoother, and NIS consistency tests.
- ``test_tracker_synthetic.py``: Synthetic multi-target stress tests (3D crossing, coasting, certainty lifecycle, exports, visualizers).
- ``test_validation_utils.py``: Fortem R20 external validation and RTK GPS interpolation utility tests.
- ``test_integration.py``: End-to-end integration tests on the benchmark ``plots.json`` dataset.
"""
