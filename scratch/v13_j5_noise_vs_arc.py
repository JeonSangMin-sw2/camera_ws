"""Per-sweep noise of the v1.3 J5 estimate against the sweep arc (synthetic, no robot).

The v1.3 J5 offset is the angle between the J6-sweep circle axis and the J4-sweep circle axis
(JointCalibrator.compute_calibration_results, wrist_pitch_v13). This draws both circles with the
radii seen in the simulator (J6 127 mm, J4 68 mm), adds independent position noise per frame,
fits each with the production circle fit and reports the scatter of
  - the angle between the two fitted axes about the J5 axis (what the J5 estimate is made of), and
  - the fitted J4 radius converted to a J5 angle (dr4/dJ5 = r6, the radius alternative).
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_j5_noise_vs_arc.py [noise_mm] [trials]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

R6_MM, R4_MM, FRAMES = 127.0, 68.0, 520
J4_AXIS, J5_AXIS, J6_AXIS = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])


def sweep(axis, start, radius, half_arc_deg, noise_mm, rng):
    """Marker poses of one sweep: the marker starts at `start` (unit, normal to the axis) and
    turns about `axis` through +-half_arc_deg. Orientation is not used by the fit."""
    angles = np.linspace(-half_arc_deg, half_arc_deg, FRAMES)
    side = np.cross(axis, start)
    poses = []
    for angle in np.radians(angles):
        pose = np.eye(4)
        pose[:3, 3] = (radius * (np.cos(angle) * start + np.sin(angle) * side) + rng.normal(0.0, noise_mm, 3)) / 1000.0
        poses.append(pose)
    fit = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, list(angles), axis_prior=axis, robust=True)
    normal = fit["axis_opt"] / np.linalg.norm(fit["axis_opt"])
    return normal * np.sign(normal @ axis), fit["radius"]


def j5_angle(n6, n4):
    """Signed angle from the J4 axis to the J6 axis about the J5 axis, minus the nominal 90 deg."""
    a = n4 - (n4 @ J5_AXIS) * J5_AXIS
    b = n6 - (n6 @ J5_AXIS) * J5_AXIS
    return np.degrees(np.arctan2(np.cross(a, b) @ J5_AXIS, a @ b)) - 90.0


def main(argv):
    noise_mm = float(argv[0]) if argv else 0.1
    trials = int(argv[1]) if len(argv) > 1 else 30
    rng = np.random.default_rng(20261006)
    print(f"position noise {noise_mm} mm per axis and frame, {FRAMES} frames per sweep, {trials} trials per arc")
    print(" half arc (deg) | axis-angle J5 estimate, std (deg) | sweeps for SE <= 0.06 deg | radius J5 estimate, std (deg)")
    for half_arc in (15.0, 20.0, 22.5, 30.0, 40.0):
        angle_values, radius_values = [], []
        for _ in range(trials):
            # The marker sits on the J6 axis side of the wrist point for the J4 sweep (start +x)
            # and below the J6 axis for the J6 sweep (start -z), as on the v1.3 bracket.
            n4, r4 = sweep(J4_AXIS, J6_AXIS, R4_MM, half_arc, noise_mm, rng)
            n6, _ = sweep(J6_AXIS, -J4_AXIS, R6_MM, half_arc, noise_mm, rng)
            angle_values.append(j5_angle(n6, n4))
            radius_values.append(np.degrees((r4 - R4_MM) / R6_MM))
        angle_std = float(np.std(angle_values, ddof=1))
        needed = max(2, int(np.ceil((angle_std / 0.06) ** 2)))
        print(f" {half_arc:14.1f} | {angle_std:33.3f} | {needed:25d} | {float(np.std(radius_values, ddof=1)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
