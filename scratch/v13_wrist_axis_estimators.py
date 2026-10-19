"""How should v1.3's wrist axes be measured for the orthogonality (angle between axes) estimates?

v1.3 J5 (wrist_pitch_v13) is the angle, about J5, between the J6-sweep axis and the J4-sweep axis.
Today both axes are fitted circle normals from marker positions only; v1.2's J6 (and v1.3's J6)
instead map the axes through the measured marker orientation. In the simulator the orientation
noise is set to 0 (config/simulation.yaml), so the simulator cannot compare the two fairly.

1. Measure the per-frame noise of a real sweep (the 2026-10-08 17:50 v1.2 J6 bracket sweeps):
   position = circle-fit RMSE, orientation = spread of each frame rotated back by its encoder angle.
2. With that noise, draw synthetic v1.3 J6/J4 sweeps (radii from the v1.3 simulator) and compare
   the J5 scatter per sweep pair of: position circle normals (today), orientation rotation axes,
   for +-15 deg (today) and +-30 deg arcs.
Offline, nothing is commanded. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_wrist_axis_estimators.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

RUN = ROOT / "archive" / "runs" / "20261008_1750_newrobot_head" / "result_txt"
R6_MM, R4_MM, FRAMES, TRIALS = 127.0, 68.0, 240, 200
J4_AXIS, J5_AXIS, J6_AXIS = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])


def load_last_block(path):
    angles, poses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            angles, poses = [], []
        elif line and not line.startswith("#"):
            values = [float(v) for v in line.split(",") if v.strip()]
            if len(values) >= 26:
                angles.append(values[0])
                poses.append(np.array(values[10:26]).reshape(4, 4))
    return np.array(angles), poses


def real_noise(path):
    angles, poses = load_last_block(path)
    fit = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, list(angles), robust=True)
    axis = fit["axis_opt"] / np.linalg.norm(fit["axis_opt"])
    for sign in (1.0, -1.0):  # the encoder angle and the fitted axis may point opposite ways
        back = [Rotation.from_rotvec(-sign * np.radians(a) * axis) * Rotation.from_matrix(T[:3, :3])
                for a, T in zip(angles, poses)]
        mean = Rotation.concatenate(back).mean()
        spread = np.degrees(np.array([(mean.inv() * r).as_rotvec() for r in back]))
        if sign > 0 or np.linalg.norm(spread, axis=1).mean() < best[1]:
            best = (spread, np.linalg.norm(spread, axis=1).mean())
    spread = best[0]
    return len(angles), float(np.ptp(angles)), float(fit["rmse"]), spread.std(axis=0)


def sweep(axis, start, radius, half_arc, pos_mm, rot_deg, rng):
    angles = np.linspace(-half_arc, half_arc, FRAMES)
    side = np.cross(axis, start)
    poses, rotations = [], []
    for a in np.radians(angles):
        turn = Rotation.from_rotvec(a * axis)
        noise = Rotation.from_rotvec(np.radians(rng.normal(0.0, rot_deg, 3)))
        pose = np.eye(4)
        pose[:3, :3] = (noise * turn).as_matrix()
        pose[:3, 3] = (radius * (np.cos(a) * start + np.sin(a) * side) + rng.normal(0.0, pos_mm, 3)) / 1000.0
        poses.append(pose)
        rotations.append(noise * turn)
    fit = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, list(angles), axis_prior=axis, robust=True)
    position_axis = fit["axis_opt"] / np.linalg.norm(fit["axis_opt"])
    # Orientation: every frame's rotation relative to the middle one is (angle difference) x axis.
    middle = rotations[FRAMES // 2]
    steps = np.radians(angles - angles[FRAMES // 2])
    vectors = np.array([(r * middle.inv()).as_rotvec() for r in rotations])
    orientation_axis = (steps[:, None] * vectors).sum(axis=0) / (steps ** 2).sum()
    orientation_axis /= np.linalg.norm(orientation_axis)
    return [a * np.sign(a @ axis) for a in (position_axis, orientation_axis)]


def j5_angle(n6, n4):
    a = n4 - (n4 @ J5_AXIS) * J5_AXIS
    b = n6 - (n6 @ J5_AXIS) * J5_AXIS
    return np.degrees(np.arctan2(np.cross(a, b) @ J5_AXIS, a @ b)) - 90.0


def main():
    print("1. Per-frame noise of the real 2026-10-08 17:50 bracket sweeps (D405, v1.2)")
    measured = []
    for name in ("sweep_points_right_marker_axis_6.txt", "sweep_points_left_marker_axis_6.txt",
                 "sweep_points_right_marker_axis_5.txt", "sweep_points_left_marker_axis_5.txt"):
        frames, arc, rmse, rot_std = real_noise(RUN / name)
        measured.append((rmse, rot_std))
        print(f"   {name:40s} {frames:4d} frames, arc {arc:5.1f} deg | position RMSE {rmse:.3f} mm | "
              f"orientation spread per axis {np.round(rot_std, 3)} deg")
    pos_mm = float(np.median([m[0] for m in measured])) / np.sqrt(3.0)
    rot_deg = float(np.median([m[1].mean() for m in measured]))
    print(f"   -> synthetic noise: position {pos_mm:.3f} mm per axis, orientation {rot_deg:.3f} deg per axis\n")

    rng = np.random.default_rng(20261014)
    print(f"2. v1.3 J5 per sweep pair (J6 r {R6_MM:.0f} mm, J4 r {R4_MM:.0f} mm, {FRAMES} frames, {TRIALS} trials)")
    print("   half arc | J5 std, position normals (today) | J5 std, orientation axes | pairs for SE <= 0.06 deg (pos / ori)")
    for half_arc in (15.0, 30.0):
        values = {"position": [], "orientation": []}
        for _ in range(TRIALS):
            p4, o4 = sweep(J4_AXIS, J6_AXIS, R4_MM, half_arc, pos_mm, rot_deg, rng)
            p6, o6 = sweep(J6_AXIS, -J4_AXIS, R6_MM, half_arc, pos_mm, rot_deg, rng)
            values["position"].append(j5_angle(p6, p4))
            values["orientation"].append(j5_angle(o6, o4))
        std = {k: float(np.std(v, ddof=1)) for k, v in values.items()}
        needed = {k: max(2, int(np.ceil((s / 0.06) ** 2))) for k, s in std.items()}
        print(f"   {half_arc:8.1f} | {std['position']:32.3f} | {std['orientation']:23.3f} | "
              f"{needed['position']} / {needed['orientation']}")


if __name__ == "__main__":
    main()
