"""How noisy is the v1.3 bracket yaw, and would the v1.2-style axis mapping steady it?

The v1.3 bracket orientation (MarkerCalibrator.compute_marker_bracket_from_orthogonal_sweeps) takes
the J6 and J5 axes in the marker frame from extract_axis_from_rotations: every frame's rotation
relative to the MIDDLE frame, so that one frame's orientation noise enters every estimate. v1.2 and
the v1.3 J6 stage instead map the circle-fit axis (camera frame, from marker positions) into the
marker frame with every frame (axis_in_marker_frame_robust). Both are computed here from the recorded
bracket sweeps (last pass) and the yaw offset is compared with the simulator truth (bracket_rpy).

Reads <run>/result/result_txt/sweep_points_<arm>_marker_axis_<5|6>.txt. Offline.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_bracket_yaw_estimators.py <run folder> [<run folder> ...]
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R_scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402
from core.calibration.MarkerCalibrator import MarkerCalibrator  # noqa: E402

mc = MarkerCalibrator.__new__(MarkerCalibrator)


def last_block(path):
    angles, poses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "NEW ITERATION" in line:
            angles, poses = [], []
        elif line and not line.startswith("#"):
            v = [float(x) for x in line.split(",") if x.strip()]
            if len(v) >= 26:
                angles.append(v[0])
                poses.append(np.array(v[10:26]).reshape(4, 4))
    return angles, poses


def yaw_offset(n6, n5, nominal_rpy):
    """The construction of compute_marker_bracket_from_orthogonal_sweeps; yaw minus the design yaw."""
    x_col = n6 / np.linalg.norm(n6)
    y_col = n5 - np.dot(n5, x_col) * x_col
    y_col /= np.linalg.norm(y_col)
    R_m_ee = np.column_stack((x_col, y_col, np.cross(x_col, y_col)))
    yaw = R_scipy.from_matrix(R_m_ee.T).as_euler('ZYX', degrees=True)[0]
    return (yaw - nominal_rpy[2] + 180.0) % 360.0 - 180.0


for run in map(Path, sys.argv[1:]):
    truth = yaml.safe_load((run / "config" / "simulation.yaml").read_text(encoding="utf-8"))
    noise = truth.get("orientation_noise_std_deg", 0.0)
    print(f"{run.name} (orientation noise {noise} deg)")
    for side in ("right", "left"):
        nominal_rpy = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES["1.3"][side][3:6]
        R_ee_m_ideal = R_scipy.from_euler('ZYX', nominal_rpy[::-1], degrees=True).as_matrix()
        ideal = {6: R_ee_m_ideal.T @ [1.0, 0.0, 0.0], 5: R_ee_m_ideal.T @ [0.0, 1.0, 0.0]}
        rot_only, circle_fit = {}, {}
        for k in (6, 5):
            angles, poses = last_block(run / "result" / "result_txt" / f"sweep_points_{side}_marker_axis_{k}.txt")
            rot_only[k] = mc.extract_axis_from_rotations(poses, ideal[k])
            n_cam = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, robust=True)["axis_opt"]
            circle_fit[k] = MarkerCalibrator.axis_in_marker_frame_robust(n_cam, poses, ideal[k])
        true_yaw = truth["offsets"][side]["bracket_rpy"][2]
        a, b = yaw_offset(rot_only[6], rot_only[5], nominal_rpy), yaw_offset(circle_fit[6], circle_fit[5], nominal_rpy)
        print(f"  {side:5s} yaw offset: rotation-only {a:+.3f} (error {a - true_yaw:+.3f}) | "
              f"circle-fit axes {b:+.3f} (error {b - true_yaw:+.3f}) | truth {true_yaw:+.3f} deg")
