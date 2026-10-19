"""Where the marker measurement and the model disagree in the 2026-10-08 Step 2 data (offline).

For a Step 2 solution, every one of the 64 poses gives the marker pose the model predicts and the one
the camera measured. This prints, per arm, the leftover rotation about the J6 axis (the direction a J6
error or a turned marker plate shows up in) and the leftover position, with their spread over the poses
and how they follow the marker's place in the image. A constant rotation points at the plate or at J6;
one that follows the image position points at the camera's orientation measurement.
Solutions compared: as run (J6 from the marker orientation), J6 from the marker position, J6 at the
reset baseline -- the result files written by j6_both_arms_20261008.py.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step2_marker_residuals_20261008.py
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import step15_rerun_20261008 as step15
from core.calibration.CalibratorBase import BaseCalibrator

step2 = step15.step2_rerun
out = sys.__stdout__
robot, model, dynamics = step2.robot, step2.robot.model(), step2.robot.get_dynamics()
ARM_INDEX = {"right": list(model.right_arm_idx)[:7], "left": list(model.left_arm_idx)[:7]}
HEAD_INDEX = list(model.head_idx)
JOINT_COUNT = len(model.robot_joint_names)
J6_AXIS_IN_FLANGE = np.array([0.0, 0.0, 1.0])


def residuals(result):
    camera = BaseCalibrator.make_transform(list(result["mount_to_cam_new"]))
    head_offset = np.radians(result["head_joint_offset_deg"])
    rows = {"right": [], "left": []}
    for q_arm, q_head, measured in zip(step2.q_arm, step2.q_head, step2.T_meas):
        q = np.zeros(JOINT_COUNT)
        q[HEAD_INDEX] = np.asarray(q_head) + head_offset
        for k, side in enumerate(("right", "left")):
            q[ARM_INDEX[side]] = q_arm[7 * k: 7 * k + 7] + np.radians(result[f"{side}_arm_joint_offset_deg"])
        base_to_camera = BaseCalibrator.compute_fk(robot, dynamics, q, "link_head_2", "link_torso_5") @ camera
        for k, side in enumerate(("right", "left")):
            bracket = BaseCalibrator.make_transform(list(result["ee_to_marker_used"][side]))
            flange = BaseCalibrator.compute_fk(robot, dynamics, q, f"ee_{side}", "link_torso_5")
            predicted = np.linalg.inv(base_to_camera) @ flange @ bracket
            delta = np.linalg.inv(predicted) @ measured[k]
            rotation = Rotation.from_matrix(delta[:3, :3]).as_rotvec(degrees=True)
            axis_in_marker = bracket[:3, :3].T @ J6_AXIS_IN_FLANGE
            position = measured[k][:3, 3]
            rows[side].append([rotation @ axis_in_marker, np.linalg.norm(rotation), np.linalg.norm(delta[:3, 3]) * 1000.0,
                               np.degrees(np.arctan2(position[0], position[2])),
                               np.degrees(np.arctan2(position[1], position[2])), np.degrees(q_arm[7 * k + 5])])
    return {side: np.array(values) for side, values in rows.items()}


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


print("\nper arm over the 64 poses: rotation left over about the J6 axis (deg), total rotation and position left over,",
      file=out)
print("and how the J6-axis rotation follows the marker's image position (horizontal / vertical, deg) and the J5 angle",
      file=out)
for tag, name in (("as run (J6 orientation)", "result_j6both_as_run.json"),
                  ("J6 position", "result_j6both_J6_position.json"),
                  ("J6 baseline", "result_j6both_J6_baseline.json")):
    result = json.loads((step2.OUT / name).read_text(encoding="utf-8"))
    values = residuals(result)
    print(f"\n{tag}", file=out)
    for side in ("right", "left"):
        v = values[side]
        slope = np.polyfit(v[:, 3], v[:, 0], 1)[0]
        print(f"  {side:5s}: about J6 axis mean {v[:, 0].mean():+.3f} std {v[:, 0].std():.3f} "
              f"(range {v[:, 0].min():+.2f}..{v[:, 0].max():+.2f}) | total rot rms {np.sqrt(np.mean(v[:, 1] ** 2)):.3f} deg | "
              f"pos rms {np.sqrt(np.mean(v[:, 2] ** 2)):.3f} mm | image h {v[:, 3].min():+.1f}..{v[:, 3].max():+.1f} deg, "
              f"corr h {correlation(v[:, 0], v[:, 3]):+.2f} (slope {slope:+.3f} deg/deg), corr v {correlation(v[:, 0], v[:, 4]):+.2f}, "
              f"J5 {v[:, 5].min():+.1f}..{v[:, 5].max():+.1f} deg, corr J5 {correlation(v[:, 0], v[:, 5]):+.2f}", file=out)
