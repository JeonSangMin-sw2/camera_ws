"""Split the saved check-pose view (result JSON check_pose_view) into a rotation common to both markers and the rest.

A camera / head model error at the check-view head tilt turns both measured markers by the same small rotation w
(camera frame) and moves each by about w x p. Fitting one w to both markers' rotation errors and taking it out
leaves what the two arms really differ by. FK from the running simulator of the same model (127.0.0.1, model m,
v1.2); nothing is written. usage: python scratch/check_view_decompose.py <result json> [offsets json] [scale_r,scale_l]
offsets json: another Step 2 result (e.g. an offline re-solve) whose offsets and camera are used instead.
scale: the measured marker translations times these (as step2_offline_variants.py marker_scale).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rby1_sdk as rby  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
view = result["check_pose_view"]
if len(sys.argv) > 2:       # judge the same measurement with another solution's offsets and camera
    other = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    result = {**result, **{k: other[k] for k in ("right_arm_joint_offset_deg", "left_arm_joint_offset_deg",
                                                  "head_joint_offset_deg")}}
    view = {**view, "camera": {**view["camera"], "transform": other["mount_to_cam_new"]}}
robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
names = list(model.robot_joint_names)
arm_idx = list(model.right_arm_idx) + list(model.left_arm_idx)
head_idx = [names.index("head_0"), names.index("head_1")]

q = np.zeros(model.robot_dof)
q[arm_idx] = np.asarray(view["q_arm"]) + np.radians(np.r_[result["right_arm_joint_offset_deg"],
                                                           result["left_arm_joint_offset_deg"]])
q[head_idx] = np.asarray(view["q_head"]) + np.radians(result["head_joint_offset_deg"])
T_cam = BaseCalibrator.compute_fk(robot, dyn, q, view["camera"]["link"], "link_torso_5") \
    @ BaseCalibrator.make_transform(view["camera"]["transform"])
model_T, meas_T = {}, {}
for side in ("right", "left"):
    T_ee = BaseCalibrator.compute_fk(robot, dyn, q, f"ee_{side}", "link_torso_5")
    model_T[side] = np.linalg.inv(T_cam) @ T_ee @ BaseCalibrator.make_transform(view["brackets"][side])
    meas_T[side] = np.array(view["markers"][side], dtype=float)
    if len(sys.argv) > 3:
        meas_T[side][:3, 3] *= float(sys.argv[3].split(",")[0 if side == "right" else 1])
robot.disconnect()

print("camera frame (x right, y down, z forward), measured minus model:")
rot_err, pos_err = {}, {}
for side in ("right", "left"):
    rot_err[side] = R.from_matrix(meas_T[side][:3, :3] @ model_T[side][:3, :3].T).as_rotvec(degrees=True)
    pos_err[side] = (meas_T[side][:3, 3] - model_T[side][:3, 3]) * 1000
    print(f"  {side:5s} rotation {np.round(rot_err[side], 3)} deg | position {np.round(pos_err[side], 2)} mm | "
          f"range {np.linalg.norm(model_T[side][:3, 3]) * 1000:.0f} mm")

w = np.radians((rot_err["right"] + rot_err["left"]) / 2)            # one camera rotation for both
print(f"common rotation (mean of the two) {np.round(np.degrees(w), 3)} deg; "
      f"left minus right {np.round(rot_err['left'] - rot_err['right'], 3)} deg")
rest = {}
for side in ("right", "left"):
    p = model_T[side][:3, 3] * 1000
    rest[side] = pos_err[side] - np.cross(w, p)
    print(f"  {side:5s} position left after the common rotation: {np.round(rest[side], 2)} mm")

# The left/right flange gap in link_torso_5 (as the verification reports it), before and after the common rotation.
MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])
Rw = np.eye(4)
Rw[:3, :3] = R.from_rotvec(w).as_matrix()
for title, fix in (("as measured", np.eye(4)), ("common rotation taken out", np.linalg.inv(Rw))):
    flange = {s: T_cam @ fix @ meas_T[s] @ np.linalg.inv(BaseCalibrator.make_transform(view["brackets"][s]))
              for s in ("right", "left")}
    d = ((MIRROR @ flange["left"] @ MIRROR)[:3, 3] - flange["right"][:3, 3]) * 1000
    print(f"left minus mirrored right flange, {title}: x {d[0]:+.2f}, y {d[1]:+.2f}, z {d[2]:+.2f} mm "
          f"-> {np.linalg.norm(d):.2f} mm")

# The same errors in link_torso_5 (x forward, y left, z up): what each hand did, as the operator sees it.
R_t5_cam = T_cam[:3, :3]
for side in ("right", "left"):
    print(f"  {side:5s} in link_torso_5: rotation {np.round(R_t5_cam @ rot_err[side], 3)} deg | "
          f"position {np.round(R_t5_cam @ pos_err[side], 2)} mm")
