"""Search a low verification pose the head camera sees head-on: the v1.2 check pose's J0-J3 kept (the arm stays
low, as at the check pose), J4/J5/J6 and the head tilt changed so both markers face the camera within the Step 2
sample conditions (marker <= 30 deg from facing, <= 330 mm away, >= 60 px inside the image). Arms mirrored.
Simulator FK only (127.0.0.1, model m, v1.2), nominal camera and brackets, zero offsets; nothing moves.
usage: python scratch/low_view_pose_search.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rby1_sdk as rby  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

MIRROR_SIGN = np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
FACE_MAX_DEG, RANGE_MAX_MM, MARGIN_PX, W, H = 30.0, 330.0, 60.0, 1280, 720
J_LIMIT_MARGIN_DEG = 5.0

poses = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v1.2"]
check_r = np.array(poses["check_calib"]["right_arm"], dtype=float)
intr = yaml.safe_load((ROOT / "config" / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
K = np.asarray(intr["camera_matrix"], dtype=float)
T_mount_cam = BaseCalibrator.make_transform([0.0433, 0.009, 0.063, -90.0, 0.0, -90.0])
bracket = {"right": BaseCalibrator.make_transform([0, -0.054, -0.048, 90, 0.0, 180]),
           "left": BaseCalibrator.make_transform([0, 0.054, -0.048, 90, 0.0, 0])}

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
names = list(model.robot_joint_names)
head_idx = [names.index("head_0"), names.index("head_1")]
state = dyn.make_state(["link_torso_5"], names)
lo = np.degrees(np.clip(dyn.get_limit_q_lower(state), -1e3, 1e3))
hi = np.degrees(np.clip(dyn.get_limit_q_upper(state), -1e3, 1e3))
r_idx, l_idx = list(model.right_arm_idx), list(model.left_arm_idx)


def evaluate(right_deg, tilt):
    q = np.zeros(model.robot_dof)
    q[list(model.torso_idx)] = np.radians([0, 30, -60, 30, 0, 0])
    q[head_idx] = np.radians([0.0, tilt])
    q[r_idx] = np.radians(right_deg)
    q[l_idx] = np.radians(MIRROR_SIGN * right_deg)
    T_cam = BaseCalibrator.compute_fk(robot, dyn, q, "link_head_2", "link_torso_5") @ T_mount_cam
    out = {}
    for side in ("right", "left"):
        T_ee = BaseCalibrator.compute_fk(robot, dyn, q, f"ee_{side}", "link_torso_5")
        T_m = np.linalg.inv(T_cam) @ T_ee @ bracket[side]
        p = T_m[:3, 3]
        if p[2] <= 0:
            return None
        u, v = (K @ (p / p[2]))[:2]
        face = np.degrees(np.arccos(np.clip(-T_m[:3, 2] @ (-p / np.linalg.norm(p)), -1, 1)))
        out[side] = dict(u=u, v=v, face=face, range_mm=np.linalg.norm(p) * 1000, ee_mm=T_ee[:3, 3] * 1000)
    return out


def within_limits(right_deg):
    for idx, angle in zip(r_idx, right_deg):
        if not lo[idx] + J_LIMIT_MARGIN_DEG <= angle <= hi[idx] - J_LIMIT_MARGIN_DEG:
            return False
    left = MIRROR_SIGN * right_deg
    return all(lo[idx] + J_LIMIT_MARGIN_DEG <= a <= hi[idx] - J_LIMIT_MARGIN_DEG for idx, a in zip(l_idx, left))


check_ee = evaluate(check_r, 30.0)["right"]["ee_mm"]
print("check pose right ee (link_torso_5, mm):", np.round(check_ee, 1))
found = []
for d4 in np.arange(-60, 61, 10.0):
    for d5 in np.arange(-60, 61, 10.0):
        for d6 in np.arange(-180, 181, 15.0):
            q_r = check_r + np.array([0, 0, 0, 0, d4, d5, d6])
            if not within_limits(q_r):
                continue
            for tilt in np.arange(20.0, 61.0, 5.0):
                v = evaluate(q_r, tilt)
                if v is None:
                    continue
                ok = all(v[s]["face"] <= FACE_MAX_DEG and v[s]["range_mm"] <= RANGE_MAX_MM
                         and MARGIN_PX <= v[s]["u"] <= W - MARGIN_PX and MARGIN_PX <= v[s]["v"] <= H - MARGIN_PX
                         for s in v)
                if ok:
                    face = max(v[s]["face"] for s in v)
                    ee_shift = np.linalg.norm(v["right"]["ee_mm"] - check_ee)
                    found.append((face, abs(d4) + abs(d5), d4, d5, d6, tilt, v, ee_shift))
print("poses meeting all criteria:", len(found))
for face, _, d4, d5, d6, tilt, v, ee_shift in sorted(found, key=lambda f: (f[0], f[1]))[:8]:
    print(f"  dJ4 {d4:+.0f} dJ5 {d5:+.0f} dJ6 {d6:+.0f} head tilt {tilt:.0f} | face max {face:.1f} deg | "
          f"range {max(v[s]['range_mm'] for s in v):.0f} mm | right uv ({v['right']['u']:.0f},{v['right']['v']:.0f}) "
          f"left uv ({v['left']['u']:.0f},{v['left']['v']:.0f}) | right ee {np.round(v['right']['ee_mm'], 0)} "
          f"(moved {ee_shift:.0f} mm from the check pose)")
robot.disconnect()
