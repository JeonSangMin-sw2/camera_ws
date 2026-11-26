"""A pick of scratch/geometry_pose_search.py with the hands brought closer together (2026-11-26, user: the hands are
too far apart to compare by eye). The right hand keeps its link_torso_5 x, z and orientation, y moves to the target;
the left arm is the mirror. Damped least squares on the right arm joints, URDF from the simulator (127.0.0.1, model
m, v1.2), zero offsets; nothing moves. Writes a poses file with that one pick for scratch/check_path_views.py.
usage: python scratch/gather_pose_ik.py <poses json> <pick share> <target right-hand y mm> <out json>
2026-11-26: the first gathered pose (y -100) put the two marker plates into contact on the real robot; the link check
did not include the brackets and plates. The marker plate clearance (80 mm plate + the bracket from the flange) is now
reported, and the run refuses a pose under MARKER_CLEAR_MIN_MM.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation

MIRROR_SIGN = np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
MARKER_CLEAR_MIN_MM = 30.0
PLATE_HALF_MM = 45.0           # 80 mm plate, 5 mm margin for its holder
src, share, target_y, out = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

_setting = yaml.safe_load((ROOT / "config" / "setting.yaml").read_text(encoding="utf-8"))
BRACKET = {s: BaseCalibrator.make_transform(_setting["marker"][f"Tf_to_marker_{s}_v12"]) for s in ("right", "left")}


def marker_points(T_ee, side):
    """Points (m, link_torso_5) on one marker plate (a grid over +-PLATE_HALF_MM) and along its bracket."""
    T_m = T_ee @ BRACKET[side]
    g = np.linspace(-PLATE_HALF_MM, PLATE_HALF_MM, 7) / 1000.0
    plate = [T_m @ np.array([a, b, 0.0, 1.0]) for a in g for b in g]
    bracket = [T_ee[:, 3] + s * (T_m[:, 3] - T_ee[:, 3]) for s in np.linspace(0.0, 1.0, 6)]
    return np.array([p[:3] for p in plate + bracket])


def marker_clearance_mm(T):
    pr, pl = marker_points(T["ee_right"], "right"), marker_points(T["ee_left"], "left")
    return float(np.min(np.linalg.norm(pr[:, None, :] - pl[None, :, :], axis=2)) * 1000.0)
poses = json.loads(src.read_text(encoding="utf-8"))
pick = next(p for p in poses["picks"] if p["share"] == share)

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
names = list(model.robot_joint_names)
r_idx, l_idx = list(model.right_arm_idx), list(model.left_arm_idx)
links = (["link_torso_5", "ee_right"] + [f"link_right_arm_{k}" for k in range(7)] + [f"link_left_arm_{k}" for k in range(7)]
         + ["ee_left"])
state = dyn.make_state(links, names)
q_base = np.zeros(model.robot_dof)
q_base[list(model.torso_idx)] = np.radians([0, 30, -60, 30, 0, 0])


def fk(right_deg):
    q = q_base.copy()
    q[r_idx] = np.radians(right_deg)
    q[l_idx] = np.radians(MIRROR_SIGN * right_deg)
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    return {name: np.asarray(dyn.compute_transformation(state, 0, i)) for i, name in enumerate(links) if i}


def error(right_deg, T_goal):
    T = fk(right_deg)["ee_right"]
    dp = (T_goal[:3, 3] - T[:3, 3]) * 1000.0
    dr = Rotation.from_matrix(T_goal[:3, :3] @ T[:3, :3].T).as_rotvec(degrees=True)
    return np.concatenate([dp, dr])


q = np.array(pick["right_arm_deg"], dtype=float)
T_goal = fk(q)["ee_right"].copy()
T_goal[1, 3] = target_y / 1000.0
for _ in range(100):
    e = error(q, T_goal)
    if np.linalg.norm(e) < 1e-3:
        break
    J = np.zeros((6, 7))
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = 1e-3
        J[:, j] = (error(q, T_goal) - error(q + dq, T_goal)) / 1e-3
    q += np.linalg.solve(J.T @ J + 1e-2 * np.eye(7), J.T @ e)
T = fk(q)
pr = [T[f"link_right_arm_{k}"][:3, 3] for k in range(7)] + [T["ee_right"][:3, 3]]
pl = [T[f"link_left_arm_{k}"][:3, 3] for k in range(7)] + [T["ee_left"][:3, 3]]
sep = min(np.linalg.norm(a - b) for a in pr for b in pl) * 1000
print(f"pick {share}: right arm {np.round(pick['right_arm_deg'], 2).tolist()} -> {np.round(q, 2).tolist()}")
clear = marker_clearance_mm(T)
print(f"right hand {np.round(fk(np.array(pick['right_arm_deg']))['ee_right'][:3, 3] * 1000, 1)} -> {np.round(T['ee_right'][:3, 3] * 1000, 1)} mm, "
      f"residual {np.round(error(q, T_goal), 4)}, link separation {sep:.0f} mm, marker plate clearance {clear:.0f} mm")
if clear < MARKER_CLEAR_MIN_MM:
    sys.exit(f"[STOP] marker plates only {clear:.0f} mm apart (< {MARKER_CLEAR_MIN_MM:.0f} mm); widen the target y.")
gathered = dict(pick, share=share + 0.5, right_arm_deg=np.round(q, 3).tolist(), link_sep_mm=round(sep, 1),
                right_ee_mm=np.round(T["ee_right"][:3, 3] * 1000, 1).tolist())
out.write_text(json.dumps(dict(poses, picks=[gathered]), indent=1), encoding="utf-8")
print("written", out)
