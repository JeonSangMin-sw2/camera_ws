"""How far the hands would land from the stepwise result at the check pose if the one-shot offsets were applied.

The stepwise offsets put both brackets flush at check_calib on the real robot (paper Fig. 4), so they are taken as
the truth here: with the one-shot offsets applied, each joint ends up off by (stepwise - one-shot). Reports the
per-hand ee displacement and the change of the right-left relative position, link_torso_5 frame, v1.2 URDF.
Model estimate only; the one-shot offsets were never applied to the robot.

usage: .venv\\Scripts\\python.exe -X utf8 scratch\\one_shot_check_pose.py <summary.json from one_shot_osqp_compare.py>
"""
import json
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rb
import yaml

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "archive" / "reports" / "step2_rerun_20261008" / "model_m_v1.2.urdf"
POSE = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))

robot = rb.dynamics.Robot(rb.dynamics.load_robot_from_urdf(str(URDF), "base"))
joints = list(robot.get_joint_names())
idx = {s: [joints.index(f"{s}_arm_{i}") for i in range(7)] for s in ("right", "left")}
state = robot.make_state(["link_torso_5", "ee_right", "ee_left"], joints)
check = next(v["check_calib"] for v in POSE.values() if isinstance(v, dict) and "check_calib" in v)


def hands(err_deg):
    q = np.zeros(len(joints))
    for k, side in enumerate(("right", "left")):
        q[idx[side]] = np.deg2rad(np.array(check[f"{side}_arm"]) + err_deg[7 * k:7 * k + 7])
    state.set_q(q)
    robot.compute_forward_kinematics(state)
    return [robot.compute_transformation(state, 0, j)[:3, 3] * 1000.0 for j in (1, 2)]


summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
R0, L0 = hands(np.zeros(14))
rows = []
for ts, r in summary.items():
    err = np.array(r["stepwise"]["arm_deg"]) - np.array(r["one_shot"]["arm_deg"])
    R, L = hands(err)
    rel = np.linalg.norm((R - L) - (R0 - L0))
    rows.append((ts, np.linalg.norm(R - R0), np.linalg.norm(L - L0), rel, r["one_shot"]["iterations"]))
    print(f"{ts} | right {rows[-1][1]:.1f} mm, left {rows[-1][2]:.1f} mm, right-left {rel:.1f} mm "
          f"({r['one_shot']['iterations']} it)")
conv = [x for x in rows if x[4] < 200]
for name, k in (("right", 1), ("left", 2), ("right-left", 3)):
    v = [x[k] for x in conv]
    print(f"converged runs {name}: {min(v):.1f}-{max(v):.1f} mm, median {np.median(v):.1f} mm")
