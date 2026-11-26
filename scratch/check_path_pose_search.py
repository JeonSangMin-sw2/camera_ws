"""Symmetric poses between the Step 2 ready pose and the check pose that the head camera sees both markers from
(2026-11-26, user: does the check-pose gap grow with how far the arms are lowered / reached?). Shoulders and elbow
(J0-J3) go a share t of the way from the ready pose to check_calib; at each t the wrist (J4-J6) and the head tilt are
searched for a view. Arms mirrored. Simulator FK only (127.0.0.1, model m, v1.2), nominal camera and brackets,
zero offsets; nothing moves. Same criteria as scratch/low_view_pose_search.py.
usage: python scratch/check_path_pose_search.py <ready right arm deg, 7 comma-separated> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rby1_sdk as rby  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

MIRROR_SIGN = np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
RANGE_MM, MARGIN_PX, W, H, FACE_MAX = (180.0, 360.0), 60.0, 1280, 720, 40.0
J_LIMIT_MARGIN_DEG, LINK_SEP_MIN_MM = 5.0, 50.0
TILTS = np.arange(0.0, 61.0, 5.0)
SHARES = (1 / 3, 2 / 3, 0.85)

ready_r = np.array([float(v) for v in sys.argv[1].split(",")])
OUT = Path(sys.argv[2])
poses = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v1.2"]
check_r = np.array(poses["check_calib"]["right_arm"], dtype=float)
setting = yaml.safe_load((ROOT / "config" / "setting.yaml").read_text(encoding="utf-8"))
intr = yaml.safe_load((ROOT / "config" / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
K = np.asarray(intr["camera_matrix"], dtype=float)
T_mount_cam = BaseCalibrator.make_transform(setting["camera"]["mount_to_cam_nominal"])
bracket = {s: BaseCalibrator.make_transform(setting["marker"][f"Tf_to_marker_{s}_v12"]) for s in ("right", "left")}

robot = rby.create_robot("127.0.0.1:50051", "m")
assert robot.connect(), "simulator not reachable"
dyn, model = robot.get_dynamics(), robot.model()
names = list(model.robot_joint_names)
head_idx = [names.index("head_0"), names.index("head_1")]
r_idx, l_idx = list(model.right_arm_idx), list(model.left_arm_idx)
links = (["link_torso_5", "ee_right", "ee_left", "link_head_2"]
         + [f"link_right_arm_{k}" for k in range(7)] + [f"link_left_arm_{k}" for k in range(7)])
state = dyn.make_state(links, names)
lo = np.degrees(np.clip(dyn.get_limit_q_lower(state), -1e3, 1e3))
hi = np.degrees(np.clip(dyn.get_limit_q_upper(state), -1e3, 1e3))
q_base = np.zeros(model.robot_dof)
q_base[list(model.torso_idx)] = np.radians([0, 30, -60, 30, 0, 0])


def fk(right_deg, tilt=0.0):
    q = q_base.copy()
    q[head_idx] = np.radians([0.0, tilt])
    q[r_idx] = np.radians(right_deg)
    q[l_idx] = np.radians(MIRROR_SIGN * right_deg)
    state.set_q(q)
    dyn.compute_forward_kinematics(state)
    return {name: dyn.compute_transformation(state, 0, i) for i, name in enumerate(links) if i}


def within_limits(right_deg):
    left = MIRROR_SIGN * right_deg
    return all(lo[i] + J_LIMIT_MARGIN_DEG <= a <= hi[i] - J_LIMIT_MARGIN_DEG
               for idx, deg in ((r_idx, right_deg), (l_idx, left)) for i, a in zip(idx, deg))


def link_separation_mm(T):
    pr = [T[f"link_right_arm_{k}"][:3, 3] for k in range(7)] + [T["ee_right"][:3, 3]]
    pl = [T[f"link_left_arm_{k}"][:3, 3] for k in range(7)] + [T["ee_left"][:3, 3]]
    return min(np.linalg.norm(a - b) for a in pr for b in pl) * 1000


def view(T_arm, T_cam):
    out = {}
    for side in ("right", "left"):
        T_m = np.linalg.inv(T_cam) @ T_arm[f"ee_{side}"] @ bracket[side]
        p = T_m[:3, 3]
        if p[2] <= 0:
            return None
        u, v = (K @ (p / p[2]))[:2]
        face = np.degrees(np.arccos(np.clip(-T_m[:3, 2] @ (-p / np.linalg.norm(p)), -1, 1)))
        out[side] = dict(u=float(u), v=float(v), face=float(face), range_mm=float(np.linalg.norm(p) * 1000))
    return out


T_cams = {t: fk(check_r, t)["link_head_2"] @ T_mount_cam for t in TILTS}
for name, q in (("ready", ready_r), ("check", check_r)):
    T = fk(q)
    print(f"{name}: right ee (link_torso_5, mm) {np.round(T['ee_right'][:3, 3] * 1000, 1)}, link sep {link_separation_mm(T):.0f} mm")
picks = []
for share in SHARES:
    shoulders = ready_r[:4] + share * (check_r[:4] - ready_r[:4])
    wrist0 = ready_r[4:] + share * (check_r[4:] - ready_r[4:])
    found = []
    for d4 in np.arange(-90, 91, 15.0):
        for d5 in np.arange(-60, 61, 15.0):
            for d6 in np.arange(-180, 181, 30.0):
                q_r = np.concatenate([shoulders, wrist0 + [d4, d5, d6]])
                if not within_limits(q_r):
                    continue
                T = fk(q_r)
                sep = link_separation_mm(T)
                if sep < LINK_SEP_MIN_MM:
                    continue
                for tilt in TILTS:
                    v = view(T, T_cams[tilt])
                    if v is None or not all(
                            v[s]["face"] <= FACE_MAX and RANGE_MM[0] <= v[s]["range_mm"] <= RANGE_MM[1]
                            and MARGIN_PX <= v[s]["u"] <= W - MARGIN_PX and MARGIN_PX <= v[s]["v"] <= H - MARGIN_PX
                            for s in v):
                        continue
                    found.append(dict(q=q_r, tilt=float(tilt), v=v, sep=sep, face=max(v[s]["face"] for s in v),
                                      ee=T["ee_right"][:3, 3] * 1000))
    print(f"t {share:.2f}: shoulders {np.round(shoulders, 2)}, {len(found)} viewable poses")
    if found:
        best = min(found, key=lambda f: (f["face"], -f["sep"]))
        print(f"   pick: right arm {np.round(best['q'], 2).tolist()}, head tilt {best['tilt']:.0f}, face {best['face']:.0f} deg, "
              f"range {best['v']['right']['range_mm']:.0f}/{best['v']['left']['range_mm']:.0f} mm, "
              f"link sep {best['sep']:.0f} mm, right ee {np.round(best['ee'], 1)}")
        picks.append({"share": share, "right_arm_deg": np.round(best["q"], 3).tolist(), "head_deg": [0.0, best["tilt"]],
                      "face_deg": round(best["face"], 1), "link_sep_mm": round(best["sep"], 1),
                      "right_ee_mm": np.round(best["ee"], 1).tolist()})
OUT.write_text(json.dumps({"ready_right_deg": ready_r.tolist(), "check_right_deg": check_r.tolist(), "picks": picks},
                          indent=1), encoding="utf-8")
print("written", OUT)
