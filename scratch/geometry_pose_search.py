"""More symmetric poses the head camera sees both markers from, spread over the joint space, for fitting the arm
geometry (2026-11-26, user). Only inside the hand-position box the robot has already been to today (ready pose,
the three path poses, the check pose: link_torso_5 x 300-420, right y -200..-130, z 0-270 mm), so no new region of
the workspace is entered. Random shoulder/elbow (J0-J3) samples, the wrist (J4-J6) and head tilt searched for a view
(same criteria as scratch/check_path_pose_search.py), then the N most different joint sets kept (farthest-point).
Writes the picks in the format scratch/check_path_views.py reads. Simulator FK only (127.0.0.1, model m, v1.2).
usage: python scratch/geometry_pose_search.py <ready right arm deg, 7 comma-separated> <out json> [N]
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
J_LIMIT_MARGIN_DEG, LINK_SEP_MIN_MM = 5.0, 150.0
BOX = dict(x=(300.0, 420.0), y=(-200.0, -130.0), z=(0.0, 270.0))   # right hand, link_torso_5, mm
TILTS = np.arange(0.0, 61.0, 5.0)
N_SAMPLES, SEED = 400, 7

ready_r = np.array([float(v) for v in sys.argv[1].split(",")])
OUT = Path(sys.argv[2])
N_PICK = int(sys.argv[3]) if len(sys.argv) > 3 else 10
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
rng = np.random.default_rng(SEED)
lo4, hi4 = np.minimum(ready_r[:4], check_r[:4]) - 10.0, np.maximum(ready_r[:4], check_r[:4]) + 10.0
found = []
for _ in range(N_SAMPLES):
    shoulders = rng.uniform(lo4, hi4)
    T0 = fk(np.concatenate([shoulders, check_r[4:]]))
    ee = T0["ee_right"][:3, 3] * 1000          # the wrist does not move the hand by much; filter on it first
    if not (BOX["x"][0] - 40 <= ee[0] <= BOX["x"][1] + 40 and BOX["y"][0] - 40 <= ee[1] <= BOX["y"][1] + 40
            and BOX["z"][0] - 40 <= ee[2] <= BOX["z"][1] + 40):
        continue
    best = None
    for d4 in np.arange(-150, 151, 30.0):
        for d5 in np.arange(-60, 61, 20.0):
            for d6 in np.arange(-180, 181, 30.0):
                q_r = np.concatenate([shoulders, [d4, d5, d6]])
                if not within_limits(q_r):
                    continue
                T = fk(q_r)
                e = T["ee_right"][:3, 3] * 1000
                if not (BOX["x"][0] <= e[0] <= BOX["x"][1] and BOX["y"][0] <= e[1] <= BOX["y"][1]
                        and BOX["z"][0] <= e[2] <= BOX["z"][1]):
                    continue
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
                    face = max(v[s]["face"] for s in v)
                    if best is None or face < best["face"]:
                        best = dict(q=q_r, tilt=float(tilt), face=face, sep=sep, ee=e,
                                    range=[v["right"]["range_mm"], v["left"]["range_mm"]])
    if best is not None:
        found.append(best)
print(f"viewable poses: {len(found)} of {N_SAMPLES} shoulder samples")
# Farthest-point choice on the shoulder/elbow angles (what the geometry fit needs to vary), starting from the one
# farthest from both ready and check.
picks = []
pool = list(found)
anchors = [ready_r[:4], check_r[:4]]
while pool and len(picks) < N_PICK:
    ref = anchors + [p["q"][:4] for p in picks]
    k = int(np.argmax([min(np.linalg.norm(f["q"][:4] - r) for r in ref) for f in pool]))
    picks.append(pool.pop(k))
out = []
for i, p in enumerate(picks):
    print(f"pick {i + 1}: right arm {np.round(p['q'], 1).tolist()}, head tilt {p['tilt']:.0f}, face {p['face']:.0f} deg, "
          f"range {p['range'][0]:.0f}/{p['range'][1]:.0f} mm, link sep {p['sep']:.0f} mm, right ee {np.round(p['ee'], 0)}")
    out.append({"share": float(i + 1), "right_arm_deg": np.round(p["q"], 3).tolist(), "head_deg": [0.0, p["tilt"]],
                "face_deg": round(p["face"], 1), "link_sep_mm": round(p["sep"], 1), "right_ee_mm": np.round(p["ee"], 1).tolist()})
OUT.write_text(json.dumps({"ready_right_deg": ready_r.tolist(), "check_right_deg": check_r.tolist(), "picks": out},
                          indent=1), encoding="utf-8")
print("written", OUT)
