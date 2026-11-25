"""Search poses at or above a version's check pose that the head camera sees both markers from, for the Step 2
check-region list (ready_poses.yaml step2_check_region). Arms mirrored. Simulator FK only (127.0.0.1, model m;
the running simulator must be the requested version), nominal camera and brackets, zero offsets; nothing moves.

Criteria (user, 2026-11-25): each marker <= face_max deg from facing the camera, 180-330 mm away (Step 2 base
poses saw 182-268 mm on the 2026-11-25 13:23 real run, the check pose about 300), >= 60 px inside the image,
joint targets >= 5 deg inside their limits, left and right links >= 50 mm apart, the right hand at most 150 mm
above the check pose and within 100 mm of it sideways and forward.
--shoulders-fixed --face-max 30 is the 2026-11-24 search that found the post-Step 2 facing view (J0-J3 kept).

usage: python scratch/low_view_pose_search.py [1.2|1.3] [--face-max DEG] [--shoulders-fixed]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rby1_sdk as rby  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

MIRROR_SIGN = np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
RANGE_MM, MARGIN_PX, W, H = (180.0, 330.0), 60.0, 1280, 720
J_LIMIT_MARGIN_DEG, LINK_SEP_MIN_MM = 5.0, 50.0
RISE_MAX_MM, SHIFT_MAX_MM = 150.0, 100.0
TILTS = np.arange(15.0, 61.0, 5.0)

ap = argparse.ArgumentParser()
ap.add_argument("version", nargs="?", default="1.2", choices=["1.2", "1.3"])
ap.add_argument("--face-max", type=float, default=40.0)
ap.add_argument("--shoulders-fixed", action="store_true")
args = ap.parse_args()
vtag = "v" + args.version.replace(".", "")

poses = yaml.safe_load((ROOT / "config" / "ready_poses.yaml").read_text(encoding="utf-8"))["v" + args.version]
check_r = np.array(poses["check_calib"]["right_arm"], dtype=float)
setting = yaml.safe_load((ROOT / "config" / "setting.yaml").read_text(encoding="utf-8"))
intr = yaml.safe_load((ROOT / "config" / "camera_intrinsics.yaml").read_text(encoding="utf-8"))
K = np.asarray(intr["camera_matrix"], dtype=float)
T_mount_cam = BaseCalibrator.make_transform(setting["camera"]["mount_to_cam_nominal"])
bracket = {s: BaseCalibrator.make_transform(setting["marker"][f"Tf_to_marker_{s}_{vtag}"]) for s in ("right", "left")}

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
        out[side] = dict(u=u, v=v, face=face, range_mm=np.linalg.norm(p) * 1000)
    return out


# The head chain does not depend on the arms: one camera pose per tilt.
T_cams = {t: fk(check_r, t)["link_head_2"] @ T_mount_cam for t in TILTS}
check_ee = fk(check_r)["ee_right"][:3, 3] * 1000
check_sep = link_separation_mm(fk(check_r))
print(f"v{args.version} check pose right ee (link_torso_5, mm): {np.round(check_ee, 1)}, link sep {check_sep:.0f} mm")
if args.shoulders_fixed:
    shoulder_grid = [np.zeros(4)]
else:
    shoulder_grid = [np.array([d0, d1, 0.0, d3]) for d0 in np.arange(-30, 11, 10.0)
                     for d1 in (-10.0, 0.0, 10.0) for d3 in np.arange(-20, 21, 10.0)]
found = []
for ds in shoulder_grid:
    for d4 in np.arange(-60, 61, 15.0):
        for d5 in np.arange(-60, 61, 15.0):
            for d6 in np.arange(-180, 181, 30.0):
                dq = np.concatenate([ds, [d4, d5, d6]])
                q_r = check_r + dq
                if not within_limits(q_r):
                    continue
                T = fk(q_r)
                ee = T["ee_right"][:3, 3] * 1000
                rise = ee[2] - check_ee[2]
                if not (-1.0 <= rise <= RISE_MAX_MM and np.all(np.abs(ee[:2] - check_ee[:2]) <= SHIFT_MAX_MM)):
                    continue
                for tilt in TILTS:
                    v = view(T, T_cams[tilt])
                    if v is None or not all(
                            v[s]["face"] <= args.face_max and RANGE_MM[0] <= v[s]["range_mm"] <= RANGE_MM[1]
                            and MARGIN_PX <= v[s]["u"] <= W - MARGIN_PX and MARGIN_PX <= v[s]["v"] <= H - MARGIN_PX
                            for s in v):
                        continue
                    sep = link_separation_mm(T)
                    if sep < LINK_SEP_MIN_MM:
                        continue
                    found.append(dict(dq=dq, q=q_r, tilt=tilt, v=v, rise=rise, sep=sep,
                                      face=max(v[s]["face"] for s in v), travel=np.abs(dq).sum()))
print(f"poses meeting all criteria: {len(found)} (face <= {args.face_max:g} deg)")


def line(f):
    v = f["v"]
    return (f"dq {np.round(f['dq']).astype(int).tolist()} tilt {f['tilt']:.0f} | face R {v['right']['face']:.0f} "
            f"L {v['left']['face']:.0f} deg | range {v['right']['range_mm']:.0f}/{v['left']['range_mm']:.0f} mm | "
            f"uv R ({v['right']['u']:.0f},{v['right']['v']:.0f}) L ({v['left']['u']:.0f},{v['left']['v']:.0f}) | "
            f"rise {f['rise']:.0f} mm | link sep {f['sep']:.0f} mm\n    right_arm {np.round(f['q'], 2).tolist()}")


# Pick: near the check pose (rise <= 50 mm) and higher (70-150 mm); in each, one oblique (face 28-40) and one
# facing (<= 22) pose, both markers >= 280 mm away (the check-pose distance, beyond the Step 2 base poses' 268),
# links no closer together than at the check pose, least joint change from the check pose, with a different
# head tilt from its band partner.
PICK_RANGE_MIN_MM = 280.0
picked = []
for band in ((-1.0, 50.0), (70.0, RISE_MAX_MM)):
    used_tilts = []
    for face_band in ((28.0, args.face_max), (0.0, 22.0)):
        pool = [f for f in found if band[0] <= f["rise"] <= band[1] and face_band[0] <= f["face"] <= face_band[1]
                and f["tilt"] not in used_tilts
                and min(f["v"][x]["range_mm"] for x in f["v"]) >= PICK_RANGE_MIN_MM and f["sep"] >= check_sep]
        if pool:
            best = min(pool, key=lambda f: (f["travel"], -min(f["v"][s]["u"] for s in f["v"])))
            picked.append(best)
            used_tilts.append(best["tilt"])
for band in ((-1.0, 50.0), (70.0, RISE_MAX_MM)):
    n = sum(band[0] <= f["rise"] <= band[1] and min(f["v"][x]["range_mm"] for x in f["v"]) >= PICK_RANGE_MIN_MM
            for f in found)
    print(f"rise {band[0]:.0f}-{band[1]:.0f} mm, both markers >= {PICK_RANGE_MIN_MM:.0f} mm: {n}")
print("picked:")
for f in picked:
    print("  " + line(f))
robot.disconnect()
