"""Camera view of the left/right flange gap at symmetric poses from the Step 2 ready pose down to the check pose
(2026-11-26, user): a gap that grows as the arms go lower / further points to sag or arm geometry, one that comes
with the wrist turn points elsewhere. Each pose is commanded offset-corrected like the post-Step 2 check (pose minus
the result's arm / head offsets) and photographed once; the gap is left minus mirrored right in link_torso_5 (mm).

Poses, in order: ready (Step 2 ready pose = last dataset sample, right arm mirrored), the picks of
scratch/check_path_pose_search.py, the check view (check_calib + J6 turn, head as in ready_poses.yaml) and the
facing view (J4/J5 also turned, head 35). Every move is checked first on the model: arm links stay >= 50 mm apart
along the joint path (15 steps), or the run stops before moving.

Config is copied to <out>/config and used from there; nothing in result/ or config/ is written.
usage: python scratch/check_path_views.py <robot address> <result json> <dataset npz> <path poses json> <out dir> [stop]
       127.0.0.1:50051 is the simulator (model m, simulated camera); anything else is a real robot (model a).
       stop: a pose name (e.g. "path t3.00"); the run ends there and the robot stays at that pose, so the operator
       can look at it (2026-11-26).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ADDR, RESULT, DATASET, PATHS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
SIM = ADDR.startswith("127.0.0.1")
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "config").exists():
    shutil.rmtree(OUT / "config")
shutil.copytree(ROOT / "config", OUT / "config")
(OUT / "config" / "home_reset_baseline.json").unlink(missing_ok=True)

from core import storage  # noqa: E402
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.data import capture_one_sample, get_both_arm_config  # noqa: E402
from core.calibration.sequences import verify  # noqa: E402

MOVE_S, LINK_SEP_MIN_MM, PATH_STEPS = 8.0, 50.0, 15
result = json.loads(RESULT.read_text(encoding="utf-8"))
paths = json.loads(PATHS.read_text(encoding="utf-8"))
data = np.load(DATASET, allow_pickle=True)
off = {"right_arm_joint_offset_deg": result["right_arm_joint_offset_deg"],
       "left_arm_joint_offset_deg": result["left_arm_joint_offset_deg"],
       "head_joint_offset_deg": result["head_joint_offset_deg"]}
off_r, off_l = np.radians(off["right_arm_joint_offset_deg"]), np.radians(off["left_arm_joint_offset_deg"])
off_h = np.radians(off["head_joint_offset_deg"])
SIGN = verify.MIRROR_JOINT_SIGN

logs = []
core = CalibrationCore()
core.on_event = lambda kind, value: logs.append(str(value)) if kind == "log" else None
core.connect_robot(ADDR, "m" if SIM else "a", include_head=True, simulated=SIM)
core.bind_robot(core.robot, "1.2")
core.connect_camera(sim=SIM, robot=core.robot, robot_version="1.2")
core.include_head_motion = True
for side in ("right", "left"):
    for calibrator in core.calibrators:
        calibrator.camera_config[f"Tf_to_marker_{side}"] = list(result["ee_to_marker_used"][side])
cfg = get_both_arm_config(core.model, "1.2")
head_idx = list(core.model.head_idx[:2])
camera = ("link_head_2", list(result["mount_to_cam_new"]))
view_cfg = core.marker_calibrator.ready_poses["v1.2"]["check_calib_view"]
check_r = np.radians(paths["check_right_deg"])
check_r[6] += np.radians(view_cfg["j6_turn_deg"])


def pose(name, right, head_deg):
    right = np.asarray(right, dtype=float)
    q = np.concatenate([right, SIGN * right])
    q[:7] -= off_r
    q[7:] -= off_l
    return name, q, np.radians(head_deg) - off_h


ready_r = np.asarray(data["q_arm"][-1][:7], dtype=float)
plan = [pose("ready", ready_r, np.degrees(np.asarray(data["q_head"][-1], dtype=float)))]
plan += [pose(f"path t{p['share']:.2f}", np.radians(p["right_arm_deg"]), p["head_deg"]) for p in paths["picks"]]
plan.append(pose("check view", check_r, view_cfg["head_deg"]))
facing = check_r.copy()
facing[4:6] += np.radians([-20.0, -30.0])
plan.append(pose("facing view", facing, [0.0, 35.0]))
if len(sys.argv) > 6:
    stop_at = [name for name, _, _ in plan].index(sys.argv[6])
    plan = plan[:stop_at + 1]


MARKER_CLEAR_MIN_MM, PLATE_HALF_MM = 30.0, 45.0
BRACKET_T = {s: verify.BaseCalibrator.make_transform(list(result["ee_to_marker_used"][s])) for s in ("right", "left")}


def marker_clearance_mm(q_arm):
    """Smallest distance between the two marker plates (+-PLATE_HALF_MM grid) and brackets (flange to plate centre).
    2026-11-26: a gathered pose put the plates into contact; the link check alone did not see the brackets."""
    dyn = core.robot.get_dynamics()
    q_full = np.zeros(len(core.model.robot_joint_names))
    q_full[cfg["arm_idx"]] = q_arm
    g = np.linspace(-PLATE_HALF_MM, PLATE_HALF_MM, 7) / 1000.0
    pts = {}
    for side in ("right", "left"):
        T_ee = verify._fk(dyn, core.model, q_full, cfg["ee_links"][side])
        T_m = T_ee @ BRACKET_T[side]
        plate = [(T_m @ np.array([a, b, 0.0, 1.0]))[:3] for a in g for b in g]
        bracket = [(T_ee[:, 3] + s * (T_m[:, 3] - T_ee[:, 3]))[:3] for s in np.linspace(0.0, 1.0, 6)]
        pts[side] = np.array(plate + bracket)
    return float(np.min(np.linalg.norm(pts["right"][:, None, :] - pts["left"][None, :, :], axis=2)) * 1000.0)


def path_clear(q_from, q_to, h_from, h_to):
    """(smallest link separation, smallest marker clearance) along the joint path, mm."""
    steps = np.linspace(0.0, 1.0, PATH_STEPS)
    links = min(np.min(verify._link_separations(core, cfg, head_idx, q_from + s * (q_to - q_from), h_from + s * (h_to - h_from)))
                for s in steps)
    markers = min(marker_clearance_mm(q_from + s * (q_to - q_from)) for s in steps)
    return links, markers


rows = []
state = np.asarray(core.robot.get_state().position, dtype=float)
q_now, h_now = state[list(cfg["arm_idx"])], state[head_idx]
for name, q_cmd, h_cmd in plan:
    worst, marker_gap = path_clear(q_now, q_cmd, h_now, h_cmd)
    print(f"{name}: path link separation >= {worst:.0f} mm, marker plates >= {marker_gap:.0f} mm", flush=True)
    # 2026-11-26 real run: after a pose without markers the start pose came back without values, the check gave NaN and
    # 'NaN < limit' let the move through. Anything but a finite separation above the limit stops the run.
    if not np.isfinite(worst) or worst < LINK_SEP_MIN_MM:
        print(f"[STOP] {name}: the arms would come within {worst:.0f} mm on the way; not moving.", flush=True)
        break
    # Starting closer than the limit (e.g. after the contact) is allowed only to move apart: never closer than at
    # the start, and at least the limit at the end.
    gap_start, gap_end = marker_clearance_mm(q_now), marker_clearance_mm(q_cmd)
    floor = min(MARKER_CLEAR_MIN_MM, gap_start - 1.0)
    if not np.isfinite(marker_gap) or marker_gap < floor or gap_end < MARKER_CLEAR_MIN_MM:
        print(f"[STOP] {name}: marker plates {gap_start:.0f} mm at the start, {marker_gap:.0f} mm on the way, "
              f"{gap_end:.0f} mm at the end (need >= {MARKER_CLEAR_MIN_MM:.0f} at the end, never closer than the start); "
              "not moving.", flush=True)
        break
    ok = core.marker_calibrator.movej(core.robot, torso=None, right_arm=list(q_cmd[:7]), left_arm=list(q_cmd[7:]),
                                      head=list(h_cmd), minimum_time=MOVE_S, apply_offsets=False)
    if not ok:
        print(f"[STOP] {name}: move failed.", flush=True)
        break
    q_arm, q_head, markers = capture_one_sample(core.robot, cfg["arm_idx"], core.observer, sampling_time=0 if SIM else 1,
                                                side="all", head_idx=head_idx)
    state = np.asarray(core.robot.get_state().position, dtype=float)
    q_now, h_now = state[list(cfg["arm_idx"])], state[head_idx]
    if markers is None:
        print(f"   {name}: both markers not detected", flush=True)
        rows.append({"pose": name, "seen": False})
        continue
    ev = verify._evaluate(core, cfg, head_idx, q_arm, q_head, markers, camera, True, off)
    m = ev["measured_symmetry"]
    row ={"pose": name, "seen": True, "gap_x": round(m["dx_mm"], 2), "gap_y": round(m["dy_mm"], 2),
           "gap_z": round(m["dz_mm"], 2), "gap_xz": round(float(np.hypot(m["dx_mm"], m["dz_mm"])), 2),
           "model_vs_camera": {s: [round(v["pos_err_mm"], 2), round(v["rot_err_deg"], 2)] for s, v in ev["per_arm"].items()},
           "q_cmd_arm_deg": np.degrees(q_cmd).round(3).tolist(), "q_cmd_head_deg": np.degrees(h_cmd).round(3).tolist(),
           "markers": {s: np.asarray(markers[i]).tolist() for i, s in enumerate(("right", "left"))},
           "q_arm": np.asarray(q_arm).tolist(), "q_head": np.asarray(q_head).tolist()}
    rows.append(row)
    print(f"   {name}: gap x {row['gap_x']:+.2f} y {row['gap_y']:+.2f} z {row['gap_z']:+.2f} mm -> x/z {row['gap_xz']:.2f} mm; "
          f"model vs camera R {row['model_vs_camera']['right']}, L {row['model_vs_camera']['left']}", flush=True)
(OUT / "check_path_views.json").write_text(json.dumps({"address": ADDR, "result": str(RESULT), "rows": rows}, indent=1),
                                           encoding="utf-8")
core.close()
print("written", OUT / "check_path_views.json")
