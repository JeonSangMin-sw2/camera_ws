"""Simulator check of the Step 2 wrist-diversity poses (G4, v1.2 simulator, model "m").

Moves the SIMULATED robot (127.0.0.1:50051) through the production Step 2 collection with the
ready_poses.yaml step2_wrist_diversity list forced on (in-process patch; setting.yaml untouched), then
solves Step 2 offline with the production optimize_step2 on
  - the base samples only, and
  - base + wrist-diversity samples,
with the Step 1 anchors (J3/J5/J6) at the simulator truth and with one J6 anchor off by +1 deg.
Reports per pose: command success / skip, marker tilt to the camera and image margin (D405
pinhole; the simulated sensor itself always "sees" the marker), and per solve the joint errors and
the hand error at the check pose after applying the result.

Simulator truth: config/simulation.yaml with the bracket errors set to zero in memory only, and the
Step 1.5 camera/head given as the truth, so the comparison isolates Step 2. Outputs go to a temp
folder; result/ and config/ are not written.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\validate_wrist_diversity_sim.py
"""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402
from core.calibration.sequences import collection, step2 as step2_mod  # noqa: E402
from core.marker_detection import SimulationModel, load_truth_config  # noqa: E402
from core.robot.motion import AutoCollectionConfig  # noqa: E402

ADDR, MODEL, VERSION = "127.0.0.1:50051", "m", "1.2"
OUT = Path(tempfile.mkdtemp(prefix="wrist_sim_"))
LOG = open(OUT / "log.txt", "w", encoding="utf-8")
K = np.array(yaml.safe_load(open("config/camera_intrinsics_d405.yaml"))["camera_matrix"])
W, H, HALF = 1280, 720, 0.032


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + "\n")
    LOG.flush()


truth = load_truth_config()
truth_nb = copy.deepcopy(truth)
for side in ("right", "left"):
    truth_nb["offsets"][side]["bracket_pos"] = [0.0, 0.0, 0.0]
    truth_nb["offsets"][side]["bracket_rpy"] = [0.0, 0.0, 0.0]
sim_model = SimulationModel.create(VERSION, config=truth_nb)
T_ARM = {s: np.degrees(sim_model.arm_offsets(s)) for s in ("right", "left")}   # delta = physical - encoder
T_HEAD = np.array([truth["offsets"]["head"]["pan"], truth["offsets"]["head"]["tilt"]])
wrist = yaml.safe_load(open("config/ready_poses.yaml", encoding="utf-8"))[f"v{VERSION}"]["step2_wrist_diversity"]
log(f"output: {OUT}\ntruth arm offsets R {np.round(T_ARM['right'], 2)} L {np.round(T_ARM['left'], 2)} head {T_HEAD}")
log(f"wrist-diversity poses: {len(wrist)}")

core = CalibrationCore()
core.on_event = lambda kind, value: log(value) if kind == "log" else None
core.connect_robot(ADDR, MODEL, simulated=True)
core.bind_robot(core.robot, VERSION)
core.connect_camera(sim=True, robot=core.robot, robot_version=VERSION)
core.observer.engine.simulation_model = sim_model
core.include_head_motion = True
core.auto_config = AutoCollectionConfig()
core.auto_config.max_loops = 1
core.prompt_teaching = lambda *a, **k: True


def no_spacing_prompt(gap, min_gap):
    raise RuntimeError(f"markers too close in the simulator ({gap:.3f} < {min_gap:.3f} m)")


core.prompt_marker_spacing = no_spacing_prompt
pose_total = {}
orig_emit = core.emit


def emit(kind, value):
    if kind == "progress" and isinstance(value, dict) and value.get("stage") == "collect":
        pose_total["n"] = value["pose_total"]
    return orig_emit(kind, value)


core.emit = emit

# ---- 1. Collection (simulated robot moves) ----
t0 = time.time()
with patch.object(collection, "load_step2_wrist_diversity", return_value=(wrist, True)):
    result = core.run("collect", prepare=True)
log(f"collection: {result.status} {result.error or ''} in {time.time() - t0:.0f} s")
samples = result.completed.get("samples") or result.partial.get("samples") or []
n_plan = pose_total.get("n", 0)
first_extra = n_plan - len(wrist) - 1          # extras, then one restore_baseline step
is_extra = [first_extra <= s["motion_index"] < n_plan - 1 for s in samples]
extra = [s for s, e in zip(samples, is_extra) if e]
base = [s for s, e in zip(samples, is_extra) if not e]
log(f"plan {n_plan} poses, samples {len(samples)}: base {len(base)}, wrist-diversity {len(extra)} of {len(wrist)}")


def view(T):
    p = T[:3, 3]
    tilt = np.degrees(np.arccos(abs(np.dot(T[:3, 2], p / np.linalg.norm(p)))))
    c = [T[:3, :3] @ np.array([sx * HALF, sy * HALF, 0]) + p for sx in (-1, 1) for sy in (-1, 1)]
    uv = np.array([[K[0, 0] * x[0] / x[2] + K[0, 2], K[1, 1] * x[1] / x[2] + K[1, 2]] for x in c])
    margin = min(uv[:, 0].min(), W - uv[:, 0].max(), uv[:, 1].min(), H - uv[:, 1].max())
    return tilt, margin, np.linalg.norm(p) * 1000


for tag, group in (("base", base), ("wrist", extra)):
    v = np.array([view(np.asarray(s["marker"][k])) for s in group for k in range(2)]).reshape(-1, 2, 3) if group else None
    if v is not None:
        log(f"{tag:5s}: tilt R max {v[:, 0, 0].max():.1f} / L max {v[:, 1, 0].max():.1f} deg, "
            f"poses > 30 deg: R {(v[:, 0, 0] > 30).sum()} L {(v[:, 1, 0] > 30).sum()} | image margin min "
            f"{v[:, :, 1].min():.0f} px | range {v[:, :, 2].min():.0f}..{v[:, :, 2].max():.0f} mm")
for s in extra:
    (tr, mr, dr), (tl, ml, dl) = view(np.asarray(s["marker"][0])), view(np.asarray(s["marker"][1]))
    log(f"  pose {s['motion_index'] - first_extra + 1:2d}: tilt R {tr:4.1f} L {tl:4.1f} deg | margin R {mr:4.0f} L {ml:4.0f} px | "
        f"range R {dr:.0f} L {dl:.0f} mm")

# ---- 2. Offline Step 2 solves (no motion) ----
chk = yaml.safe_load(open("config/ready_poses.yaml"))[f"v{VERSION}"]["check_calib"]
dyn = core.robot.get_dynamics()
model = core.robot.model()
NQ = len(model.robot_joint_names)


def hands(offsets_deg):
    q = np.zeros(NQ)
    for side, idx in (("right", model.right_arm_idx[:7]), ("left", model.left_arm_idx[:7])):
        q[list(idx)] = np.radians(np.array(chk[f"{side}_arm"]) + offsets_deg[side])
    return {side: BaseCalibrator.compute_fk(core.robot, dyn, q, f"ee_{side}", "link_torso_5")[:3, 3] * 1000
            for side in ("right", "left")}


def solve(group, tag, j6_bias=None):
    store = {s: {"joint3": -T_ARM[s][3], "joint5": -T_ARM[s][5], "joint6": -T_ARM[s][6]} for s in ("right", "left")}
    if j6_bias:
        store[j6_bias]["joint6"] -= 1.0          # store = -result: result J6 anchor +1 deg
    core.joint_offsets_store = {**store, "head": {"pan": float(T_HEAD[0]), "tilt": float(T_HEAD[1])}}
    core.apply_joint_offset_flag = True
    core.head_camera_calibrator.calibrated_results = {"success": True, "calibrated_mount_to_cam": truth["mount_to_cam"],
                                                      "head_offsets_deg": {"pan": float(T_HEAD[0]), "tilt": float(T_HEAD[1])}}
    for cal in core.calibrators:
        for side in ("right", "left"):
            cal.camera_config[f"Tf_to_marker_{side}"] = truth["brackets"][VERSION][side]
        cal.camera_config["mount_to_cam"] = truth["mount_to_cam"]
        cal.camera_config["mount_to_cam_nominal"] = truth["mount_to_cam"]
    q_arm = np.array([s["q_arm"] for s in group])
    q_head = np.array([s["q_head"] for s in group])
    T = np.array([s["marker"] for s in group])
    path = OUT / f"result_{tag}.json"
    step2_mod.optimize_step2(core, ["right", "left"], True, True, q_arm, q_head, T, str(path),
                             lambda_cam_pos=1.0, lambda_cam_rot=100.0)
    r = json.load(open(path))
    est = {"right": np.array(r["right_arm_joint_offset_deg"]), "left": np.array(r["left_arm_joint_offset_deg"])}
    err = {s: est[s] - T_ARM[s] for s in ("right", "left")}
    true_h, off_h = hands(T_ARM), hands({s: T_ARM[s] - err[s] for s in ("right", "left")})
    hand = {s: off_h[s] - true_h[s] for s in ("right", "left")}   # physical hand error after applying est
    log(f"{tag:22s}: err R {' '.join(f'{v:+.2f}' for v in err['right'])} | L {' '.join(f'{v:+.2f}' for v in err['left'])} | "
        f"check-pose hand R x{hand['right'][0]:+.2f} y{hand['right'][1]:+.2f} z{hand['right'][2]:+.2f} / "
        f"L x{hand['left'][0]:+.2f} y{hand['left'][1]:+.2f} z{hand['left'][2]:+.2f} mm")
    return err, hand


core.on_event = lambda kind, value: LOG.write(str(value) + "\n") if kind == "log" else None
log("\nStep 2 solves (errors = estimate - truth, deg; hand error at check_calib after applying the estimate)")
for bias in (None, "left", "right"):
    for tag, group in (("base", base), ("base+wrist", base + extra)):
        if group:
            solve(group, f"{tag} J6 {'truth' if bias is None else bias + ' +1deg'}", bias)
log(f"\nfull log: {OUT / 'log.txt'}")
core.close()
