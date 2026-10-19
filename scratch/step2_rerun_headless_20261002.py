"""Offline re-solve of the 2026-10-02 11:49 headless Step 2 run with the production optimize_step2.
Robot connection is read-only (model/dynamics only, no commands). Output goes to the scratchpad.

 a_as_run     : Step 1 anchors as in setting.yaml (reproduction check)
 b_j6_zero    : J6 locked to 0 on both arms (user: J6 did not change since the last apply)
 c_j6_base    : J6 locked to the reset baseline value (same idea, exact numbers)
 d_cam_free   : as run, but the camera extrinsics may move (headless mode locks them to CAD nominal)
 e_cam_free_j6_zero
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = r"C:\Users\User\Desktop\camera_ws"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import rby1_sdk as rby  # noqa: E402
from core.storage import ConfigStorage  # noqa: E402
from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.data import load_npz_dataset  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

TS = "20261002_114936_921873"
OUT = Path(tempfile.mkdtemp(prefix="rerun1002_", dir=os.path.dirname(os.path.abspath(__file__))))

robot = rby.create_robot("192.168.1.40:50051", "a")
if not robot.connect():
    raise SystemExit("robot not reachable (needed read-only for the kinematic model)")

SETTING = ConfigStorage.load("config/setting.yaml")
BASE = json.load(open("config/home_reset_baseline.json"))
assert BASE["metadata"]["created_at"] == "2026-10-02T11:35:36", BASE["metadata"]
SAVED = json.load(open(f"result/result_step2/result_{TS}.json"))
q_arm, q_head, T_meas = load_npz_dataset(f"result/result_step2/dataset_{TS}.npz")
store = SETTING["joint_offset"]
print("dataset:", q_arm.shape, None if q_head is None else np.shape(q_head), T_meas.shape, file=sys.__stdout__)
print("setting store:", {s: store[s] for s in ("right", "left")}, file=sys.__stdout__)
print("camera:", SETTING["camera"]["mount_to_cam"], SETTING["camera"]["head_base_to_cam"], file=sys.__stdout__)


PREV = json.load(open("result/result_step2/result_20261001_194927_154137.json"))["ee_to_marker_used"]
NOM = {"right": SETTING["marker"]["Tf_to_marker_right_v12"], "left": SETTING["marker"]["Tf_to_marker_left_v12"]}


def solve(tag, j6_store=None, optimize_camera=True, brackets=None, pitch_zero=False):
    core = CalibrationCore()
    core.on_event = lambda kind, value: None
    core.bind_robot(robot, "1.2")
    core.include_head_motion = False
    core.apply_joint_offset_flag = True
    core.joint_offsets_store = {s: {k: float(store[s][k]) for k in ("joint3", "joint5", "joint6")} for s in ("right", "left")}
    core.joint_offsets_store["head"] = {"pan": 0.0, "tilt": 0.0}
    if j6_store:
        for s in ("right", "left"):
            core.joint_offsets_store[s]["joint6"] = j6_store[s]
    for cal in core.calibrators:
        used = {k: list(v) for k, v in (brackets or SAVED["ee_to_marker_used"]).items()}
        if pitch_zero:
            for k in used:
                used[k][4] = 0.0
        cal.camera_config["Tf_to_marker_right"] = used["right"]
        cal.camera_config["Tf_to_marker_left"] = used["left"]
    path = OUT / f"result_{tag}.json"
    step2_mod.optimize_step2(core, ["right", "left"], False, optimize_camera, q_arm, q_head, T_meas, str(path),
                             lambda_cam_pos=1.0, lambda_cam_rot=1e6)
    return json.load(open(path))


def arm(r):
    return np.array(r["right_arm_joint_offset_deg"] + r["left_arm_joint_offset_deg"])


def fmt(v):
    return " ".join(f"{x:+.2f}" for x in v)


base = arm(BASE)
out = sys.__stdout__
print("\ncolumns: R J0..J6 | L J0..J6   (values are result - reset baseline, deg; 0 = same as the zero applied on 10-01 19:5x)\n", file=out)
print(f"{'saved':20s}: {fmt(arm(SAVED) - base)} | pos {SAVED['measurement_noise']['measurement_noise_pos_std_mm']:.3f} mm"
      f" rot {SAVED['measurement_noise']['measurement_noise_rot_std_deg']:.3f} deg", file=out)
Z = {"right": 0.0, "left": 0.0}
cases = {
    "d_as_run": dict(),
    "e_j6_zero": dict(j6_store=Z),
    "i_prev_brackets": dict(brackets=PREV),
    "j_prev_brackets_j6_zero": dict(brackets=PREV, j6_store=Z),
    "k_pitch0": dict(pitch_zero=True),
    "l_pitch0_j6_zero": dict(pitch_zero=True, j6_store=Z),
    "m_nominal_brackets_j6_zero": dict(brackets=NOM, j6_store=Z),
}
results = {}
for tag, kwargs in cases.items():
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = solve(tag, **kwargs)
    except Exception as error:
        print(f"{tag:20s}: FAILED {type(error).__name__}: {error}", file=out)
        continue
    results[tag] = r
    extra = f" | reproduction max {np.abs(arm(r) - arm(SAVED)).max():.3f}" if tag == "d_as_run" else ""
    print(f"{tag:20s}: {fmt(arm(r) - base)} | pos {r['measurement_noise']['measurement_noise_pos_std_mm']:.3f} mm"
          f" rot {r['measurement_noise']['measurement_noise_rot_std_deg']:.3f} deg | cam {np.round(r.get('mount_to_cam_new') or [], 4).tolist()}"
          f" hb {np.round(r.get('head_base_to_cam_new') or [], 4).tolist()}{extra}", file=out)
print(f"\noutputs: {OUT}", file=out)
robot.disconnect()
