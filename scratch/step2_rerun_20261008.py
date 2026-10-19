"""Offline re-solve of the 2026-10-08 09:51 Step 2 run (new robot, camera bracket raised 6 mm, head mode)
with the production optimize_step2, to see how the camera height, the head tilt anchor and J0 move together.

The kinematic model is the URDF of the v1.2 simulator (127.0.0.1:50052, model "m"), read once and kept in
the output folder: that simulator app exits by itself about half a minute after it starts, so the solves
run on dynamics built from the saved URDF, with no connection. Nothing is commanded.
Inputs are the backup of the run; outputs go to archive/reports/step2_rerun_20261008/.

 as_run        : anchors, Step 1.5 camera and head as in the run (reproduction check)
 cam_z=<mm>    : camera fixed (not optimized) at the run's Step 2 result, with its height set to <mm>
 tilt<delta>   : head tilt anchor moved by <delta> deg, camera free as in the run
 cam0=cad+6    : camera free, started from the CAD position plus the bracket's 6 mm
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step2_rerun_20261008.py
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unittest.mock import patch  # noqa: E402

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration import calibration_optimizer as optimizer_module  # noqa: E402
from core.calibration.data import load_npz_dataset  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

RUN = ROOT / "archive" / "runs" / "20261008_robot3_head_newbracket"
TS = "20261008_095123_975885"
OUT = ROOT / "archive" / "reports" / "step2_rerun_20261008"
OUT.mkdir(parents=True, exist_ok=True)
SIM_ADDR, SIM_MODEL = "127.0.0.1:50052", "m"

URDF_PATH = OUT / "model_m_v1.2.urdf"


def fetch_urdf():
    client = rby.create_robot(SIM_ADDR, SIM_MODEL)
    deadline = time.time() + 30.0
    while not client.connect():
        if time.time() > deadline:
            raise SystemExit(f"simulator not reachable at {SIM_ADDR} (needed once, read-only, for the URDF)")
        time.sleep(1.0)
    info = client.get_robot_info()
    if info.robot_model_name.lower() != "m" or "1.2" not in info.robot_model_version:
        raise SystemExit(f"the simulator model is {info.robot_model_name} {info.robot_model_version}, not M v1.2")
    URDF_PATH.write_text(client.get_robot_model(), encoding="utf-8")
    client.disconnect()


class OfflineRobot:
    """What the Step 2 solve asks of a robot: the model, its dynamics and a joint vector of the right
    size. FK is taken from link_torso_5, so the torso joints of that vector do not matter."""

    def __init__(self, urdf):
        self._model = rby.Model_M()
        count = len(self._model.robot_joint_names)
        self._dynamics = getattr(rby.dynamics, f"Robot_{count}")(rby.dynamics.load_robot_from_urdf_data(urdf, "base"))
        self._state = SimpleNamespace(position=np.zeros(count))

    def model(self):
        return self._model

    def get_dynamics(self):
        return self._dynamics

    def get_state(self):
        return self._state


if not URDF_PATH.exists():
    fetch_urdf()
robot = OfflineRobot(URDF_PATH.read_text(encoding="utf-8"))

SAVED = json.loads((RUN / f"result_{TS}.json").read_text(encoding="utf-8"))
STEP15 = json.loads((RUN / "head_camera_result_latest.json").read_text(encoding="utf-8"))
BASE = SAVED["home_reset_baseline"]
STORE = SAVED["setting_snapshot"]["joint_offset"]
q_arm, q_head, T_meas = load_npz_dataset(str(RUN / f"dataset_{TS}.npz"))
CAM_NOMINAL = list(STEP15["nominal_mount_to_cam"])
CAM_STEP15 = list(STEP15["calibrated_mount_to_cam"])
CAM_STEP2 = list(SAVED["mount_to_cam_new"])
HEAD_STEP15 = STEP15["head_offsets_deg"]
out = sys.__stdout__
print(f"model: {URDF_PATH.name}", file=out)
print(f"dataset {q_arm.shape}, head {np.shape(q_head)}, reset baseline {BASE['metadata']['created_at']}", file=out)
print(f"Step 1 store R {STORE['right']} L {STORE['left']}", file=out)
print(f"Step 1.5 camera {np.round(np.array(CAM_STEP15[:3]) * 1000, 2).tolist()} mm, head {HEAD_STEP15}", file=out)


def solve(tag, camera=None, optimize_camera=True, tilt_delta=0.0, head=None, j6_result_delta=None,
          j0_common_weight=None, bracket_x_mm=None):
    """j6_result_delta: {side: deg} added to the Step 1 J6 anchor, in the sign of the result JSON (the
    store has the opposite sign). j0_common_weight: replaces the optimizer's J0 common-mode damping.
    bracket_x_mm: {side: mm} added to the bracket's x (the marker's sideways position about the J6 axis)."""
    core = CalibrationCore()
    core.on_event = lambda kind, value: None
    core.bind_robot(robot, "1.2")
    core.include_head_motion = True
    core.apply_joint_offset_flag = True
    core.joint_offsets_store = {side: {k: float(STORE[side][k]) for k in ("joint3", "joint5", "joint6")}
                                for side in ("right", "left")}
    for side, delta in (j6_result_delta or {}).items():
        core.joint_offsets_store[side]["joint6"] -= float(delta)
    head = dict(head or HEAD_STEP15)
    head = {"pan": float(head["pan"]), "tilt": float(head["tilt"]) + tilt_delta}
    core.joint_offsets_store["head"] = head
    camera = list(camera or CAM_STEP15)
    core.head_camera_calibrator.calibrated_results = {"success": True, "calibrated_mount_to_cam": camera,
                                                      "head_offsets_deg": head}
    for calibrator in core.calibrators:
        for side in ("right", "left"):
            bracket = list(SAVED["ee_to_marker_used"][side])
            bracket[0] += float((bracket_x_mm or {}).get(side, 0.0)) / 1000.0
            calibrator.camera_config[f"Tf_to_marker_{side}"] = bracket
        calibrator.camera_config["mount_to_cam"] = camera
        calibrator.camera_config["mount_to_cam_nominal"] = CAM_NOMINAL
    path = OUT / f"result_{tag}.json"
    weight = optimizer_module.J0_COMMON_WEIGHT_REF if j0_common_weight is None else float(j0_common_weight)
    with contextlib.redirect_stdout(io.StringIO()), patch.object(optimizer_module, "J0_COMMON_WEIGHT_REF", weight):
        step2_mod.optimize_step2(core, ["right", "left"], True, optimize_camera, q_arm, q_head, T_meas, str(path),
                                 lambda_cam_pos=1.0, lambda_cam_rot=1e6)
    return json.loads(path.read_text(encoding="utf-8"))


def arms(result):
    return np.array(result["right_arm_joint_offset_deg"] + result["left_arm_joint_offset_deg"])


def row(tag, result, reference=None):
    delta = arms(result) - arms(BASE)
    head = result.get("head_joint_offset_deg") or [float("nan")] * 2
    camera = np.array(result.get("mount_to_cam_new") or [np.nan] * 6)
    noise = result["measurement_noise"]
    text = (f"{tag:12s} | R {' '.join(f'{v:+.2f}' for v in delta[:7])} | L {' '.join(f'{v:+.2f}' for v in delta[7:])} | "
            f"pan {head[0]:+.2f} tilt {head[1]:+.2f} | cam {camera[0] * 1000:5.1f} {camera[1] * 1000:4.1f} {camera[2] * 1000:5.1f} | "
            f"pos {noise['measurement_noise_pos_std_mm']:.3f} mm rot {noise['measurement_noise_rot_std_deg']:.3f} deg")
    if reference is not None:
        text += f" | max diff from saved {np.abs(arms(result) - arms(reference)).max():.3f} deg"
    print(text, file=out)


COLUMNS = ("columns: arm J0..J6 = result - reset baseline (deg) | head offsets (deg) | "
           "camera x y z in link_head_2 (mm) | residual noise")


def main():
    print("\n" + COLUMNS, file=out)
    row("saved", SAVED)
    summary = {}
    cases = [("as_run", dict())]
    cases += [(f"cam_z={z:.0f}", dict(camera=CAM_STEP2[:2] + [z / 1000.0] + CAM_STEP2[3:], optimize_camera=False))
              for z in (57.0, 60.0, 63.0, 66.0, 69.0, 71.1, 73.0, 76.0)]
    cases += [(f"tilt{d:+.1f}", dict(tilt_delta=d)) for d in (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0)]
    # Camera free as in the run, but started (and regularized) from the CAD position with the bracket's
    # +6 mm instead of from the Step 1.5 estimate.
    cases += [("cam0=cad+6", dict(camera=CAM_NOMINAL[:2] + [CAM_NOMINAL[2] + 0.006] + CAM_STEP15[3:]))]
    for tag, options in cases:
        try:
            result = solve(tag, **options)
        except Exception as error:
            print(f"{tag:12s} | FAILED {type(error).__name__}: {error}", file=out)
            continue
        row(tag, result, SAVED if tag == "as_run" else None)
        summary[tag] = {"arm_minus_baseline_deg": (arms(result) - arms(BASE)).tolist(),
                        "head_deg": result.get("head_joint_offset_deg"), "mount_to_cam": result.get("mount_to_cam_new"),
                        "noise": result["measurement_noise"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\noutputs: {OUT}", file=out)


if __name__ == "__main__":
    main()
