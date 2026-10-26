"""Offline re-solve of a saved v1.2 head Step 2 run with the production optimize_step2: how much do the arm
offsets depend on the camera range? Also the base that the other scratch analyses re-solve Step 2 with.

Written for the 2026-10-08 17:50 run (the default without arguments). Reset baselines are not compared: the
columns are the offsets themselves and their change from the run. Kinematics: the v1.2 URDF kept in
archive/reports/step2_rerun_20261008/ (arm and head chains are the same on models A and M; FK starts at
link_torso_5). The first case must reproduce the saved result. No connection, nothing is commanded.

usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step2_rerun.py [<run folder> <timestamp>]
  another run (a folder with result_step2/ and result_txt/, read only) goes to archive/reports/step2_rerun_<date_time>/.
"""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.data import load_npz_dataset  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 2 else ROOT / "archive" / "runs" / "20261008_1750_newrobot_head"
TS = sys.argv[2] if len(sys.argv) > 2 else "20261008_175047_514053"
OUT = ROOT / "archive" / "reports" / ("step2_rerun_" + (TS[:13] if len(sys.argv) > 2 else "20261008_1750"))
OUT.mkdir(parents=True, exist_ok=True)
URDF_PATH = ROOT / "archive" / "reports" / "step2_rerun_20261008" / "model_m_v1.2.urdf"


class OfflineRobot:
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


robot = OfflineRobot(URDF_PATH.read_text(encoding="utf-8"))
SAVED = json.loads((RUN / "result_step2" / f"result_{TS}.json").read_text(encoding="utf-8"))
STEP15 = json.loads((RUN / "result_txt" / "head_camera_result_latest.json").read_text(encoding="utf-8"))
STORE = SAVED["setting_snapshot"]["joint_offset"]
CONSTRAINT = SAVED["camera_mount_constraint"]
DESIGN = list(CONSTRAINT["mount_to_cam"])
RANGE_M = np.array(CONSTRAINT["position_range_m"], dtype=float)
CAM_STEP15 = list(STEP15["calibrated_mount_to_cam"])
HEAD_STEP15 = STEP15["head_offsets_deg"]
q_arm, q_head, T_meas = load_npz_dataset(str(RUN / "result_step2" / f"dataset_{TS}.npz"))
out = sys.__stdout__


def solve(tag, range_mm=None, constraint=True, camera=None, optimize_camera=True, j6_result_delta=None):
    """range_mm: {axis: (low, high)} replacing the run's range on that camera axis (x, y, z).
    constraint=False: the mode before 2026-10-08 (camera free from `camera`, J6 anchored at Step 1)."""
    core = CalibrationCore()
    core.on_event = lambda kind, value: None
    core.bind_robot(robot, "1.2")
    core.include_head_motion = True
    core.apply_joint_offset_flag = True
    core.joint_offsets_store = {side: {k: float(STORE[side][k]) for k in ("joint3", "joint5", "joint6")}
                                for side in ("right", "left")}
    for side, delta in (j6_result_delta or {}).items():
        core.joint_offsets_store[side]["joint6"] -= float(delta)
    head = {"pan": float(HEAD_STEP15["pan"]), "tilt": float(HEAD_STEP15["tilt"])}
    core.joint_offsets_store["head"] = head
    camera = list(camera or CAM_STEP15)
    core.head_camera_calibrator.calibrated_results = {"success": True, "calibrated_mount_to_cam": camera,
                                                      "head_offsets_deg": head}
    for calibrator in core.calibrators:
        for side in ("right", "left"):
            calibrator.camera_config[f"Tf_to_marker_{side}"] = list(SAVED["ee_to_marker_used"][side])
        calibrator.camera_config["mount_to_cam"] = camera
        calibrator.camera_config["mount_to_cam_nominal"] = DESIGN
    limits = RANGE_M.copy()
    for axis, (low, high) in (range_mm or {}).items():
        limits["xyz".index(axis)] = [low / 1000.0, high / 1000.0]
    loaded = {"model": CONSTRAINT["model"], "mount_to_cam": DESIGN, "position_range_m": limits} if constraint else None
    path = OUT / f"result_{tag}.json"
    with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(step2_mod, "load_camera_mount_constraint", lambda *a, **k: loaded):
        step2_mod.optimize_step2(core, ["right", "left"], True, optimize_camera, q_arm, q_head, T_meas, str(path),
                                 lambda_cam_pos=1.0, lambda_cam_rot=1e6)
    return json.loads(path.read_text(encoding="utf-8"))


def arms(result):
    return np.array(result["right_arm_joint_offset_deg"] + result["left_arm_joint_offset_deg"])


def row(tag, result):
    values, change = arms(result), arms(result) - arms(SAVED)
    head = result.get("head_joint_offset_deg") or [float("nan")] * 2
    camera = np.array(result.get("mount_to_cam_new") or [np.nan] * 6)
    noise = result["measurement_noise"]
    print(f"{tag:14s} | R {' '.join(f'{v:+.2f}' for v in values[:7])} | L {' '.join(f'{v:+.2f}' for v in values[7:])} | "
          f"pan {head[0]:+.2f} tilt {head[1]:+.2f} | cam {camera[0] * 1000:5.2f} {camera[1] * 1000:5.2f} {camera[2] * 1000:5.2f} | "
          f"pos {noise['measurement_noise_pos_std_mm']:.3f} mm rot {noise['measurement_noise_rot_std_deg']:.3f} deg | "
          f"rms {result.get('observability', {}).get('residual_rms_sigma', float('nan')):.3f} | "
          f"max change {np.abs(change).max():.3f} (J{int(np.abs(change).argmax()) % 7})", file=out)
    return {"arm_deg": values.tolist(), "change_from_run_deg": change.tolist(), "head_deg": head,
            "mount_to_cam": camera.tolist(), "noise": noise}


def main():
    print(f"dataset {q_arm.shape}, Step 1 store R {STORE['right']} L {STORE['left']}", file=out)
    print(f"Step 1.5 camera {np.round(np.array(CAM_STEP15[:3]) * 1000, 2).tolist()} mm, head {HEAD_STEP15}", file=out)
    print("\ncolumns: arm J0..J6 offsets (deg) | head (deg) | camera x y z in link_head_2 (mm) | noise | "
          "weighted residual RMS | largest change from the run", file=out)
    summary = {"saved": row("saved", SAVED)}
    cases = [("as_run", dict())]
    cases += [(f"z+-{w:g}", dict(range_mm={"z": (-w, w)})) for w in (0.0, 2.0, 4.0, 8.0)]
    cases += [(f"x+-{w:g}", dict(range_mm={"x": (-w, w)})) for w in (0.0, 5.0)]
    cases += [("y+-2", dict(range_mm={"y": (-2.0, 2.0)}))]
    cases += [("xyz+-8", dict(range_mm={"x": (-8.0, 8.0), "y": (-8.0, 8.0), "z": (-8.0, 8.0)}))]
    cases += [("old_mode", dict(constraint=False))]
    for tag, options in cases:
        try:
            summary[tag] = row(tag, solve(tag, **options))
        except Exception as error:
            print(f"{tag:14s} | FAILED {type(error).__name__}: {error}", file=out)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\noutputs: {OUT}", file=out)


if __name__ == "__main__":
    main()
