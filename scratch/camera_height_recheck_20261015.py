"""Re-solve the two 2026-10-08 v1.2 head runs with the D405 camera height measured again: 63 mm.

The bracket height was measured at 67 mm on 2026-10-08 (the whole part); at the bolt position it is
63 mm (2026-10-15). Both v1.2 runs on the new bracket are solved again offline, Step 1.5 (head sweep)
then Step 2 on its head offsets, with the production solvers:
  as run   : the conditions of the run (reproduction check against the saved results)
  67 +-0.5 : design height 67 mm, the range in camera_info.yaml before today
  63 +-0.5 : design height 63 mm (camera_info.yaml now)
Camera x range -3/+1 mm around 43.3, y 9.0 held, angles held, as in camera_info.yaml.
Kinematics: v1.2 URDF kept in archive/reports/step2_rerun_20261008 (arm and head chains as on model A).
The 17:50 result is already applied to that robot; nothing here is applied or written to config.
Outputs: archive/reports/camera_height_recheck_20261015/.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\camera_height_recheck_20261015.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.data import load_npz_dataset  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

head_module = sys.modules["core.calibration.HeadCameraCalibrator"]
OUT = ROOT / "archive" / "reports" / "camera_height_recheck_20261015"
OUT.mkdir(parents=True, exist_ok=True)
URDF = ROOT / "archive" / "reports" / "step2_rerun_20261008" / "model_m_v1.2.urdf"
RANGE_X = (-0.003, 0.001)
RUNS = {
    "09:51 robot3 (camera free in the run)": dict(
        result=ROOT / "archive/runs/20261008_robot3_head_newbracket/result_20261008_095123_975885.json",
        dataset=ROOT / "archive/runs/20261008_robot3_head_newbracket/dataset_20261008_095123_975885.npz",
        step15=ROOT / "archive/runs/20261008_robot3_head_newbracket/head_camera_result_latest.json",
        sweep=ROOT / "archive/runs/20261008_robot3_head_newbracket/head_camera_sweep_debug.txt",
        as_run=None),
    "17:50 new robot (67 +-1 in the run)": dict(
        result=ROOT / "archive/runs/20261008_1750_newrobot_head/result_step2/result_20261008_175047_514053.json",
        dataset=ROOT / "archive/runs/20261008_1750_newrobot_head/result_step2/dataset_20261008_175047_514053.npz",
        step15=ROOT / "archive/runs/20261008_1750_newrobot_head/result_txt/head_camera_result_latest.json",
        sweep=ROOT / "archive/runs/20261008_1750_newrobot_head/result_txt/head_camera_sweep_debug.txt",
        as_run=(0.067, 0.001)),
}
CASES = [("67 +-0.5", (0.067, 0.0005)), ("63 +-0.5", (0.063, 0.0005))]
LINE = re.compile(r"(Tilt|Pan)=\s*([-+\d.]+).*?t=([\d.]+) side=(\w+) -> Marker Cam Pos: "
                  r"\[\s*([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)\]")
out = sys.__stdout__


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


robot = OfflineRobot(URDF.read_text(encoding="utf-8"))


def design(z):
    return [0.0433, 0.009, z, -90.0, 0.0, -90.0]


def constraint(z, half):
    return {"model": "D405", "mount_to_cam": design(z),
            "position_range_m": np.array([RANGE_X, (0.0, 0.0), (-half, half)], dtype=float)}


def load_phases(path):
    phases = {name: {"joint": joint, "name": name, "missing": 0, "samples": []} for name, joint in (("Tilt", 1), ("Pan", 0))}
    for line in open(path, encoding="utf-8"):
        match = LINE.search(line)
        if match:
            name, deg, t, side, x, y, z = match.groups()
            phases[name]["samples"].append({"t": float(t), "side": side, "deg": float(deg),
                                            "p": np.array([float(x), float(y), float(z)]) / 1000.0})
    for phase in phases.values():
        times = np.array([s["t"] for s in phase["samples"]])
        order = np.argsort(times)
        phase["enc_t"] = times[order]
        phase["enc_deg"] = np.array([s["deg"] for s in phase["samples"]])[order]
    return [phases["Tilt"], phases["Pan"]]


def solve_step15(run, z, half):
    saved = json.loads(run["step15"].read_text(encoding="utf-8"))
    priors = {side: (np.array(saved["marker_positions_t5_mm"][side]) - np.array(saved["marker_prior_shift_mm"][side])) / 1000.0
              for side in saved["markers_used"]}
    nominal = design(z)
    r_nom = Rotation.from_euler("ZYX", [nominal[5], nominal[4], nominal[3]], degrees=True).as_matrix()
    calibrator = head_module.HeadCameraCalibrator(None, robot)
    calibrator.robot_version = "1.2"
    with contextlib.redirect_stdout(io.StringIO()):
        return calibrator._solve_head_camera(load_phases(run["sweep"]), nominal, r_nom, marker_priors=priors,
                                             position_range_m=constraint(z, half)["position_range_m"])


def solve_step2(run, tag, camera, head, mount):
    saved = json.loads(run["result"].read_text(encoding="utf-8"))
    store = saved["setting_snapshot"]["joint_offset"]
    q_arm, q_head, T_meas = load_npz_dataset(str(run["dataset"]))
    core = CalibrationCore()
    core.on_event = lambda kind, value: None
    core.bind_robot(robot, "1.2")
    core.include_head_motion = True
    core.apply_joint_offset_flag = True
    core.joint_offsets_store = {side: {k: float(store[side][k]) for k in ("joint3", "joint5", "joint6")}
                                for side in ("right", "left")}
    head = {"pan": float(head["pan"]), "tilt": float(head["tilt"])}
    core.joint_offsets_store["head"] = head
    core.head_camera_calibrator.calibrated_results = {"success": True, "calibrated_mount_to_cam": list(camera),
                                                      "head_offsets_deg": head}
    for calibrator in core.calibrators:
        for side in ("right", "left"):
            calibrator.camera_config[f"Tf_to_marker_{side}"] = list(saved["ee_to_marker_used"][side])
        calibrator.camera_config["mount_to_cam"] = list(camera)
        calibrator.camera_config["mount_to_cam_nominal"] = list(camera)
    path = OUT / f"result_{tag}.json"
    with contextlib.redirect_stdout(io.StringIO()), patch.object(step2_mod, "load_camera_mount_constraint", lambda *a, **k: mount):
        step2_mod.optimize_step2(core, ["right", "left"], True, True, q_arm, q_head, T_meas, str(path),
                                 lambda_cam_pos=1.0, lambda_cam_rot=1e6)
    return json.loads(path.read_text(encoding="utf-8"))


def arms(result):
    return np.array(result["right_arm_joint_offset_deg"] + result["left_arm_joint_offset_deg"])


def step15_text(result):
    camera, head = result["calibrated_mount_to_cam"], result["head_offsets_deg"]
    ends = result.get("camera_at_range_end") or []
    return (f"pan {head['pan']:+.3f} tilt {head['tilt']:+.3f} | cam {camera[0] * 1000:5.2f} {camera[1] * 1000:5.2f} "
            f"{camera[2] * 1000:5.2f} mm | rmse {result['quality']['rmse_3d_marker_mm']:.3f} mm"
            + (f" | at end: {', '.join(ends)}" if ends else ""))


def step2_text(result, reference):
    values, head = arms(result), result.get("head_joint_offset_deg") or [float("nan")] * 2
    camera, noise = np.array(result["mount_to_cam_new"]), result["measurement_noise"]
    ends = (result.get("camera_mount_constraint") or {}).get("camera_at_range_end") or []
    text = (f"R {' '.join(f'{v:+.2f}' for v in values[:7])} | L {' '.join(f'{v:+.2f}' for v in values[7:])} | "
            f"pan {head[0]:+.2f} tilt {head[1]:+.2f} | cam {camera[0] * 1000:5.2f} {camera[1] * 1000:5.2f} {camera[2] * 1000:5.2f} | "
            f"pos {noise['measurement_noise_pos_std_mm']:.3f} mm | rms {result.get('observability', {}).get('residual_rms_sigma', float('nan')):.3f}")
    if reference is not None:
        change = values - arms(reference)
        text += f" | change from the run: J0 {change[0]:+.2f}/{change[7]:+.2f}, J6 {change[6]:+.2f}/{change[13]:+.2f}, max {np.abs(change).max():.2f} (J{int(np.abs(change).argmax()) % 7})"
    return text + (f" | at end: {', '.join(ends)}" if ends else "")


summary = {}
for name, run in RUNS.items():
    saved = json.loads(run["result"].read_text(encoding="utf-8"))
    saved15 = json.loads(run["step15"].read_text(encoding="utf-8"))
    print(f"\n===== {name} =====", file=out)
    print(f" saved Step 1.5          | {step15_text(saved15)}", file=out)
    print(f" saved Step 2            | {step2_text(saved, None)}", file=out)
    tag = name.split()[0].replace(":", "")
    mount = None if run["as_run"] is None else constraint(*run["as_run"])
    as_run = solve_step2(run, f"{tag}_as_run", saved15["calibrated_mount_to_cam"], saved15["head_offsets_deg"], mount)
    print(f" Step 2 as run           | {step2_text(as_run, saved)}", file=out)
    summary[name] = {"as_run": arms(as_run).tolist()}
    for case, (z, half) in CASES:
        s15 = solve_step15(run, z, half)
        s2 = solve_step2(run, f"{tag}_{case.replace(' ', '').replace('+-', 'pm')}", s15["calibrated_mount_to_cam"],
                         s15["head_offsets_deg"], constraint(z, half))
        print(f" {case:8s} Step 1.5     | {step15_text(s15)}", file=out)
        print(f" {case:8s} Step 2       | {step2_text(s2, saved)}", file=out)
        summary[name][case] = {"step15": {"head": s15["head_offsets_deg"], "camera": s15["calibrated_mount_to_cam"],
                                          "rmse_mm": s15["quality"]["rmse_3d_marker_mm"]},
                               "arm_deg": arms(s2).tolist(), "head_deg": s2.get("head_joint_offset_deg"),
                               "camera": s2["mount_to_cam_new"], "noise": s2["measurement_noise"]}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\noutputs: {OUT}", file=out)
