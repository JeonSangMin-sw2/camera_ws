"""Per-sweep scatter of the v1.3 wrist stage estimates in the simulator (G4, simulator only).

The convergence check of a joint stage averages the estimates measured near the staged offset and
stops when their standard error is <= 0.06 deg; JOINT_ESTIMATE_NOISE_DEG holds each mode's
per-sweep scatter as a floor and MIN_MEAN_SAMPLES_BY_MODE the minimum count. The v1.3 modes had
no entries (2026-10-15), so they fell back to 3 samples and no floor. This repeats one stage at a
fixed staged offset and reports the scatter of its estimate, per arm, sweep duration and noise.

Drives the SIMULATED robot (127.0.0.1:50051, v1.3). config/ is copied into the output folder and
every path points there. The raw sweep points are kept (result/result_txt) for the image-margin check.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_stage_scatter_sim.py --tag NAME
         [--position-noise-mm 0.1] [--orientation-noise-deg 0.0] [--repeats 8]
         [--stages wrist_pitch_v13:8,wrist_pitch_v13:12,wrist_roll_v13:12]
"""
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import storage  # noqa: E402


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TAG = option("--tag", "run")
OUT = ROOT / "archive" / "reports" / f"v13_stage_scatter_{TAG}_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copytree(ROOT / "config", OUT / "config")
(OUT / "config" / "home_reset_baseline.json").unlink(missing_ok=True)
truth_path = OUT / "config" / "simulation.yaml"
truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
truth["position_noise_std_m"] = float(option("--position-noise-mm", truth["position_noise_std_m"] * 1000.0)) / 1000.0
truth["orientation_noise_std_deg"] = float(option("--orientation-noise-deg", truth["orientation_noise_std_deg"]))
truth_path.write_text(yaml.safe_dump(truth, sort_keys=False), encoding="utf-8")
# --ready-j5 DEG: J5 of the v1.3 wrist_pitch ready pose (both arms); --ready-j6 DEG: J6 of the v1.3
# wrist_pitch and wrist_roll ready poses (right arm; the left arm gets -DEG), in the copied ready_poses.yaml.
if "--ready-j5" in sys.argv or "--ready-j6" in sys.argv:
    poses_path = OUT / "config" / "ready_poses.yaml"
    poses = yaml.safe_load(poses_path.read_text(encoding="utf-8"))
    for arm in ("right_arm", "left_arm"):
        if "--ready-j5" in sys.argv:
            poses["v1.3"]["joint"]["wrist_pitch"][arm][5] = float(option("--ready-j5", 0.0))
        if "--ready-j6" in sys.argv:
            for key in ("wrist_pitch", "wrist_roll"):
                poses["v1.3"]["joint"][key][arm][6] = float(option("--ready-j6", 0.0)) * (1.0 if arm == "right_arm" else -1.0)
    poses_path.write_text(yaml.safe_dump(poses, sort_keys=False), encoding="utf-8")
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402

# --arcs A B: half arcs (deg) of the wrist_pitch_v13 J6 (A) and J4 (B) sweeps; --roll-arcs A B: the
# same for wrist_roll_v13 (J6, J5). This process only.
for flag, mode in (("--arcs", "wrist_pitch_v13"), ("--roll-arcs", "wrist_roll_v13")):
    if flag in sys.argv:
        position = sys.argv.index(flag)
        BaseCalibrator.JOINT_CONFIGS[mode].update(sweep_range_A=float(sys.argv[position + 1]),
                                                  sweep_range_B=float(sys.argv[position + 2]))

# Staged at the values the 2026-10-14 simulator Full Auto converged to, so the sweeps run where the
# convergence check averages them (near the target); estimates taken 2-4 deg away scatter more.
STAGED = {"right": {"wrist_roll": -1.5307, "wrist_pitch": 1.9211},
          "left": {"wrist_roll": -4.2365, "wrist_pitch": -3.5887}}
OFFSET_KEY = {"wrist_roll_v13": "wrist_roll", "wrist_pitch_v13": "wrist_pitch"}
REPEATS = int(option("--repeats", 8))
STAGES = [(s.split(":")[0], float(s.split(":")[1]))
          for s in option("--stages", "wrist_pitch_v13:8,wrist_pitch_v13:12,wrist_roll_v13:12").split(",")]
LOG = open(OUT / "log.txt", "w", encoding="utf-8")


def log(message):
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


core = CalibrationCore()
core.on_event = lambda kind, value: None
log(f"output: {OUT}")
log(f"wrist_pitch_v13: arcs J6 +-{BaseCalibrator.JOINT_CONFIGS['wrist_pitch_v13']['sweep_range_A']:g} / "
    f"J4 +-{BaseCalibrator.JOINT_CONFIGS['wrist_pitch_v13']['sweep_range_B']:g} deg, ready J5 {option('--ready-j5', 'as in ready_poses.yaml')}; "
    f"wrist_roll_v13: J6 +-{BaseCalibrator.JOINT_CONFIGS['wrist_roll_v13']['sweep_range_A']:g} / "
    f"J5 +-{BaseCalibrator.JOINT_CONFIGS['wrist_roll_v13']['sweep_range_B']:g} deg")
log(f"noise: position {truth['position_noise_std_m'] * 1000:.3f} mm per axis, orientation "
    f"{truth['orientation_noise_std_deg']:.3f} deg per axis; repeats {REPEATS}; stages {STAGES}")
summary = {"noise": {"position_mm": truth["position_noise_std_m"] * 1000.0,
                     "orientation_deg": truth["orientation_noise_std_deg"]}, "stages": {}}
try:
    core.connect_robot(option("--addr", "127.0.0.1:50051"), "m", include_head=True, simulated=True)
    info = core.robot.get_robot_info()
    if "1.3" not in info.robot_model_version:
        raise RuntimeError(f"the simulator is {info.robot_model_version}, not v1.3")
    core.bind_robot(core.robot, "1.3")
    core.connect_camera(sim=True, robot=core.robot, robot_version="1.3")
    core.include_head_motion = True
    calibrator = core.joint_calibrator
    calibrator.include_head_motion = True
    calibrator.joint_offsets = {side: {"wrist_yaw2": 0.0, "elbow": 0.0, **STAGED[side]} for side in ("left", "right")}
    calibrator.marker_problem_callback = lambda *args, **kwargs: False
    for side in ("right", "left"):
        for mode, duration in STAGES:
            staged = STAGED[side][OFFSET_KEY[mode]]
            calibrator.perform_move_to_ready_pose(side, mode=mode, log_callback=None)
            # As perform_joint_calibration: every sweep starts from the pose read here, not from where
            # the previous sweep ended.
            arm_idx = core.robot.model().left_arm_idx if side == "left" else core.robot.model().right_arm_idx
            start_pose = list(np.array(core.robot.get_state().position)[arm_idx])
            estimates = []
            for repeat in range(REPEATS):
                res = calibrator.perform_calibration_sweep_continuous(
                    side, mode, current_offset_deg=staged, sweep_duration=duration, first_starting_pose=start_pose,
                    debug_points_note=f"{mode} {duration:g} s repeat {repeat + 1}")
                if res is None:
                    log(f"{side} {mode} {duration:g} s repeat {repeat + 1}: no result")
                    continue
                # J6 (wrist_roll_v13) reports the absolute target, J5 a correction to the staged value,
                # as in perform_joint_calibration.
                raw = float(res["optimal_offset"])
                estimates.append(raw if mode == "wrist_roll_v13" else staged + raw)
            values = np.array(estimates)
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            key = f"{side} {mode} {duration:g}s"
            summary["stages"][key] = {"estimates_deg": estimates, "std_deg": std, "mean_deg": float(np.mean(values))}
            log(f"{key:32s} n={len(values)} mean {np.mean(values):+.4f} std {std:.4f} deg | "
                f"estimates {' '.join(f'{v:+.3f}' for v in values)}")
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
finally:
    core.close()
    LOG.close()
