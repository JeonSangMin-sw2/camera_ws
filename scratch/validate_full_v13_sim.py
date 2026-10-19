"""Simulator run of the whole Full Auto sequence on a v1.3 robot with a head (G4).

Drives the SIMULATED robot (127.0.0.1:50051, model "m" v1.3) through the core calls the GUI makes:
exposure-check ready poses with the head at zero, the marker monitor, the Full Auto ready pose,
then CalibrationCore.run("full") = Step 1 -> Step 1.5 -> Step 2 -> post-Step-2 verification, and
the check-pose preview of the new result. The values are compared with config/simulation.yaml.

Not run: Home Offset Reset / Apply. The simulated sensor adds the truth offsets to the encoder
readings, so moving the simulator's zero would not remove them.

config/ is copied into the output folder and every path points there; the repository's config/
and result/ are not written.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\validate_full_v13_sim.py [--position-noise-mm X] [--orientation-noise-deg X]
         [--seed N] [--wrist-pitch-v13-arcs A B]
  --orientation-noise-deg / --seed: marker orientation noise and noise seed for this run only (copied simulation.yaml).
  --position-noise-mm: marker position noise of the simulated sensor for this run only (the copied
  simulation.yaml is edited; default: the value in config/simulation.yaml, 0.1 mm).
  --wrist-pitch-v13-arcs: half arcs (deg) of the J6 (A) and J4 (B) sweeps of the v1.3 J5 stage for
  this run only, to try other sweeps (set in this process; default: JOINT_CONFIGS, 22.5 and 30 since 2026-10-14).
"""
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import storage  # noqa: E402

OUT = ROOT / "archive" / "reports" / f"sim_v13_full_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copytree(ROOT / "config", OUT / "config")
# A real robot's reset baseline does not belong in a simulator result.
(OUT / "config" / "home_reset_baseline.json").unlink(missing_ok=True)
truth_path = OUT / "config" / "simulation.yaml"
truth_config = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
if "--position-noise-mm" in sys.argv:
    truth_config["position_noise_std_m"] = float(sys.argv[sys.argv.index("--position-noise-mm") + 1]) / 1000.0
if "--orientation-noise-deg" in sys.argv:
    truth_config["orientation_noise_std_deg"] = float(sys.argv[sys.argv.index("--orientation-noise-deg") + 1])
if "--seed" in sys.argv:
    truth_config["seed"] = int(sys.argv[sys.argv.index("--seed") + 1])
truth_path.write_text(yaml.safe_dump(truth_config, sort_keys=False), encoding="utf-8")
# Before the other core imports: some modules read these paths when they are imported.
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.CalibratorBase import BaseCalibrator  # noqa: E402
from core.robot.motion import AutoCollectionConfig  # noqa: E402

ADDR = sys.argv[sys.argv.index("--addr") + 1] if "--addr" in sys.argv else "127.0.0.1:50051"
MODEL, EXPECTED_VERSION = "m", "1.3"
J5_STAGE = BaseCalibrator.JOINT_CONFIGS["wrist_pitch_v13"]
if "--wrist-pitch-v13-arcs" in sys.argv:
    position = sys.argv.index("--wrist-pitch-v13-arcs")
    J5_STAGE.update(sweep_range_A=float(sys.argv[position + 1]), sweep_range_B=float(sys.argv[position + 2]))
LOG = open(OUT / "log.txt", "w", encoding="utf-8")
prompts = []


def log(message):
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def on_event(kind, value):
    if kind == "log":
        log(value)
    elif kind == "progress" and isinstance(value, dict) and value.get("stage") == "collect":
        if value["pose_index"] % 8 == 0 or value["pose_index"] == value["pose_total"]:
            log(f"[collect] pose {value['pose_index']}/{value['pose_total']}, samples {value['samples']}, "
                f"failures {value['failures']}")


def refuse_teaching(arm_side, mode=None):
    """The GUI would ask the operator to re-teach the posture here. Nobody can in this run."""
    prompts.append(f"teaching requested: {arm_side} {mode}")
    log(f"[PROMPT] Marker problem on {arm_side} ({mode}): the GUI would ask for re-teaching. Refused.")
    return False


def refuse_spacing(gap, min_gap):
    prompts.append(f"marker spacing {gap:.3f} < {min_gap:.3f} m")
    raise RuntimeError(f"markers too close at the Step 2 ready pose ({gap:.3f} < {min_gap:.3f} m)")


def step(title, name, **options):
    log(f"\n##### {title} #####")
    started = time.time()
    result = core.run(name, **options)
    log(f"##### {title}: {result.status}{' -- ' + str(result.error) if result.error else ''} "
        f"({time.time() - started:.0f} s) #####")
    return result


def pose_gap(estimate, truth):
    delta = np.linalg.inv(truth) @ BaseCalibrator.make_transform(list(estimate))
    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return delta[:3, 3] * 1000.0, float(angle)


def compare_with_truth(result_path):
    sim = core.observer.engine.simulation_model
    truth = sim.config
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summary = {"result_path": str(result_path)}
    log("\n===== RESULT vs SIMULATOR TRUTH (estimate - truth) =====")
    for side in ("right", "left"):
        estimate = np.asarray(data[f"{side}_arm_joint_offset_deg"], dtype=float)
        error = estimate - np.degrees(sim.arm_offsets(side))
        summary[f"{side}_arm_error_deg"] = error.tolist()
        log(f" {side:5s} arm J0..J6 (deg): " + " ".join(f"{v:+.3f}" for v in error)
            + f"   | worst {np.abs(error).max():.3f}")
    head = data.get("head_joint_offset_deg")
    if head is not None:
        error = np.asarray(head, dtype=float) - [truth["offsets"]["head"]["pan"], truth["offsets"]["head"]["tilt"]]
        summary["head_error_deg"] = error.tolist()
        log(f" head pan/tilt (deg): {error[0]:+.3f} {error[1]:+.3f}")
    camera = data.get("mount_to_cam_new")
    if camera is not None:
        d_pos, d_rot = pose_gap(camera, BaseCalibrator.make_transform(truth["mount_to_cam"]))
        summary["camera_error"] = {"pos_mm": d_pos.tolist(), "rot_deg": d_rot}
        log(f" camera mount: pos {np.linalg.norm(d_pos):.2f} mm ({' '.join(f'{v:+.2f}' for v in d_pos)}), rot {d_rot:.3f} deg")
    for side in ("right", "left"):
        bracket = core.get_marker_bracket(side)
        d_pos, d_rot = pose_gap(bracket, sim.bracket_transform(side))
        summary[f"{side}_bracket_error"] = {"pos_mm": d_pos.tolist(), "rot_deg": d_rot}
        log(f" {side:5s} bracket (Step 1, after symmetry): pos {np.linalg.norm(d_pos):.2f} mm "
            f"({' '.join(f'{v:+.2f}' for v in d_pos)}), rot {d_rot:.3f} deg")
    return summary


log(f"output: {OUT}")
log(f"simulated sensor noise: position {truth_config['position_noise_std_m'] * 1000.0:.3f} mm, "
    f"orientation {truth_config['orientation_noise_std_deg']:.3f} deg")
log(f"v1.3 J5 stage sweeps: J6 +-{J5_STAGE['sweep_range_A']:g} deg, J4 +-{J5_STAGE['sweep_range_B']:g} deg")
core = CalibrationCore()
core.on_event = on_event
outcome = {"output": str(OUT), "position_noise_mm": truth_config["position_noise_std_m"] * 1000.0,
           "wrist_pitch_v13_half_arcs_deg": [J5_STAGE["sweep_range_A"], J5_STAGE["sweep_range_B"]], "steps": {}}
try:
    core.connect_robot(ADDR, MODEL, include_head=True, simulated=True)
    info = core.robot.get_robot_info()
    log(f"[INFO] Connected robot model '{info.robot_model_name}', version string '{info.robot_model_version}'")
    version = "1.3" if "1.3" in info.robot_model_version else "1.2"
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"The simulator is v{version}, not v{EXPECTED_VERSION}")
    core.bind_robot(core.robot, version)
    core.connect_camera(sim=True, robot=core.robot, robot_version=version)
    outcome["head_idx"] = [int(i) for i in getattr(core.model, "head_idx", [])]
    log(f"[INFO] head joints in the model: {outcome['head_idx']}")

    # What main_ui.start_full_auto and _sync_core_state set before the worker starts.
    core.include_head_motion = True
    core.robot_address, core.robot_model_name = ADDR, MODEL
    core.auto_config = AutoCollectionConfig(angle_step_deg=5.0, position_step_m=0.03, step_x_m=0.03,
                                            max_x=0.4, max_loops=1)
    core.prompt_teaching = refuse_teaching
    core.prompt_marker_spacing = refuse_spacing
    joint_offsets = {side: {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0}
                     for side in ("left", "right")}
    core.joint_offsets_store = {"left": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
                                "right": {"joint5": 0.0, "joint6": 0.0, "joint3": 0.0},
                                "head": {"pan": 0.0, "tilt": 0.0}}
    core.head_camera_calibrator.calibrated_results = None
    for calibrator in core.calibrators:
        calibrator.include_head_motion = True
        calibrator.marker_problem_callback = refuse_teaching
    core.joint_calibrator.joint_offsets = joint_offsets
    core.marker_calibrator.joint_offsets = joint_offsets

    for side in ("right", "left"):
        result = step(f"exposure-check ready pose ({side}, head to zero)", "ready", kind="joint",
                      arm_side=side, mode="wrist_pitch_v13", head_zero=True)
        outcome["steps"][f"ready_{side}"] = result.status

    monitor = []
    core.on_event = lambda kind, value: monitor.append(value) if kind == "marker_monitor" else on_event(kind, value)
    core.start_marker_monitor()
    time.sleep(5.0)
    core.stop_marker_monitor()
    core.on_event = on_event
    last = monitor[-1] if monitor else None
    outcome["steps"]["marker_monitor"] = f"{len(monitor)} summaries"
    log(f"[monitor] {len(monitor)} summaries in 5 s; last: "
        + (json.dumps({k: v for k, v in last.items() if k in ("right", "left", "window")}, default=str) if last else "none"))

    result = step("Full Auto ready pose", "full_ready")
    outcome["steps"]["full_ready"] = result.status

    full = step("FULL AUTO (Step 1 -> Step 1.5 -> Step 2)", "full",
                step1={"save_debug": True},
                step1_5={"pan_range_deg": 10.0, "tilt_range_deg": 10.0, "num_steps": 11})
    outcome["steps"]["full"] = {"status": full.status, "error": full.error}
    for name in ("step1", "step1_5", "step2"):
        child = full.completed.get(name, full.partial.get(name))
        if child is not None:
            outcome["steps"][name] = {"status": child.status, "error": child.error}
            log(f" {name}: {child.status}{' -- ' + str(child.error) if child.error else ''}")
    log(f" joint_offsets_store after the run: {json.dumps(core.joint_offsets_store)}")

    if full.success and core.last_result_path:
        outcome["truth_comparison"] = compare_with_truth(core.last_result_path)
        outcome["result_apply_allowed"] = core.result_apply_allowed(core.last_result_path)
        result = step("check-pose preview with the new result", "home", task_type="move_check",
                      json_path=core.last_result_path, label="Optimized Result", arm="both", include_head=True)
        outcome["steps"]["move_check"] = {"status": result.status, "error": result.error}
except Exception as error:
    outcome["exception"] = f"{type(error).__name__}: {error}"
    log("[FATAL] " + traceback.format_exc())
finally:
    outcome["prompts"] = prompts
    (OUT / "summary.json").write_text(json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nsummary: {OUT / 'summary.json'}")
    try:
        core.close()
    except Exception as error:
        log(f"[WARN] close failed: {error}")
    LOG.close()
