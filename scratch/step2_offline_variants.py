"""Re-solve a real Step 2 dataset offline under a few variants (no real robot; nothing in result/ is written).

usage: python step2_offline_variants.py <repo root> <result json> <dataset npz> <step1.5 json> <out dir> [sim addr]
The robot model (URDF, dynamics) comes from a running simulator of the same model/version (model "m" v1.2).
The run-time settings are restored from the result's setting_snapshot into a copy of config/.
"""
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT, RESULT, DATASET, STEP15, OUT = (Path(p) for p in sys.argv[1:6])
ADDR = sys.argv[6] if len(sys.argv) > 6 else "127.0.0.1:50051"
sys.path.insert(0, str(ROOT))
OUT.mkdir(parents=True, exist_ok=True)
CFG = OUT / "config"
if CFG.exists():
    shutil.rmtree(CFG)
shutil.copytree(ROOT / "config", CFG)
result = json.loads(RESULT.read_text(encoding="utf-8"))
snap = result["setting_snapshot"]
setting_path = CFG / "setting.yaml"
setting = yaml.safe_load(setting_path.read_text(encoding="utf-8"))
for section in ("camera", "joint_offset", "marker", "step2"):
    setting.setdefault(section, {}).update(snap[section])
setting_path.write_text(yaml.safe_dump(setting, sort_keys=False, allow_unicode=True), encoding="utf-8")
camera_info_path = CFG / "camera_info.yaml"
camera_info_text = camera_info_path.read_text(encoding="utf-8")

from core import storage  # noqa: E402
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration import calibration_optimizer  # noqa: E402
from core.calibration.sequences import step2 as step2_module  # noqa: E402
from core.calibration.sequences.step2 import optimize_step2  # noqa: E402
from core.calibration.calibration_optimizer import QPCalibrationOptimizer  # noqa: E402


class FreeJ5Optimizer(QPCalibrationOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["anchor_joint5"] = False
        super().__init__(*args, **kwargs)


class PinJ24Optimizer(QPCalibrationOptimizer):
    """J2 and J4 of both arms held at 0 (+-0.01 deg), the home they had before the reset (user, 2026-11-24)."""
    def get_joint_offset_limits(self):
        lower, upper = super().get_joint_offset_limits()
        if self.optimize_arm and len(self.arm_idx) == 14:      # head-only passes have no arm bounds
            for idx in (2, 4, 9, 11):
                lower[idx], upper[idx] = np.radians(-0.01), np.radians(0.01)
        return lower, upper

data = np.load(DATASET, allow_pickle=True)
q_arm, q_head, markers = data["q_arm"], data["q_head"], data["marker"]
step15 = json.loads(STEP15.read_text(encoding="utf-8"))
logs = []
core = CalibrationCore()
core.on_event = lambda kind, value: logs.append(value) if kind == "log" else None
core.connect_robot(ADDR, "m", include_head=True, simulated=True)
core.bind_robot(core.robot, "1.2")
core.include_head_motion = True
for calibrator in core.calibrators:
    calibrator.include_head_motion = True
STORE = {"right": dict(snap["joint_offset"]["right"]), "left": dict(snap["joint_offset"]["left"]),
         "head": {"pan": step15["head_offsets_deg"]["pan"], "tilt": step15["head_offsets_deg"]["tilt"]}}


def solve(name, keep=None, head_tilt=None, j0_common_weight=2000.0, camera_range=None, tilt_weight=None,
          cam_axis_rot=None, cam_rot_lambda=1e6, free_j5=False, pin_j24=False, marker_scale=(1.0, 1.0)):
    """tilt_weight None = step2.head_tilt_anchor_weight absent from the copied setting.yaml (code default)."""
    core.joint_offsets_store = copy.deepcopy(STORE)
    run_setting = copy.deepcopy(setting)
    run_setting["step2"].pop("head_tilt_anchor_weight", None)
    if tilt_weight is not None:
        run_setting["step2"]["head_tilt_anchor_weight"] = float(tilt_weight)
    setting_path.write_text(yaml.safe_dump(run_setting, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if head_tilt is not None:
        core.joint_offsets_store["head"]["tilt"] = head_tilt
    core.head_camera_calibrator.calibrated_results = copy.deepcopy(step15)
    if cam_axis_rot is not None:  # rotation about the optical axis (middle ZYX angle), the image roll
        core.head_camera_calibrator.calibrated_results["calibrated_mount_to_cam"][4] = float(cam_axis_rot)
    for side in ("right", "left"):
        core.marker_calibrator.camera_config[f"Tf_to_marker_{side}"] = list(snap["marker"][f"Tf_to_marker_{side}"])
    calibration_optimizer.J0_COMMON_WEIGHT_REF = j0_common_weight
    text = camera_info_text
    if camera_range is not None:
        text = re.sub(r"^(\s*)mount_to_cam_position_range_m: \[\[.*$", rf"\g<1>mount_to_cam_position_range_m: {camera_range}",
                      text, count=1, flags=re.M)
        assert text != camera_info_text, "camera range line not found"
    camera_info_path.write_text(text, encoding="utf-8")
    # free_j5: v1.2 holds J5 at Step 1 (anchor_joint5); this run lets Step 2 solve it, for this process only.
    step2_module.QPCalibrationOptimizer = (PinJ24Optimizer if pin_j24 else FreeJ5Optimizer if free_j5
                                           else QPCalibrationOptimizer)
    rows = slice(None) if keep is None else keep
    # marker_scale (right, left): measured marker translations times this, i.e. a marker size / focal length error.
    T_rows = np.array(markers[rows], dtype=float, copy=True)
    for i, scale in enumerate(marker_scale):
        T_rows[:, i, :3, 3] *= scale
    logs.clear()
    value = optimize_step2(core, active_arms=["right", "left"], optimize_head=True, optimize_camera=True,
                           q_arm_list=q_arm[rows], q_head_list=q_head[rows], T_meas_list=T_rows,
                           result_path=str(OUT / f"result_{name}.json"), lambda_cam_pos=1.0, lambda_cam_rot=cam_rot_lambda)
    r, l = np.array(value["right_arm_joint_offset_deg"]), np.array(value["left_arm_joint_offset_deg"])
    head = value.get("head_joint_offset_deg")
    noise = value.get("measurement_noise", {})
    rms = next((line for line in logs if "weighted residuals, RMS" in str(line)), "")
    cam = value.get("mount_to_cam_new")
    row = {"variant": name, "n": int(q_arm[rows].shape[0]), "R": r.round(3).tolist(), "L": l.round(3).tolist(),
           "head": None if head is None else np.round(head, 3).tolist(),
           "cam_xyz_mm": None if cam is None else (np.array(cam[:3]) * 1000).round(2).tolist(),
           "pos_noise_mm": round(noise.get("measurement_noise_pos_std_mm", float("nan")), 4), "rms": str(rms).strip()}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / f"log_{name}.txt").write_text("\n".join(str(x) for x in logs), encoding="utf-8")
    return row


# Right-arm wrist poses like the three that were lost (J4 +10 or more and J5 -10 or less from the ready
# pose, which is the first sample), found among the poses that were kept.
rJ4, rJ5 = np.degrees(q_arm[:, 4] - q_arm[0, 4]), np.degrees(q_arm[:, 5] - q_arm[0, 5])
similar = [i for i in range(len(q_arm)) if rJ4[i] > 10 and rJ5[i] < -10]
print("poses like the lost ones (right dJ4 > 10, dJ5 < -10):", similar,
      [(round(rJ4[i], 1), round(rJ5[i], 1)) for i in similar], flush=True)

wide = "[[-0.006, 0.006], [-0.003, 0.003], [-0.004, 0.004]]"
wider = "[[-0.01, 0.01], [-0.006, 0.006], [-0.01, 0.01]]"
variants = {
    "A_as_run": dict(),
    "B_no_j0_common_damping": dict(j0_common_weight=0.0),
    "C_head_tilt_anchor_0": dict(head_tilt=0.0),
    "D_camera_range_wide": dict(camera_range=wide),
    "E_drop_similar_wrist": dict(keep=[i for i in range(len(q_arm)) if i not in similar]),
    "G_C_and_drop_similar": dict(head_tilt=0.0, keep=[i for i in range(len(q_arm)) if i not in similar]),
    "H_tilt_anchor_weak": dict(tilt_weight=5.0),
    "K_tilt_anchor_2e5": dict(tilt_weight=2.0e5),
    "I_tilt_anchor_off": dict(tilt_weight=0.0),
    "J_tilt_weak_camera_wide": dict(tilt_weight=5.0, camera_range=wide),
    # 2026-11-24 real v1.2 run (tilt anchor 2e5 as run): camera x/z ended at the range ends, y held at 0.
    "L_cam_wide_2e5": dict(tilt_weight=2.0e5, camera_range=wide),
    "M_cam_wider_2e5": dict(tilt_weight=2.0e5, camera_range=wider),
    "N_cam_wider_tilt_weak": dict(tilt_weight=5.0, camera_range=wider),
    # 14:45 run: left hand ~4 mm low at the check pose; is the image roll from the head-camera step off?
    "P_roll_minus_0p8": dict(tilt_weight=2.0e5, camera_range=wider, cam_axis_rot=0.2538 - 0.8),
    "Q_roll_plus_0p8": dict(tilt_weight=2.0e5, camera_range=wider, cam_axis_rot=0.2538 + 0.8),
    "R_cam_rot_free": dict(tilt_weight=2.0e5, camera_range=wider, cam_rot_lambda=1.0),
    # 2026-11-24 21:01 run: the right hand pitched 0.8 deg at the check pose; is J5 (held at Step 1 on v1.2) off?
    "S_free_j5_2e5": dict(tilt_weight=2.0e5, free_j5=True),
    # user: J2 / J4 may be off as before; hold them at the pre-reset home (0) and see what the fit says.
    "T_pin_j24_2e5": dict(tilt_weight=2.0e5, pin_j24=True),
    # 2026-11-24: the check-pose error lies along the camera depth and the camera keeps hitting its x range end;
    # is the measured depth scaled (marker size, focal length)? Both markers, then one at a time.
    "U_scale_m1.0": dict(tilt_weight=2.0e5, marker_scale=(0.990, 0.990)),
    "U_scale_m0.5": dict(tilt_weight=2.0e5, marker_scale=(0.995, 0.995)),
    "U_scale_p0.5": dict(tilt_weight=2.0e5, marker_scale=(1.005, 1.005)),
    "U_scale_p1.0": dict(tilt_weight=2.0e5, marker_scale=(1.010, 1.010)),
    "V_left_m0.5": dict(tilt_weight=2.0e5, marker_scale=(1.0, 0.995)),
    "V_left_m1.0": dict(tilt_weight=2.0e5, marker_scale=(1.0, 0.990)),
    "V_right_m1.0": dict(tilt_weight=2.0e5, marker_scale=(0.990, 1.0)),
    # 2026-11-25 second robot: camera y stopped at its -0.5 mm range end and both hands sat ~1.6 mm +y at the check
    # pose. Only the y range widened (D405 x / z ranges as in camera_info.yaml).
    "W_as_run_2e5": dict(tilt_weight=2.0e5),
    "W_cam_y1p5": dict(tilt_weight=2.0e5, camera_range="[[-0.003, 0.002], [-0.0015, 0.0015], [-0.0005, 0.0015]]"),
    "W_cam_y3": dict(tilt_weight=2.0e5, camera_range="[[-0.003, 0.002], [-0.003, 0.003], [-0.0005, 0.0015]]"),
    "W_cam_y6": dict(tilt_weight=2.0e5, camera_range="[[-0.003, 0.002], [-0.006, 0.006], [-0.0005, 0.0015]]"),
    # 2026-11-25 another v1.2 robot on rby1-calibration (11-20): its camera range before the 11-24 widening; the
    # camera x / z ended at these upper ends in Step 1.5 and Step 2 on 11-24 and 11-25.
    "Y_old_range_2e5": dict(tilt_weight=2.0e5, camera_range="[[-0.003, 0.001], [0.0, 0.0], [-0.0005, 0.0005]]"),
}
# 2026-11-25 second robot, 13:23 run: the right hand sits ~3 mm below the left at a symmetric pose though the camera
# saw 0.3 mm. Is Step 1.5's optical-axis rotation (image roll) off? Rotation relative to this run's Step 1.5 value.
for _d in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0):
    variants[f"Z_roll_{_d:+.1f}"] = dict(tilt_weight=2.0e5, cam_axis_rot=step15["calibrated_mount_to_cam"][4] + _d)
chosen = sys.argv[7].split(",") if len(sys.argv) > 7 else list(variants)
rows = [solve(name, **variants[name]) for name in chosen]
(OUT / f"variants_{'_'.join(n[0] for n in chosen)}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
core.close()
