"""How tightly does a run's Step 1.5 pin the head tilt, and what does that tilt do to J0 and the check pose?

2026-10-19 (v1.2 robot, applied): at the symmetric check pose both hands are thought to sit 2-3 mm off in z
and the right J0 looked off. That combination (both J0, head tilt, camera height) is barely seen by Step 2,
which holds the head tilt at Step 1.5's value, so the common J0 follows Step 1.5's tilt. This re-solves the
run's Step 1.5 from its head sweeps with the production solver in the current camera mode (camera_info.yaml
range, optical-axis rotation estimated) and the tilt held at a series of values, then Step 2
(scratch/step2_rerun.py) at a few of them, and reports the right/left hand z at the v1.2 check
pose. Offline, nothing is applied. Outputs in archive/reports/step2_rerun_<date_time>/ (the Step 2 reruns).
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step15_tilt_scan.py <run folder> <timestamp>
"""
import contextlib
import io
import json
import re
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1])
m = runpy.run_path(str(ROOT / "scratch" / "step2_rerun.py"), run_name="rerun")
head_module = sys.modules["core.calibration.HeadCameraCalibrator"]
from core.calibration.data import load_camera_mount_constraint  # noqa: E402
import core.calibration.observability as obs  # noqa: E402
from core.calibration.sequences import step2 as step2_mod  # noqa: E402

out = sys.__stdout__
constraint = load_camera_mount_constraint()
DESIGN = list(constraint["mount_to_cam"])
RANGE = np.asarray(constraint["position_range_m"], dtype=float)
R_DESIGN = Rotation.from_euler("ZYX", [DESIGN[5], DESIGN[4], DESIGN[3]], degrees=True).as_matrix()
LINE = re.compile(r"(Tilt|Pan)=\s*([-+\d.]+).*?t=([\d.]+) side=(\w+) -> Marker Cam Pos: "
                  r"\[\s*([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)\]")


def load_phases(path):
    phases = {name: {"joint": joint, "name": name, "missing": 0, "samples": []} for name, joint in (("Tilt", 1), ("Pan", 0))}
    for line in open(path, encoding="utf-8"):
        match = LINE.search(line)
        if match:
            name, deg, t, side, x, y, z = match.groups()
            phases[name]["samples"].append({"t": float(t), "side": side, "deg": float(deg),
                                            "p": np.array([float(x), float(y), float(z)]) / 1000.0})
    for phase in phases.values():
        order = np.argsort([s["t"] for s in phase["samples"]])
        phase["enc_t"] = np.array([phase["samples"][i]["t"] for i in order])
        phase["enc_deg"] = np.array([phase["samples"][i]["deg"] for i in order])
    return [phases["Tilt"], phases["Pan"]]


saved15 = json.loads((RUN / "result_txt" / "head_camera_result_latest.json").read_text(encoding="utf-8"))
priors = {s: (np.array(saved15["marker_positions_t5_mm"][s]) - np.array(saved15["marker_prior_shift_mm"][s])) / 1000.0
          for s in saved15["markers_used"]}
phases = load_phases(RUN / "result_txt" / "head_camera_sweep_debug.txt")


def solve15(tilt_deg=None):
    def bounded(fun, x0, bounds, **options):
        x0, low, high = np.array(x0, float), np.array(bounds[0], float), np.array(bounds[1], float)
        if tilt_deg is not None:
            x0[1] = np.radians(tilt_deg)
            low[1], high[1] = x0[1] - 1e-9, x0[1] + 1e-9
        return least_squares(fun, x0, bounds=(low, high), **options)

    calibrator = head_module.HeadCameraCalibrator(None, m["robot"])
    calibrator.robot_version = "1.2"
    with contextlib.redirect_stdout(io.StringIO()), patch.object(head_module, "least_squares", bounded):
        return calibrator._solve_head_camera(phases, list(DESIGN), R_DESIGN, marker_priors=dict(priors),
                                             position_range_m=RANGE)


free = solve15()
t0 = free["head_offsets_deg"]["tilt"]
print(f"\nStep 1.5 free: tilt {t0:+.3f}, pan {free['head_offsets_deg']['pan']:+.3f} deg, "
      f"3D RMSE {free['quality']['rmse_3d_marker_mm']:.4f} mm (saved run: tilt {saved15['head_offsets_deg']['tilt']:+.3f})", file=out)
print("tilt held at (deg) | 3D RMSE (mm) | rise over free | pan | camera x z (mm)", file=out)
scan = {}
for d in (-1.0, -0.6, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6, 1.0):
    r = solve15(t0 + d)
    scan[d] = r
    cam = r["calibrated_mount_to_cam"]
    rmse = r["quality"]["rmse_3d_marker_mm"]
    print(f"  {t0 + d:+.3f} ({d:+.1f}) | {rmse:.4f} | {rmse - free['quality']['rmse_3d_marker_mm']:+.4f} | "
          f"{r['head_offsets_deg']['pan']:+.3f} | {cam[0] * 1000:.2f} {cam[2] * 1000:.2f}", file=out)

captured = {}
original = step2_mod.step2_weak_directions


def keep(*args, **kwargs):
    captured["args"], captured["kwargs"] = args, kwargs
    return original(*args, **kwargs)


def hands_z_at_check():
    opt, *_rest = captured["args"]
    q_arm_offset = captured["args"][4]
    reference = captured["kwargs"]["reference_q_arm"]
    return obs._hand_positions_mm(opt, reference, np.asarray(q_arm_offset, float))


print("\nStep 2 with the head tilt of each Step 1.5 (camera angles from that Step 1.5):", file=out)
base_hands = None
for d in (0.0, -0.4, 0.4):
    r = free if d == 0.0 else scan[d]
    m["HEAD_STEP15"].update({k: float(v) for k, v in r["head_offsets_deg"].items()})
    m["CAM_STEP15"][:] = r["calibrated_mount_to_cam"]
    with patch.object(step2_mod, "step2_weak_directions", keep):
        res = m["solve"](f"tilt_scan{d:+.1f}")
    hands = hands_z_at_check()
    base_hands = hands if base_hands is None else base_hands
    dz = hands - base_hands
    R, L = res["right_arm_joint_offset_deg"], res["left_arm_joint_offset_deg"]
    print(f"  tilt {r['head_offsets_deg']['tilt']:+.3f}: J0 R {R[0]:+.3f} L {L[0]:+.3f}, J1 R {R[1]:+.3f} L {L[1]:+.3f}, "
          f"pos {res['measurement_noise']['measurement_noise_pos_std_mm']:.3f} mm | check-pose hands vs tilt {t0:+.3f}: "
          f"right {np.round(dz[0:3], 2)} left {np.round(dz[3:6], 2)} mm", file=out)
