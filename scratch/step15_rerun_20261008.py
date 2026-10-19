"""Offline re-solve of the 2026-10-08 09:46 Step 1.5 head sweep, then Step 2 on top of each variant.

Question: how much do the head tilt (and through it J0) move when the camera mount is held at its
CAD values instead of being estimated? Step 1.5 is solved with the production _solve_head_camera on
the sweep samples in head_camera_sweep_debug.txt; a variant only narrows the bounds of that solve
(the camera rotation about its x / y axes is already fixed at nominal in production).

 free            : as in the run (reproduction check against head_camera_result_latest.json)
 ez=0            : camera rotation about the optical axis held at nominal too
 ez=0 z=cad+6    : and the camera height held at CAD plus the bracket's 6 mm
 ez=0 pos=cad+6  : and the whole camera position held there
 ez=0 z=prev+6 / pos=prev+6 : the same with the position Step 1.5 measured on the old bracket, plus 6 mm
No robot or simulator connection: the model is the URDF saved by step2_rerun_20261008.py.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step15_rerun_20261008.py
"""
import re
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import step2_rerun_20261008 as step2_rerun
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator

# The package re-exports the class under the module's name, so take the module itself.
head_module = sys.modules["core.calibration.HeadCameraCalibrator"]

out = sys.__stdout__
SAVED = step2_rerun.STEP15
NOMINAL = list(SAVED["nominal_mount_to_cam"])
R_NOMINAL = Rotation.from_euler("ZYX", [NOMINAL[5], NOMINAL[4], NOMINAL[3]], degrees=True).as_matrix()
BRACKET_RAISE_M = 0.006
# Camera position Step 1.5 gave for this camera model on the old bracket: mean of the four
# 2026-10-01 entries of head_camera_result_history.txt (the 09-29 ones are within 0.5 mm).
PREVIOUS_STEP15_M = [0.0428, 0.0077, 0.0625]
LINE = re.compile(r"(Tilt|Pan)=\s*([-+\d.]+).*?t=([\d.]+) side=(\w+) -> Marker Cam Pos: "
                  r"\[\s*([-+\d.]+),\s*([-+\d.]+),\s*([-+\d.]+)\]")


def load_phases(path):
    phases = {name: {"joint": joint, "name": name, "missing": 0, "samples": []}
              for name, joint in (("Tilt", 1), ("Pan", 0))}
    for line in open(path, encoding="utf-8"):
        match = LINE.search(line)
        if match:
            name, deg, t, side, x, y, z = match.groups()
            phases[name]["samples"].append({"t": float(t), "side": side, "deg": float(deg),
                                            "p": np.array([float(x), float(y), float(z)]) / 1000.0})
    for phase in phases.values():
        # The file has the encoder angle read with each frame, not the separate encoder history the
        # run interpolated the latency on; the frame times stand in for it.
        times = np.array([s["t"] for s in phase["samples"]])
        order = np.argsort(times)
        phase["enc_t"] = times[order]
        phase["enc_deg"] = np.array([s["deg"] for s in phase["samples"]])[order]
    return [phases["Tilt"], phases["Pan"]]


PHASES = load_phases(step2_rerun.RUN / "head_camera_sweep_debug.txt")
PRIORS = {side: (np.array(SAVED["marker_positions_t5_mm"][side]) - np.array(SAVED["marker_prior_shift_mm"][side])) / 1000.0
          for side in SAVED["markers_used"]}


def solve_step15(lock_ez=False, lock_position_delta=None, lock_z_delta=None, nominal=None, position_range_m=None):
    """nominal: camera pose the position deltas are counted from (default: the one the run used).
    position_range_m: [(low, high)] * 3 allowed camera position delta in link_head_2 axes."""
    def bounded(fun, x0, bounds, **options):
        x0, low, high = np.array(x0, dtype=float), np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)
        fixed = {}
        if lock_ez:
            fixed[0] = 0.0
        if lock_position_delta is not None:
            fixed.update({3 + i: float(v) for i, v in enumerate(lock_position_delta)})
        if lock_z_delta is not None:
            fixed[5] = float(lock_z_delta)
        for index, value in fixed.items():
            x0[index], low[index], high[index] = value, value - 1e-9, value + 1e-9
        for axis, (range_low, range_high) in enumerate(position_range_m or []):
            low[3 + axis], high[3 + axis] = range_low - 1e-9, range_high + 1e-9
            x0[3 + axis] = float(np.clip(x0[3 + axis], range_low, range_high))
        return least_squares(fun, x0, bounds=(low, high), **options)

    calibrator = HeadCameraCalibrator(None, step2_rerun.robot)
    calibrator.robot_version = "1.2"
    with patch.object(head_module, "least_squares", bounded):
        return calibrator._solve_head_camera(PHASES, list(nominal or NOMINAL), R_NOMINAL, marker_priors=dict(PRIORS))


def describe(result):
    camera, head, check = result["calibrated_mount_to_cam"], result["head_offsets_deg"], result["direction_consistency"] or {}
    return (f"tilt {head['tilt']:+.3f} (out {check.get('tilt_out_deg', float('nan')):+.2f} / back {check.get('tilt_back_deg', float('nan')):+.2f}) "
            f"pan {head['pan']:+.3f} | cam {camera[0] * 1000:5.1f} {camera[1] * 1000:4.1f} {camera[2] * 1000:5.1f} mm, "
            f"optical-axis rot {camera[4]:+.3f} deg | latency {result['latency_ms']:.1f} ms | rmse {result['quality']['rmse_3d_marker_mm']:.3f} mm")


def main():
    print(f"\nsweep samples: tilt {len(PHASES[0]['samples'])}, pan {len(PHASES[1]['samples'])} (the run solved on 300 + 300)", file=out)
    print(f"saved Step 1.5  | {describe(SAVED)}", file=out)
    step15 = {}
    for tag, options in VARIANTS:
        step15[tag] = solve_step15(**options)
        print(f"{tag:15s} | {describe(step15[tag])}", file=out)

    print("\nStep 2 on each Step 1.5 variant (camera position free in Step 2, as in production)", file=out)
    print(step2_rerun.COLUMNS, file=out)
    step2_rerun.row("saved", step2_rerun.SAVED)
    for tag, result in step15.items():
        solved = step2_rerun.solve("s15_" + tag.replace(" ", "_").replace("=", "").replace("+", "p"),
                                   camera=result["calibrated_mount_to_cam"], head=result["head_offsets_deg"])
        step2_rerun.row(tag[:12], solved)


VARIANTS = [("free", dict()),
            ("ez=0", dict(lock_ez=True)),
            ("ez=0 z=cad+6", dict(lock_ez=True, lock_z_delta=BRACKET_RAISE_M)),
            ("ez=0 pos=cad+6", dict(lock_ez=True, lock_position_delta=[0.0, 0.0, BRACKET_RAISE_M])),
            ("ez=0 z=prev+6", dict(lock_ez=True, lock_z_delta=PREVIOUS_STEP15_M[2] + BRACKET_RAISE_M - NOMINAL[2])),
            ("ez=0 pos=prev+6", dict(lock_ez=True, lock_position_delta=[
                PREVIOUS_STEP15_M[0] - NOMINAL[0], PREVIOUS_STEP15_M[1] - NOMINAL[1],
                PREVIOUS_STEP15_M[2] + BRACKET_RAISE_M - NOMINAL[2]]))]

if __name__ == "__main__":
    main()
