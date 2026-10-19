"""Offline check of the design-accurate camera mount mode on the 2026-10-08 09:51 run.

 1. With the camera_info.yaml key ignored (as for a model without it), the production Step 2 must
    still reproduce the saved result exactly.
 2. With the key as configured, the production Step 1.5 solve and Step 2 give the new result; it is
    printed next to the stage 0 prototype's numbers (proto_camera_design_20261008.py, variant B).
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\verify_camera_design_mode_20261008.py
"""
import contextlib
import io
import json
import sys
from unittest.mock import patch

import numpy as np

import step15_rerun_20261008 as step15
from core.calibration.data import format_mount_constraint, load_camera_mount_constraint
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator
from core.calibration.sequences import step2 as step2_module

step2 = step15.step2_rerun
out = sys.__stdout__

print("\n" + step2.COLUMNS, file=out)
step2.row("saved", step2.SAVED)
with patch.object(step2_module, "load_camera_mount_constraint", lambda: None):
    step2.row("key ignored", step2.solve("verify_key_ignored"), step2.SAVED)

constraint = load_camera_mount_constraint()
if constraint is None:
    raise SystemExit("config/camera_info.yaml has no mount_to_cam_position_range_m for the model in setting.yaml")
print(f"\nconstraint ({constraint['model']}): {format_mount_constraint(constraint)}", file=out)
calibrator = HeadCameraCalibrator(None, step2.robot)
calibrator.robot_version = "1.2"
with contextlib.redirect_stdout(io.StringIO()):
    first = calibrator._solve_head_camera(step15.PHASES, list(constraint["mount_to_cam"]), step15.R_NOMINAL,
                                          marker_priors=dict(step15.PRIORS),
                                          position_range_m=constraint["position_range_m"])
print(f"Step 1.5        | {step15.describe(first)} | at range end: {first['camera_at_range_end'] or 'none'}", file=out)
result = step2.solve("verify_design_mode", head=first["head_offsets_deg"])
step2.row("design mode", result)
print(f"                  {json.dumps(result['camera_mount_constraint'], default=float)}", file=out)
prototype = json.loads((step2.OUT / "result_proto_B.json").read_text(encoding="utf-8"))
print(f"largest joint difference from the stage 0 prototype: {np.abs(step2.arms(result) - step2.arms(prototype)).max():.4f} deg",
      file=out)
