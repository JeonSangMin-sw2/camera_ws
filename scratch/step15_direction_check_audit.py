"""Are the out-only / back-only tilt values of the 2026-10-08 Step 1.5 run real estimates?

Step 1.5 reports a "direction consistency" check: the same solve on the outbound half and on the
return half of the head sweeps. On this run they read tilt -0.09 / -0.13 deg while the full solve
read -1.655 deg, with a HIGHER residual on each half (0.29 / 0.28 mm) than the full solve has
overall (0.19 mm). A half-data optimum cannot fit its own half worse than the full solution does,
so this checks whether those half solves stopped short: it evaluates the full solution on each half
and re-runs each half solve starting from the full solution.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\step15_direction_check_audit.py
"""
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

import step15_rerun_20261008 as step15

out = sys.__stdout__
calls = []


def recording(fun, x0, bounds, **options):
    solution = least_squares(fun, x0, bounds=bounds, **options)
    calls.append((fun, bounds, options, solution))
    return solution


def rmse_mm(fun, x):
    residual = fun(x).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) * 1000.0)


calibrator = step15.HeadCameraCalibrator(None, step15.step2_rerun.robot)
calibrator.robot_version = "1.2"
with patch.object(step15.head_module, "least_squares", recording):
    calibrator._solve_head_camera(step15.PHASES, step15.NOMINAL, step15.R_NOMINAL, marker_priors=dict(step15.PRIORS))

full_fun, full_bounds, full_options, full = calls[0]
print(f"\nfull solve      : tilt {np.degrees(full.x[1]):+.3f} deg, pan {np.degrees(full.x[2]):+.3f} deg, "
      f"rmse {rmse_mm(full_fun, full.x):.3f} mm ({full.nfev} evaluations, status {full.status})", file=out)
# The same question for the full solve itself: does it move when continued, or when started elsewhere?
longer = least_squares(full_fun, full.x, bounds=full_bounds, **{**full_options, "max_nfev": 20000})
print(f"full continued  : tilt {np.degrees(longer.x[1]):+.3f} deg, pan {np.degrees(longer.x[2]):+.3f} deg, "
      f"rmse {rmse_mm(full_fun, longer.x):.4f} mm ({longer.nfev} evaluations, status {longer.status})", file=out)
for tilt_start in (0.0, 1.0):
    start = full.x.copy()
    start[1] = np.radians(tilt_start)
    other = least_squares(full_fun, start, bounds=full_bounds, **{**full_options, "max_nfev": 20000})
    print(f"full from tilt {tilt_start:+.1f}: tilt {np.degrees(other.x[1]):+.3f} deg, pan {np.degrees(other.x[2]):+.3f} deg, "
          f"rmse {rmse_mm(full_fun, other.x):.4f} mm ({other.nfev} evaluations, status {other.status})", file=out)
# How sharply the sweep data fix the tilt: hold it at a value, solve everything else again.
print("tilt held at (deg) | rmse (mm) | camera x y z (mm) | pan (deg)", file=out)
for tilt_deg in (-3.0, -2.5, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.5):
    low, high, start = np.array(full_bounds[0]), np.array(full_bounds[1]), full.x.copy()
    start[1] = np.radians(tilt_deg)
    low[1], high[1] = start[1] - 1e-9, start[1] + 1e-9
    held = least_squares(full_fun, start, bounds=(low, high), **{**full_options, "max_nfev": 20000})
    camera = (np.array(step15.NOMINAL[:3]) + held.x[3:6]) * 1000.0
    print(f"tilt {tilt_deg:+5.1f}         | {rmse_mm(full_fun, held.x):.4f}    | {camera[0]:5.1f} {camera[1]:4.1f} {camera[2]:5.1f} | "
          f"{np.degrees(held.x[2]):+.3f} (status {held.status}, {held.nfev} evaluations)", file=out)

for name, (fun, bounds, options, solution) in zip(("out half", "back half"), calls[1:3]):
    low, high = np.array(bounds[0]), np.array(bounds[1])
    start = np.clip(full.x, low, high)
    again = least_squares(fun, start, bounds=bounds, **options)
    print(f"{name:9s} as run: tilt {np.degrees(solution.x[1]):+.3f} deg, pan {np.degrees(solution.x[2]):+.3f} deg, "
          f"rmse {rmse_mm(fun, solution.x):.3f} mm ({solution.nfev} evaluations, status {solution.status})", file=out)
    print(f"{'':9s} full solution on this half: rmse {rmse_mm(fun, start):.3f} mm", file=out)
    print(f"{'':9s} re-solved from the full solution: tilt {np.degrees(again.x[1]):+.3f} deg, "
          f"pan {np.degrees(again.x[2]):+.3f} deg, rmse {rmse_mm(fun, again.x):.3f} mm", file=out)
