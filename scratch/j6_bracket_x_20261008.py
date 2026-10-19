"""Is the 2026-10-08 Step 2 model at odds with Step 1's own position data? (offline)

Step 1 gave two J6 values per arm: from the marker orientation (used) and from the marker position.
They differ by 0.47 deg (right) and 1.12 deg (left). With J6 taken from the orientation, that
difference says where the marker sits sideways about the J6 axis: r6 * difference = 0.44 / 1.06 mm
off the design. Step 2 was run with J6 from the orientation AND the marker at the design position
(bracket x = 0), which the position data contradict by that amount.
This re-solves Step 2 with J6 as run and the bracket x moved by that amount (both signs, one arm at
a time, then both arms with the better signs) to see whether the residual and the camera height
then match the "J6 from position" solve.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\j6_bracket_x_20261008.py
"""
import sys

import numpy as np

import step15_rerun_20261008 as step15

step2 = step15.step2_rerun
out = sys.__stdout__
R6_MM = {"right": 54.27, "left": 54.14}
ORIENTATION_STORE = {"right": 0.8041, "left": -1.2778}
POSITION_STORE = {"right": 0.3332, "left": -0.1584}
implied_mm = {side: R6_MM[side] * np.radians(POSITION_STORE[side] - ORIENTATION_STORE[side]) for side in R6_MM}
print(f"\nsideways offset implied by the two J6 values: right {implied_mm['right']:+.2f} mm, left {implied_mm['left']:+.2f} mm "
      f"(sign of the store difference; both signs are tried)", file=out)
print(step2.COLUMNS, file=out)
step2.row("as run", step2.solve("bx_as_run"))
noise = {}
for side in ("right", "left"):
    for sign in (1.0, -1.0):
        value = sign * implied_mm[side]
        result = step2.solve(f"bx_{side}_{value:+.2f}", bracket_x_mm={side: value})
        noise[side, sign] = result["measurement_noise"]["measurement_noise_pos_std_mm"]
        step2.row(f"{side[0].upper()} x{value:+.2f}", result)
best = {side: min((1.0, -1.0), key=lambda sign: noise[side, sign]) * implied_mm[side] for side in R6_MM}
step2.row("both x", step2.solve("bx_both", bracket_x_mm=best))
print(f"both x = right {best['right']:+.2f} mm, left {best['left']:+.2f} mm", file=out)
