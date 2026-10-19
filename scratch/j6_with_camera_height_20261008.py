"""How sharply the 2026-10-08 Step 2 data fix J6 once the camera height is known (offline).

With the camera position free, moving the J6 anchors by 1 deg changed the Step 2 residual by 12-14 %:
J4 and the camera height absorb a J6 error. Here the camera is held at the design height (67 mm, with
the x / y of the free solve) and the J6 anchors of both arms are stepped from the Step 1 orientation
value (0) through the Step 1 position value towards the reset baseline, mirrored on the two arms.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\j6_with_camera_height_20261008.py
"""
import json
import sys

import step15_rerun_20261008 as step15

step2 = step15.step2_rerun
out = sys.__stdout__
DESIGN_HEIGHT_M = 0.067
camera = list(json.loads((step2.OUT / "result_j6both_J6_position.json").read_text(encoding="utf-8"))["mount_to_cam_new"])
camera[2] = DESIGN_HEIGHT_M
print(f"\ncamera held at {[round(v * 1000, 1) for v in camera[:3]]} mm; J6 anchor step: right +d, left -d (deg, result sign)", file=out)
print("Step 1 position method is at right +0.47 / left -1.12, the reset baseline at right +0.80 / left -1.26", file=out)
print(step2.COLUMNS, file=out)
for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    result = step2.solve(f"j6z67_{d:.2f}", j6_result_delta={"right": d, "left": -d}, camera=camera, optimize_camera=False)
    step2.row(f"d={d:.2f}", result)
