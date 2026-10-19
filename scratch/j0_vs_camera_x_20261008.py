"""How J0 follows the camera's fore-aft position in the 2026-10-08 Step 2 data (offline).

Camera held at the design values the user gave on 2026-10-08 (y 9 mm from the baseline, height 67 mm,
nominal angles) with its x (along the optical axis; 47 mm at the module face, the optical centre is
not known) stepped, and the J6 anchors at the value the data preferred with the height fixed
(right +1.25 / left -1.25 deg from the Step 1 orientation value). A second block repeats it with the
J6 anchors as run.
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\j0_vs_camera_x_20261008.py
"""
import sys

import step15_rerun_20261008 as step15

step2 = step15.step2_rerun
out = sys.__stdout__
for title, d in (("J6 anchors moved by right +1.25 / left -1.25 deg", 1.25), ("J6 anchors as run", 0.0)):
    print(f"\n=== camera fixed at [x, 9.0, 67.0] mm, nominal angles; {title} ===", file=out)
    print(step2.COLUMNS, file=out)
    for x_mm in (41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0):
        camera = [x_mm / 1000.0, 0.009, 0.067, -90.0, 0.0, -90.0]
        result = step2.solve(f"camx_{x_mm:.0f}_d{d:.2f}", j6_result_delta={"right": d, "left": -d},
                             camera=camera, optimize_camera=False)
        step2.row(f"x={x_mm:.0f}", result)
