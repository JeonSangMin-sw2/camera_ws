"""Step 2 of the 2026-10-08 09:51 run with the Step 1 J6 anchors of BOTH arms replaced (offline).

j0_j6_camz_20261008.py moved one arm's J6 at a time: the residual fell when the right J6 went up and
the left J6 went down by about 1 deg, and the camera height came down with it. This solves the
combined cases: the position-method J6 of Step 1 pass 2, J6 at the reset baseline, and +-1 / +-1.5 deg;
then the position-method case again with the camera held at the design height (67 mm).
No robot or simulator connection. usage: .venv\\Scripts\\python.exe -X utf8 scratch\\j6_both_arms_20261008.py
"""
import sys

import step15_rerun_20261008 as step15

step2 = step15.step2_rerun
out = sys.__stdout__
# J6 of pass 2 in joint_calib_debug_*_wrist_yaw2.txt (store sign): orientation method (used) and position method.
ORIENTATION_STORE = {"right": 0.8041, "left": -1.2778}
POSITION_STORE = {"right": 0.3332, "left": -0.1584}
AS_RUN_MINUS_BASELINE = {"right": -0.80, "left": 1.26}
DESIGN_HEIGHT_M = 0.067

# Result sign is the opposite of the store sign.
to_position = {side: -(POSITION_STORE[side] - ORIENTATION_STORE[side]) for side in ("right", "left")}
to_baseline = {side: -AS_RUN_MINUS_BASELINE[side] for side in ("right", "left")}
cases = [("as run", {}),
         ("J6 position", to_position),
         ("J6 baseline", to_baseline),
         ("R+1.0 L-1.0", {"right": 1.0, "left": -1.0}),
         ("R+1.5 L-1.5", {"right": 1.5, "left": -1.5})]
print("\n" + step2.COLUMNS, file=out)
solved = {}
for tag, delta in cases:
    solved[tag] = step2.solve("j6both_" + tag.replace(" ", "_").replace("+", "p").replace("-", "m"), j6_result_delta=delta)
    step2.row(tag[:12], solved[tag])

camera = list(solved["J6 position"]["mount_to_cam_new"])
camera[2] = DESIGN_HEIGHT_M
step2.row("pos z=67fix", step2.solve("j6both_position_z67_fixed", j6_result_delta=to_position, camera=camera,
                                     optimize_camera=False))
camera = list(step2.CAM_STEP2)
camera[2] = DESIGN_HEIGHT_M
step2.row("run z=67fix", step2.solve("j6both_as_run_z67_fixed", camera=camera, optimize_camera=False))
