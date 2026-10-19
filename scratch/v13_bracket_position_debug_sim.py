"""Why did the v1.3 bracket position from radii land on its limits in a simulator run?

Runs the three bracket sweeps of one arm the way Step 1 does (run_bracket_sweeps) on the v1.3
simulator, then prints for each sweep the radius and first-frame joint angles that reach
MarkerCalibrator.v13_bracket_position_from_radii, the axis and wrist centre it derives, and the result.
Simulator only (127.0.0.1:50052 by default). config/ is copied to the output folder.
--wrist-roll-offset X also re-solves with that J6 offset (deg), as the pass 1 evaluation does.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\v13_bracket_position_debug_sim.py [--addr host:port] [--side right]
         [--wrist-roll-offset X]
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import storage  # noqa: E402

OUT = ROOT / "archive" / "reports" / f"v13_bracket_position_debug_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copytree(ROOT / "config", OUT / "config")
(OUT / "config" / "home_reset_baseline.json").unlink(missing_ok=True)
storage.StoragePaths.root = OUT
for key, value in storage.CONFIG_PATHS.items():
    storage.CONFIG_PATHS[key] = str(OUT / Path(value).relative_to(ROOT))

from core.calibration import CalibrationCore  # noqa: E402
from core.calibration.sequences.step1 import run_bracket_sweeps  # noqa: E402

ADDR = sys.argv[sys.argv.index("--addr") + 1] if "--addr" in sys.argv else "127.0.0.1:50052"
SIDE = sys.argv[sys.argv.index("--side") + 1] if "--side" in sys.argv else "right"
core = CalibrationCore()
core.on_event = lambda kind, value: None
try:
    core.connect_robot(ADDR, "m", include_head=True, simulated=True)
    core.bind_robot(core.robot, "1.3")
    core.connect_camera(sim=True, robot=core.robot, robot_version="1.3")
    core.include_head_motion = True
    mc = core.marker_calibrator
    mc.include_head_motion = True
    mc.marker_problem_callback = lambda *a, **k: False
    mc.joint_offsets = {s: {"wrist_pitch": 0.0, "wrist_roll": 0.0, "wrist_yaw2": 0.0, "elbow": 0.0} for s in ("right", "left")}
    if not mc.perform_move_to_ready_pose(SIDE, log_callback=None):
        raise RuntimeError("ready pose failed")
    model = core.robot.model()
    arm_idx = list(model.left_arm_idx if SIDE == "left" else model.right_arm_idx)
    start = list(np.array(core.robot.get_state().position)[arm_idx])
    sweeps = run_bracket_sweeps(mc, SIDE, start, log=lambda m: None, emit_status=lambda *a: None,
                                save_debug=False, pass_idx=1, is_stopped=lambda: False, labels=None)
    dyn = core.robot.get_dynamics()
    for k in (4, 5, 6):
        d = sweeps[k]
        q0 = np.array(d["captured_q_full"][0])
        angles = [np.degrees(q[arm_idx[k]] - q0[arm_idx[k]]) for q in d["captured_q_full"]]
        refit = mc.fit_circle_3d_and_6dof_misalignment(d["captured_poses"], angles, robust=True)
        print(f"axis {k}: refit radius {refit['radius']:.2f} mm (rmse {refit['rmse']:.3f}), joint angle range "
              f"{min(angles):+.1f}..{max(angles):+.1f} deg, first pose {np.round(np.asarray(d['captured_poses'][0])[:3, 3] * 1000, 1)} mm")
        print(f"axis {k}: radius {d['radius']:.2f} mm, frames {len(d['captured_q_full'])}, "
              f"first-frame arm joints {np.round(np.degrees(q0[arm_idx]), 2)} deg")
    result = mc.v13_bracket_position_from_radii(sweeps[5], sweeps[6], SIDE, (67.0, 0.0, 0.0))
    print(f"result {result}")
    if "--wrist-roll-offset" in sys.argv:
        mc.joint_offsets[SIDE]["wrist_roll"] = float(sys.argv[sys.argv.index("--wrist-roll-offset") + 1])
        print(f"result with wrist_roll offset {mc.joint_offsets[SIDE]['wrist_roll']:+.2f} deg "
              f"{mc.v13_bracket_position_from_radii(sweeps[5], sweeps[6], SIDE, (67.0, 0.0, 0.0))}")
    for k, (local, link) in {4: ([0, 0, 1.0], 4), 5: ([0, 1.0, 0], 4), 6: ([1.0, 0, 0], 5)}.items():
        q = sweeps[k]["captured_q_full"][0]
        to_ee = np.linalg.inv(mc.compute_fk(core.robot, dyn, q, f"ee_{SIDE}"))
        T = mc.compute_fk(core.robot, dyn, q, f"link_{SIDE}_arm_{link}")
        print(f"axis {k}: direction in ee {np.round(to_ee[:3, :3] @ T[:3, :3] @ np.array(local), 3)}, "
              f"point {np.round((to_ee @ T[:, 3])[:3] * 1000, 2)} mm")
finally:
    core.close()
