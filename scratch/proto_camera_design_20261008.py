"""Stage 0 prototype on the 2026-10-08 09:51 run: camera held near its design pose, J6 left to Step 2.

Design under test (user decisions 2026-10-08): D405 camera at x 43.3 mm (module face 47 mm minus the
3.7 mm depth start point), y 9 mm, height 67 mm, all three camera angles at nominal; the position may
move by x +1 / -3 mm and height +-1 mm; J6 is no longer anchored to the Step 1 value in Step 2.
Production code is not changed: the Step 1.5 and Step 2 solves are the production ones with their
bounds narrowed in this process, and the J6 anchor term taken back out of the QP before it is solved.
Variants show what each part does. No robot or simulator connection.
usage: .venv\\Scripts\\python.exe -X utf8 scratch\\proto_camera_design_20261008.py
"""
import sys
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

import step15_rerun_20261008 as step15
from core.calibration.calibration_optimizer import QPCalibrationOptimizer

step2 = step15.step2_rerun
optimizer_module = step2.optimizer_module
out = sys.__stdout__
D2R = np.pi / 180.0
DESIGN = [0.0433, 0.009, 0.067, -90.0, 0.0, -90.0]
RANGE_X_M, RANGE_Z_M = (-0.003, 0.001), (-0.001, 0.001)
J6_ANCHOR_WEIGHT = 1e7          # calibration_optimizer.compute_step
J6_INDEX = {"right": 6, "left": 13}
R_DESIGN = step15.R_NOMINAL


@contextmanager
def step2_prototype(position_range_m=None, free_j6=False, j6_start_deg=None):
    """position_range_m: [(low, high)] * 3 camera position delta from the camera passed to the solve,
    in link_head_2 axes; all three camera rotations are then held. free_j6: J6 bounds opened to the
    other joints' +-15 deg and the Step 1 anchor removed. j6_start_deg: {side: deg} J6 start of Pass 1."""
    build_bounds, offset_limits = QPCalibrationOptimizer._build_qp_bounds, QPCalibrationOptimizer.get_joint_offset_limits
    compute_step, optimize, solve_qp = QPCalibrationOptimizer.compute_step, QPCalibrationOptimizer.optimize, optimizer_module.qpsolvers.solve_qp
    current = {}

    def bounds(self, dim, q_arm_offset, q_head_offset, xi):
        low, high = build_bounds(self, dim, q_arm_offset, q_head_offset, xi)
        if self.optimize_camera and position_range_m is not None:
            # xi is in the camera frame; the two columns are the low and the high corner of the box.
            corners = R_DESIGN.T @ np.array(position_range_m)
            rotation, position = slice(dim - 6, dim - 3), slice(dim - 3, dim)
            low[rotation] = high[rotation] = -xi[:3]
            low[position], high[position] = corners.min(axis=1) - xi[3:], corners.max(axis=1) - xi[3:]
        return low, high

    def limits(self):
        low, high = offset_limits(self)
        if free_j6 and self.optimize_arm and len(self.active_arms) == 2:
            for index in J6_INDEX.values():
                low[index], high[index] = -15.0 * D2R, 15.0 * D2R
        return low, high

    def step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi):
        current.update(optimizer=self, q_arm_offset=np.array(q_arm_offset, dtype=float))
        return compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi)

    def qp(P, q, lb=None, ub=None, **options):
        optimizer = current.get("optimizer")
        if free_j6 and optimizer is not None and optimizer.optimize_arm:
            P, q = P.tolil(), np.array(q, dtype=float)
            for side, index in J6_INDEX.items():
                target = -optimizer.joint_offsets_to_apply[side]["joint6"] * D2R
                P[index, index] -= J6_ANCHOR_WEIGHT
                q[index] -= J6_ANCHOR_WEIGHT * (current["q_arm_offset"][index] - target)
            P = P.tocsc()
        return solve_qp(P, q, lb=lb, ub=ub, **options)

    def run(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset_init=None, q_head_offset_init=None, xi_mount_cam_init=None):
        if j6_start_deg is not None and self.optimize_arm and q_arm_offset_init is None:
            q_arm_offset_init = np.zeros(len(self.arm_idx))
            for side, index in J6_INDEX.items():
                q_arm_offset_init[index] = j6_start_deg[side] * D2R
        return optimize(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset_init, q_head_offset_init, xi_mount_cam_init)

    with patch.object(QPCalibrationOptimizer, "_build_qp_bounds", bounds), \
            patch.object(QPCalibrationOptimizer, "get_joint_offset_limits", limits), \
            patch.object(QPCalibrationOptimizer, "compute_step", step), \
            patch.object(QPCalibrationOptimizer, "optimize", run), \
            patch.object(optimizer_module.qpsolvers, "solve_qp", qp):
        yield


def at_bound(camera, position_range_m, centre):
    notes = []
    for axis, name in enumerate("xyz"):
        low, high = position_range_m[axis]
        delta = camera[axis] - centre[axis]
        if high - low > 1e-6 and (delta - low < 2e-5 or high - delta < 2e-5):
            notes.append(f"{name} at its {'lower' if delta - low < 2e-5 else 'upper'} bound")
    return ", ".join(notes) or "inside the range"


def run(tag, y_range_m=(0.0, 0.0), free_j6=True, j6_start_deg=None, step2_x="range"):
    box = [RANGE_X_M, y_range_m, RANGE_Z_M]
    first = step15.solve_step15(lock_ez=True, nominal=DESIGN, position_range_m=box)
    camera15, head = first["calibrated_mount_to_cam"], first["head_offsets_deg"]
    print(f"\n{tag}", file=out)
    print(f"  Step 1.5: {step15.describe(first)} | {at_bound(camera15, box, DESIGN)}", file=out)
    if step2_x == "step1.5":
        centre = [camera15[0], DESIGN[1], DESIGN[2]] + DESIGN[3:]
        box2 = [(0.0, 0.0), y_range_m, RANGE_Z_M]
    else:
        centre, box2 = list(DESIGN), box
    with step2_prototype(position_range_m=box2, free_j6=free_j6, j6_start_deg=j6_start_deg):
        result = step2.solve("proto_" + tag.split(":")[0].strip(), camera=centre, head=head)
    step2.row("  Step 2", result)
    print(f"            camera {at_bound(result['mount_to_cam_new'], box2, centre)}", file=out)
    return result


print("\n" + step2.COLUMNS, file=out)
step2.row("as run", step2.SAVED)
run("A: camera near design, J6 anchored to Step 1 (orientation)", free_j6=False)
run("B: camera near design, J6 free  <- the design under test")
run("C: as B, J6 started from the Step 1 orientation values",
    j6_start_deg={"right": -0.8041, "left": 1.2778})
run("D: as B, camera y free by +-1 mm", y_range_m=(-0.001, 0.001))
run("E: as B, Step 2 keeps the x Step 1.5 found", step2_x="step1.5")
