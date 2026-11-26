"""Stepwise calibration vs. one OSQP solve over the same Step 3 (code: Step 2) data with everything free.

One-shot: both arms (14), head pan/tilt and the camera mount (6) start at zero / design values and are solved
together from the 64-pose dataset alone, with only the data term (no Step 1 values or brackets, no anchors, no
J0/J2/J4 damping, no camera regularization or rotation lock) and the production wide bounds (arm +-15 deg,
head +-20 deg, camera +-25 mm / +-10 deg). The residual, Jacobian, noise weights and OSQP call are the
production QPCalibrationOptimizer's. The stepwise solution is the saved result, re-scored with the same residual.
Offline: v1.2 URDF from archive/reports/step2_rerun_20261008, no robot or simulator connection.

usage: .venv\\Scripts\\python.exe -X utf8 scratch\\one_shot_osqp_compare.py <out dir>
"""
import contextlib
import glob
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rby1_sdk as rby

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.calibration import calibration_optimizer as co  # noqa: E402
from core.calibration.data import get_both_arm_config, get_head_config, load_npz_dataset  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
URDF_PATH = ROOT / "archive" / "reports" / "step2_rerun_20261008" / "model_m_v1.2.urdf"
# First run after each of the six resets used in the paper (Table 3).
RUNS = ["20261008_095123", "20261008_175047", "20261016_191105", "20261019_095439", "20261021_192015",
        "20261027_125837"]
D2R = np.pi / 180.0


class OfflineRobot:
    def __init__(self, urdf):
        self._model = rby.Model_M()
        count = len(self._model.robot_joint_names)
        self._dynamics = getattr(rby.dynamics, f"Robot_{count}")(rby.dynamics.load_robot_from_urdf_data(urdf, "base"))
        self._state = SimpleNamespace(position=np.zeros(count))

    def model(self):
        return self._model

    def get_dynamics(self):
        return self._dynamics

    def get_state(self):
        return self._state


class OneShotOptimizer(co.QPCalibrationOptimizer):
    """Data term only. The parent's compute_step adds Step 1 / head anchors, J2/J4 and J0 common-mode damping
    and the camera locks after the data loop; this keeps the loop and the OSQP call and drops the rest."""

    def compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
        dim = self.total_dim()
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        weights = self.noise_estimator.weights()
        samples, total_err = [], 0.0
        for q_arm, q_head, T_pair in zip(q_arm_list, q_head_list, T_meas_list):
            for side_idx, side in enumerate(co.ARM_SIDES):
                Jb, _, T_ee_to_marker, T_model = self.evaluate_sample(q_arm, q_head, side, q_arm_offset,
                                                                      q_head_offset, xi_mount_cam)
                xi = co.se3_log(np.linalg.inv(T_model) @ T_pair[side_idx])
                J = self.build_jacobian(q_arm, q_head, side, q_arm_offset, q_head_offset, xi_mount_cam, Jb,
                                        T_ee_to_marker, T_model)
                co.add_weighted_normal_equation(H, g, J, xi, weights)
                total_err += np.linalg.norm(xi)
                samples.append(xi)
        self.noise_estimator.update(samples)
        self.last_H = H.copy()
        P = 0.5 * (H + H.T) + self.qp_regularization * np.eye(dim)
        lb, ub = self._build_qp_bounds(dim, q_arm_offset, q_head_offset, xi_mount_cam)
        import qpsolvers
        import scipy.sparse as spa
        dx = qpsolvers.solve_qp(spa.csc_matrix(P), -g, lb=lb, ub=ub, solver=self.qp_solver, **self.qp_kwargs)
        if dx is None:
            raise RuntimeError("OSQP returned no solution")
        return np.asarray(dx, dtype=float).reshape(-1), total_err


def make(cls, robot, cfg, head_idx, mount_to_cam, ee_to_marker, min_rot_noise_rad, **kw):
    return cls(robot=robot, arm_idx=cfg["arm_idx"], ee_links=cfg["ee_links"], mount_to_cam_nom=list(mount_to_cam),
               head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"), ee_to_marker_nom=ee_to_marker,
               active_arms=["right", "left"], optimize_arm=True, optimize_head=True, optimize_camera=True,
               head_idx=head_idx, use_head_kinematics=True, estimate_measurement_noise=True,
               min_rot_noise_std_rad=min_rot_noise_rad, **kw)


def residuals(opt, q_arm, q_head, T_meas, arm_deg, head_deg, xi_cam):
    """Per-sample marker error (mm, deg) of the model at the given parameters, both arms."""
    pos, rot = [], []
    for qa, qh, T_pair in zip(q_arm, q_head, T_meas):
        for side_idx, side in enumerate(co.ARM_SIDES):
            _, _, _, T_model = opt.evaluate_sample(qa, qh, side, np.asarray(arm_deg) * D2R,
                                                   np.asarray(head_deg) * D2R, xi_cam)
            xi = co.se3_log(np.linalg.inv(T_model) @ T_pair[side_idx])
            rot.append(np.linalg.norm(xi[:3]) / D2R)
            pos.append(np.linalg.norm(xi[3:]) * 1000.0)
    return np.array(pos), np.array(rot)


def weak_directions(H, count=3):
    """Eigen-directions of the data Hessian in deg / mm units, weakest first."""
    names = [f"R.J{i}" for i in range(7)] + [f"L.J{i}" for i in range(7)] + ["pan", "tilt"] + \
            ["cam.rx", "cam.ry", "cam.rz", "cam.x", "cam.y", "cam.z"]
    scale = np.array([D2R] * 16 + [D2R] * 3 + [1e-3] * 3)
    Hs = scale[:, None] * H * scale[None, :]
    w, V = np.linalg.eigh(0.5 * (Hs + Hs.T))
    out = []
    for k in range(count):
        v = V[:, k]
        top = np.argsort(-np.abs(v))[:4]
        out.append({"eigenvalue": float(w[k]), "components": {names[i]: round(float(v[i]), 3) for i in top}})
    return {"condition_number": float(w[-1] / max(w[0], 1e-300)), "weakest": out}


def main():
    robot = OfflineRobot(URDF_PATH.read_text(encoding="utf-8"))
    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_idx = get_head_config(robot.model())["head_idx"]
    design_marker = cfg["ee_to_marker_nom"]
    summary = {}
    for ts in RUNS:
        result_path = glob.glob(str(ROOT / "result" / "result_step2" / f"result_{ts}*.json"))[0]
        saved = json.loads(Path(result_path).read_text(encoding="utf-8"))
        q_arm, q_head, T_meas = load_npz_dataset(glob.glob(str(ROOT / "result" / "result_step2" / f"dataset_{ts}*.npz"))[0])
        snap = saved["setting_snapshot"]
        min_rot = float(snap.get("step2", {}).get("min_rot_noise_deg", 0.3)) * D2R
        constraint = saved.get("camera_mount_constraint")
        design_cam = list(constraint["mount_to_cam"]) if constraint else list(snap["camera"]["mount_to_cam_nominal"])

        step_arm = saved["right_arm_joint_offset_deg"] + saved["left_arm_joint_offset_deg"]
        step_head = saved["head_joint_offset_deg"]
        scorer = make(co.QPCalibrationOptimizer, robot, cfg, head_idx, saved["mount_to_cam_new"],
                      saved["ee_to_marker_used"], min_rot)
        sp, sr = residuals(scorer, q_arm, q_head, T_meas, step_arm, step_head, np.zeros(6))

        one = make(OneShotOptimizer, robot, cfg, head_idx, design_cam, design_marker, min_rot,
                   lambda_cam_pos=0.0, lambda_cam_rot=0.0, lock_camera_head_axis_rotation=False,
                   camera_pos_bound_m=0.025, camera_rot_bound_rad=10.0 * D2R, max_iter=200, eps=1e-7)
        with contextlib.redirect_stdout(io.StringIO()):
            q_off, h_off, xi_cam, mount_new, _ = one.optimize(q_arm, q_head, T_meas)
        one_arm, one_head = np.rad2deg(q_off), np.rad2deg(h_off)
        op, orr = residuals(one, q_arm, q_head, T_meas, one_arm, one_head, xi_cam)

        diff = one_arm - np.array(step_arm)
        row = {
            "stepwise": {"arm_deg": step_arm, "head_deg": step_head, "mount_to_cam": saved["mount_to_cam_new"],
                         "median_mm": float(np.median(sp)), "median_deg": float(np.median(sr)),
                         "saved_noise": saved.get("measurement_noise")},
            "one_shot": {"arm_deg": one_arm.tolist(), "head_deg": one_head.tolist(), "mount_to_cam": mount_new,
                         "median_mm": float(np.median(op)), "median_deg": float(np.median(orr)),
                         "iterations": one.iteration_count},
            "arm_diff_deg": diff.tolist(), "head_diff_deg": (one_head - np.array(step_head)).tolist(),
            "max_abs_arm_diff_deg": float(np.abs(diff).max()),
            "j0_common_diff_deg": float(diff[0] + diff[7]) / 2.0,
            "hessian": weak_directions(one.last_H),
        }
        summary[ts] = row
        print(f"{ts} | stepwise {row['stepwise']['median_mm']:.3f} mm {row['stepwise']['median_deg']:.3f} deg | "
              f"one-shot {row['one_shot']['median_mm']:.3f} mm {row['one_shot']['median_deg']:.3f} deg "
              f"({one.iteration_count} it) | max |arm diff| {row['max_abs_arm_diff_deg']:.2f} deg | "
              f"head diff {np.round(row['head_diff_deg'], 2).tolist()} | cond {row['hessian']['condition_number']:.2e}")
        print("   arm diff R " + " ".join(f"{v:+.2f}" for v in diff[:7]) + " | L " + " ".join(f"{v:+.2f}" for v in diff[7:]))
        print("   weakest: " + "; ".join(f"{d['eigenvalue']:.2e} {d['components']}" for d in row["hessian"]["weakest"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
