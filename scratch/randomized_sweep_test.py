"""
Randomized synthetic stress-test for the Step1.5 + Step2 (2-pass) calibration
pipeline. Test-code only -- does NOT modify the product SimulationModel.

Reuses REAL captured pose diversity (q_arm/q_head commands from a saved .npz
dataset) so the pose set stays representative of what the real 64-pose Step2
auto-motion plan actually produces, but synthesizes marker observations
entirely in Python for each trial (no real/simulated robot motion needed) --
so many randomized trials can run in seconds each instead of ~15-20 min for a
real end-to-end run.

Per trial, ground truth is drawn randomly:
  - arm joint offsets (7 per arm):      uniform [-5, +5] deg, independent
  - head pan/tilt offsets:              uniform [-5, +5] deg
  - joint encoder noise (per-sample):   sigma ~ uniform [0, 0.3] deg
  - camera position offset:             random direction, magnitude uniform [3, 5] cm
  - camera rotation offset:             uniform [-1, +1] deg per axis
  - marker/vision orientation noise:    sigma ~ uniform [0, 3] deg
  - marker/vision position noise:       sigma ~ uniform [0, 5] mm
  - bracket rpy/pos error (both arms):  uniform [-1,1] deg / [-2,2] mm (fixed
                                         range, not swept -- not part of the
                                         user's requested ranges)

Step 1's own arm-joint anchor is approximated as true_value + N(0, 0.03deg)
(matching its ~0.06deg convergence criterion), since JointCalibrator's own
iterative sweep fitting is not re-simulated here -- this script's focus is
Step 1.5 + Step 2, which is what changed this session. Step 1.5 itself IS
run for real (the actual _compute_head_camera_solution method) against
synthetic sweep points.
"""
import os
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk as rby
from core.calibration.calibration_core import get_both_arm_config, get_head_config
from core.calibration.calibration_optimizer import (
    QPCalibrationOptimizer, compute_fk, make_transform, prepare_q_full, D2R,
)
from core.calibration.HeadCameraCalibrator import HeadCameraCalibrator

SIM_ADDR = "127.0.0.1:50051"
SIM_MODEL = "m"
DATASET_PATH = "result/result_step2/dataset_20260913_121257.npz"
N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
SEED = int(os.environ.get("SEED", "0"))


def fk_t5(robot, dyn_model, q_full, ee_link):
    _, T = compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_torso_5")
    return T


def random_unit_vector(rng):
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def so3_perturb(rng, std_deg):
    if std_deg <= 0:
        return np.eye(3)
    return R_scipy.from_rotvec(rng.normal(0, np.deg2rad(std_deg), 3)).as_matrix()


def main():
    robot = rby.create_robot(SIM_ADDR, SIM_MODEL)
    if not robot.connect():
        raise RuntimeError("Could not connect to simulator")
    model = robot.model()
    dyn_model = robot.get_dynamics()
    num_joints = len(model.robot_joint_names)
    # Must match QPCalibrationOptimizer's own convention exactly (self.q_nominal
    # = robot.get_state().position.copy()) -- using zeros here instead caused a
    # torso-joint mismatch that broke even a noise-free round-trip test.
    q_nominal = np.array(robot.get_state().position, dtype=np.float64).copy()

    cfg_both = get_both_arm_config(model, version="1.2")
    head_cfg = get_head_config(model)
    arm_idx = cfg_both["arm_idx"]
    head_idx = head_cfg["head_idx"]
    ee_to_marker_nom = cfg_both["ee_to_marker_nom"]
    nominal_mount_to_cam = cfg_both["mount_to_cam_nom"]
    nom_t = np.array(nominal_mount_to_cam[:3], dtype=np.float64)
    R_nom = R_scipy.from_euler(
        'ZYX', [nominal_mount_to_cam[5], nominal_mount_to_cam[4], nominal_mount_to_cam[3]], degrees=True
    ).as_matrix()

    d = np.load(DATASET_PATH, allow_pickle=True)
    q_arm_list = d["q_arm"]
    q_head_list = d["q_head"]
    n_poses = q_arm_list.shape[0]
    print(f"Loaded {n_poses} real poses from {DATASET_PATH} as the pose skeleton.")

    hc_calibrator = HeadCameraCalibrator(marker_st=None, robot=robot)

    rng_master = np.random.default_rng(SEED)
    results = []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        # ---- 1. Sample random ground truth ----
        arm_offset_true_deg = {
            "right": rng.uniform(-5, 5, 7),
            "left": rng.uniform(-5, 5, 7),
        }
        head_offset_true_deg = rng.uniform(-5, 5, 2)  # [pan, tilt]
        joint_noise_std_deg = rng.uniform(0, 0.3)
        cam_pos_offset_m = random_unit_vector(rng) * rng.uniform(0.03, 0.05)
        cam_rot_offset_deg = rng.uniform(-1, 1, 3)
        marker_ori_noise_std_deg = rng.uniform(0, 3)
        marker_pos_noise_std_m = rng.uniform(0, 0.005)
        bracket_rpy_true_deg = {"right": rng.uniform(-1, 1, 3), "left": rng.uniform(-1, 1, 3)}
        bracket_pos_true_m = {"right": rng.uniform(-0.002, 0.002, 3), "left": rng.uniform(-0.002, 0.002, 3)}

        R_cam_true = R_nom @ R_scipy.from_euler('xyz', cam_rot_offset_deg, degrees=True).as_matrix()
        t_cam_true = nom_t + cam_pos_offset_m
        T_mount_to_cam_true = np.eye(4)
        T_mount_to_cam_true[:3, :3] = R_cam_true
        T_mount_to_cam_true[:3, 3] = t_cam_true

        T_ee_to_marker_true = {}
        for side in ["right", "left"]:
            T0 = make_transform(ee_to_marker_nom[side])
            Rb = R_scipy.from_euler('xyz', bracket_rpy_true_deg[side], degrees=True).as_matrix()
            T2 = T0.copy()
            T2[:3, :3] = Rb @ T0[:3, :3]
            T2[:3, 3] = T0[:3, 3] + bracket_pos_true_m[side]
            T_ee_to_marker_true[side] = T2

        arm_offset_true_rad_14 = np.concatenate(
            [arm_offset_true_deg["right"], arm_offset_true_deg["left"]]
        ) * D2R
        head_offset_true_rad_2 = head_offset_true_deg * D2R

        def true_q_full(q_arm_cmd, q_head_cmd, add_joint_noise=True):
            arm_noise = rng.normal(0, joint_noise_std_deg * D2R, 14) if add_joint_noise else 0.0
            head_noise = rng.normal(0, joint_noise_std_deg * D2R, 2) if add_joint_noise else 0.0
            return prepare_q_full(
                q_nominal=q_nominal, arm_idx=arm_idx,
                q_cmd=q_arm_cmd, q_offset=arm_offset_true_rad_14 + arm_noise,
                head_idx=head_idx, q_head=q_head_cmd,
                q_head_offset=head_offset_true_rad_2 + head_noise,
            )

        # ---- 2. Synthesize Step 1.5 sweep points (real solver, synthetic data) ----
        active_side = "right"
        q_arm_ready = q_arm_list[0]  # stationary arm pose stand-in
        pan_angles_deg = np.linspace(-15.0, 15.0, 11)
        tilt_angles_deg = np.linspace(-20.0, 20.0, 11)
        pts_tilt_cam, pts_pan_cam = [], []
        for t_deg in tilt_angles_deg:
            q_full = true_q_full(q_arm_ready, np.array([0.0, np.deg2rad(t_deg)]))
            T_t5_marker = fk_t5(robot, dyn_model, q_full, f"ee_{active_side}") @ T_ee_to_marker_true[active_side]
            T_t5_cam = fk_t5(robot, dyn_model, q_full, "link_head_2") @ T_mount_to_cam_true
            p_cam = np.linalg.inv(T_t5_cam) @ np.append(T_t5_marker[:3, 3], 1.0)
            p_cam = p_cam[:3] + rng.normal(0, marker_pos_noise_std_m, 3)
            pts_tilt_cam.append(p_cam)
        for p_deg in pan_angles_deg:
            q_full = true_q_full(q_arm_ready, np.array([np.deg2rad(p_deg), 0.0]))
            T_t5_marker = fk_t5(robot, dyn_model, q_full, f"ee_{active_side}") @ T_ee_to_marker_true[active_side]
            T_t5_cam = fk_t5(robot, dyn_model, q_full, "link_head_2") @ T_mount_to_cam_true
            p_cam = np.linalg.inv(T_t5_cam) @ np.append(T_t5_marker[:3, 3], 1.0)
            p_cam = p_cam[:3] + rng.normal(0, marker_pos_noise_std_m, 3)
            pts_pan_cam.append(p_cam)

        P_marker_t5_nom = fk_t5(robot, dyn_model, true_q_full(q_arm_ready, np.zeros(2), add_joint_noise=False),
                                 f"ee_{active_side}")[:3, 3]

        head_res = hc_calibrator._compute_head_camera_solution(
            pts_tilt_cam, pts_pan_cam, tilt_angles_deg.tolist(), pan_angles_deg.tolist(),
            nominal_mount_to_cam, R_nom, active_side=active_side,
            P_marker_t5_nom=P_marker_t5_nom, log_callback=None,
        )
        mount_cam_init = head_res["calibrated_mount_to_cam"]
        step1_5_pan = head_res["head_offsets_deg"]["pan"]
        step1_5_tilt = head_res["head_offsets_deg"]["tilt"]

        # ---- 3. Synthesize Step 1's own (noisy) J3/J5/J6 anchor ----
        step1_residual_std_deg = 0.03
        joint_offsets_to_apply = {"right": {}, "left": {}}
        for side in ["right", "left"]:
            for jname, jidx in [("joint3", 3), ("joint5", 5), ("joint6", 6)]:
                true_val = arm_offset_true_deg[side][jidx]
                noisy_val = true_val + rng.normal(0, step1_residual_std_deg)
                # anchor code negates: target = -jo[...] -- pre-negate so the
                # anchor target equals the (noisy) true offset in q_arm_offset's
                # own additive convention.
                joint_offsets_to_apply[side][jname] = -noisy_val
        joint_offsets_to_apply["head"] = {"pan": step1_5_pan, "tilt": step1_5_tilt}
        q_head_offset_init = np.radians([step1_5_pan, step1_5_tilt])

        # ---- 4. Synthesize Step 2's 64-pose marker dataset ----
        T_list = np.zeros((n_poses, 2, 4, 4))
        for i in range(n_poses):
            q_full = true_q_full(q_arm_list[i], q_head_list[i])
            T_t5_cam = fk_t5(robot, dyn_model, q_full, "link_head_2") @ T_mount_to_cam_true
            T_cam_inv_rot = T_t5_cam[:3, :3].T
            T_cam_inv_trans = -T_cam_inv_rot @ T_t5_cam[:3, 3]
            for si, side in enumerate(["right", "left"]):
                T_t5_marker = fk_t5(robot, dyn_model, q_full, f"ee_{side}") @ T_ee_to_marker_true[side]
                p_cam = T_cam_inv_rot @ T_t5_marker[:3, 3] + T_cam_inv_trans
                R_cam_marker = T_cam_inv_rot @ T_t5_marker[:3, :3]
                p_cam += rng.normal(0, marker_pos_noise_std_m, 3)
                R_cam_marker = so3_perturb(rng, marker_ori_noise_std_deg) @ R_cam_marker
                T_list[i, si, :3, :3] = R_cam_marker
                T_list[i, si, :3, 3] = p_cam
                T_list[i, si, 3, 3] = 1.0

        # ---- 5. Run the real 2-pass Step 2 optimizer ----
        try:
            optimizer_pass1 = QPCalibrationOptimizer(
                robot=robot, arm_idx=arm_idx, ee_links=cfg_both["ee_links"],
                mount_to_cam_nom=mount_cam_init, head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom, active_arms=["right", "left"],
                optimize_arm=True, optimize_head=True, optimize_camera=True,
                head_idx=head_idx, use_head_kinematics=True,
                lambda_cam_pos=1.0, lambda_cam_rot=100.0, use_sag=False,
                estimate_measurement_noise=True, apply_joint_offset_limits=True,
                joint_offsets_to_apply=joint_offsets_to_apply,
                camera_pos_bound_m=0.06, camera_rot_bound_rad=3.0 * D2R,
                eps=1e-7, max_iter=200,
            )
            qa, qh_p1, xi_p1, _, _ = optimizer_pass1.optimize(
                q_arm_list, q_head_list, T_list, q_head_offset_init=q_head_offset_init
            )
            optimizer_pass2 = QPCalibrationOptimizer(
                robot=robot, arm_idx=arm_idx, ee_links=cfg_both["ee_links"],
                mount_to_cam_nom=mount_cam_init, head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom, active_arms=["right", "left"],
                optimize_arm=False, optimize_head=True, optimize_camera=True,
                head_idx=head_idx, use_head_kinematics=True,
                lambda_cam_pos=1.0, lambda_cam_rot=100.0, use_sag=False,
                estimate_measurement_noise=True, apply_joint_offset_limits=True,
                joint_offsets_to_apply=joint_offsets_to_apply,
                camera_pos_bound_m=0.06, camera_rot_bound_rad=3.0 * D2R,
                eps=1e-7, max_iter=200,
            )
            qa, qh, xi, mount_to_cam_new, _ = optimizer_pass2.optimize(
                q_arm_list, q_head_list, T_list,
                q_arm_offset_init=qa, q_head_offset_init=qh_p1, xi_mount_cam_init=xi_p1,
            )
        except Exception as e:
            print(f"[trial {trial}] FAILED: {e}")
            results.append({"trial": trial, "failed": True, "error": str(e)})
            continue

        qa_deg = np.degrees(qa)
        qh_deg = np.degrees(qh)
        arm_err = {
            "right": np.abs(qa_deg[0:7] - arm_offset_true_deg["right"]),
            "left": np.abs(qa_deg[7:14] - arm_offset_true_deg["left"]),
        }
        head_err = np.abs(qh_deg - head_offset_true_deg)
        cam_pos_err = np.abs(np.array(mount_to_cam_new[:3]) - t_cam_true)
        R_cam_calc = R_scipy.from_euler(
            'ZYX', [mount_to_cam_new[5], mount_to_cam_new[4], mount_to_cam_new[3]], degrees=True
        ).as_matrix()
        rot_err_deg = np.degrees(
            np.arccos(np.clip((np.trace(R_cam_calc.T @ R_cam_true) - 1) / 2, -1, 1))
        )

        result = {
            "trial": trial,
            "failed": False,
            "params": {
                "joint_noise_std_deg": joint_noise_std_deg,
                "cam_pos_offset_cm": float(np.linalg.norm(cam_pos_offset_m) * 100),
                "cam_rot_offset_deg": cam_rot_offset_deg.tolist(),
                "marker_ori_noise_std_deg": marker_ori_noise_std_deg,
                "marker_pos_noise_std_mm": marker_pos_noise_std_m * 1000,
            },
            "arm_err_deg": {"right": arm_err["right"].tolist(), "left": arm_err["left"].tolist()},
            "arm_err_max_excl_j6": float(max(arm_err["right"][:6].max(), arm_err["left"][:6].max())),
            "j6_err_deg": {"right": float(arm_err["right"][6]), "left": float(arm_err["left"][6])},
            "head_err_deg": head_err.tolist(),
            "cam_pos_err_mm": (cam_pos_err * 1000).tolist(),
            "cam_rot_err_deg": float(rot_err_deg),
        }
        results.append(result)
        print(f"[trial {trial:02d}] noise(joint={joint_noise_std_deg:.3f}deg cam_pos={np.linalg.norm(cam_pos_offset_m)*100:.1f}cm "
              f"marker_ori={marker_ori_noise_std_deg:.2f}deg marker_pos={marker_pos_noise_std_m*1000:.2f}mm) "
              f"-> arm_max(excl J6)={result['arm_err_max_excl_j6']:.3f}deg "
              f"head_err={head_err[0]:.3f}/{head_err[1]:.3f}deg cam_rot_err={rot_err_deg:.3f}deg cam_pos_err={cam_pos_err.max()*1000:.2f}mm")

    out_path = "scratch/randomized_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} trial results to {out_path}")

    ok = [r for r in results if not r["failed"]]
    if ok:
        arm_maxes = [r["arm_err_max_excl_j6"] for r in ok]
        head_errs = np.array([r["head_err_deg"] for r in ok])
        cam_rot_errs = [r["cam_rot_err_deg"] for r in ok]
        cam_pos_errs = [max(r["cam_pos_err_mm"]) for r in ok]
        print("\n===== SUMMARY =====")
        print(f"Trials: {len(ok)}/{len(results)} succeeded")
        print(f"Arm error (excl J6), deg: median={np.median(arm_maxes):.3f} p90={np.percentile(arm_maxes,90):.3f} max={np.max(arm_maxes):.3f}")
        print(f"Head Pan error, deg:      median={np.median(head_errs[:,0]):.3f} p90={np.percentile(head_errs[:,0],90):.3f} max={np.max(head_errs[:,0]):.3f}")
        print(f"Head Tilt error, deg:     median={np.median(head_errs[:,1]):.3f} p90={np.percentile(head_errs[:,1],90):.3f} max={np.max(head_errs[:,1]):.3f}")
        print(f"Camera rotation error, deg: median={np.median(cam_rot_errs):.3f} p90={np.percentile(cam_rot_errs,90):.3f} max={np.max(cam_rot_errs):.3f}")
        print(f"Camera position error, mm: median={np.median(cam_pos_errs):.3f} p90={np.percentile(cam_pos_errs,90):.3f} max={np.max(cam_pos_errs):.3f}")


if __name__ == "__main__":
    main()
