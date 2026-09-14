"""
Focused test: does arm joint offset recovery have a directional (sign) bias?
Head/camera are held EXACTLY at ground truth (isolating this question from the
separate head/camera-anchor cascade issue found in randomized_sweep_test.py).
Sweeps random sign COMBINATIONS (not magnitudes) across both arms' 7 joints.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk as rby
from core.calibration.calibration_core import get_both_arm_config, get_head_config
from core.calibration.calibration_optimizer import (
    QPCalibrationOptimizer, compute_fk, make_transform, prepare_q_full, D2R,
)

N_TRIALS = int(os.environ.get("N_TRIALS", "40"))
SEED = int(os.environ.get("SEED", "1"))


def fk_head2(robot, dyn_model, q_full, ee_link):
    _, T = compute_fk(robot, dyn_model, q_full, ee_link, base_link="link_head_2")
    return T


def main():
    robot = rby.create_robot("127.0.0.1:50051", "m")
    if not robot.connect():
        raise RuntimeError("Could not connect to simulator")
    model = robot.model()
    dyn_model = robot.get_dynamics()
    q_nominal = np.array(robot.get_state().position, dtype=np.float64).copy()

    cfg_both = get_both_arm_config(model, version="1.2")
    head_cfg = get_head_config(model)
    arm_idx = cfg_both["arm_idx"]
    head_idx = head_cfg["head_idx"]
    ee_to_marker_nom = cfg_both["ee_to_marker_nom"]
    nominal_mount_to_cam = cfg_both["mount_to_cam_nom"]
    nominal_mount_to_cam_T = make_transform(nominal_mount_to_cam)

    d = np.load("result/result_step2/dataset_20260913_121257.npz", allow_pickle=True)
    q_arm_list = d["q_arm"]
    q_head_list = d["q_head"]
    n_poses = q_arm_list.shape[0]

    rng_master = np.random.default_rng(SEED)
    results = []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        # Random MAGNITUDE (1-5deg) x random SIGN, independently per joint per arm
        mag = rng.uniform(1, 5, 14)
        sign = rng.choice([-1.0, 1.0], 14)
        arm_offset_true_deg_flat = mag * sign
        arm_offset_true_deg = {
            "right": arm_offset_true_deg_flat[:7],
            "left": arm_offset_true_deg_flat[7:],
        }
        # head EXACTLY at a fixed nonzero truth (not zero, to avoid a degenerate freebie)
        head_offset_true_deg = np.array([0.6, -0.9])

        arm_offset_true_rad_14 = arm_offset_true_deg_flat * D2R
        head_offset_true_rad_2 = head_offset_true_deg * D2R

        T_list = np.zeros((n_poses, 2, 4, 4))
        for i in range(n_poses):
            q_full = prepare_q_full(
                q_nominal=q_nominal, arm_idx=arm_idx,
                q_cmd=q_arm_list[i], q_offset=arm_offset_true_rad_14,
                head_idx=head_idx, q_head=q_head_list[i], q_head_offset=head_offset_true_rad_2,
            )
            for si, side in enumerate(["right", "left"]):
                T_fk = fk_head2(robot, dyn_model, q_full, f"ee_{side}")
                T_list[i, si] = np.linalg.inv(nominal_mount_to_cam_T) @ T_fk @ make_transform(ee_to_marker_nom[side])

        joint_offsets_to_apply = {
            "right": {
                "joint3": -arm_offset_true_deg["right"][3],
                "joint5": -arm_offset_true_deg["right"][5],
                "joint6": -arm_offset_true_deg["right"][6],
            },
            "left": {
                "joint3": -arm_offset_true_deg["left"][3],
                "joint5": -arm_offset_true_deg["left"][5],
                "joint6": -arm_offset_true_deg["left"][6],
            },
        }

        try:
            opt = QPCalibrationOptimizer(
                robot=robot, arm_idx=arm_idx, ee_links=cfg_both["ee_links"],
                mount_to_cam_nom=nominal_mount_to_cam, head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom, active_arms=["right", "left"],
                optimize_arm=True, optimize_head=False, optimize_camera=False,
                head_idx=head_idx, use_head_kinematics=True,
                lambda_cam_pos=1.0, lambda_cam_rot=100.0, use_sag=False,
                estimate_measurement_noise=True, apply_joint_offset_limits=True,
                joint_offsets_to_apply=joint_offsets_to_apply,
                camera_pos_bound_m=0.06, camera_rot_bound_rad=3.0 * D2R,
                eps=1e-9, max_iter=200,
            )
            qa, qh, xi, mc, hc = opt.optimize(
                q_arm_list, q_head_list, T_list, q_head_offset_init=head_offset_true_rad_2
            )
        except Exception as e:
            print(f"[trial {trial}] FAILED: {e}")
            results.append({"trial": trial, "failed": True})
            continue

        qa_deg = np.degrees(qa)
        err = np.abs(qa_deg - arm_offset_true_deg_flat)
        result = {
            "trial": trial,
            "failed": False,
            "sign": sign.tolist(),
            "true_deg": arm_offset_true_deg_flat.tolist(),
            "err_deg": err.tolist(),
            "max_err_excl_j6": float(max(err[0:6].max(), err[7:13].max())),
        }
        results.append(result)
        print(f"[trial {trial:02d}] true={np.round(arm_offset_true_deg_flat,2)} -> max_err(excl J6)={result['max_err_excl_j6']:.4f}deg")

    with open("scratch/arm_sign_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if not r["failed"]]
    print(f"\n{len(ok)}/{len(results)} trials succeeded")

    # Per-joint, split by sign: does error depend on sign direction?
    names = ["R.J0","R.J1","R.J2","R.J3","R.J4","R.J5","R.J6","L.J0","L.J1","L.J2","L.J3","L.J4","L.J5","L.J6"]
    print("\n===== Per-joint error by sign of injected offset =====")
    for j in range(14):
        if j in (3, 5, 6, 10, 12, 13):  # anchored J3/J5/J6 both arms -- skip, always near-exact by design
            continue
        pos_errs = [r["err_deg"][j] for r in ok if r["sign"][j] > 0]
        neg_errs = [r["err_deg"][j] for r in ok if r["sign"][j] < 0]
        pos_mean = np.mean(pos_errs) if pos_errs else float("nan")
        neg_mean = np.mean(neg_errs) if neg_errs else float("nan")
        print(f"  {names[j]:6s}: n_pos={len(pos_errs):2d} mean_err={pos_mean:.4f}deg  |  "
              f"n_neg={len(neg_errs):2d} mean_err={neg_mean:.4f}deg  |  "
              f"ratio(+/-)={ (pos_mean/neg_mean) if neg_mean>1e-6 else float('nan'):.2f}")

    all_errs = [r["max_err_excl_j6"] for r in ok]
    print(f"\nOverall max_err (excl J6): median={np.median(all_errs):.4f} p90={np.percentile(all_errs,90):.4f} max={np.max(all_errs):.4f}")


if __name__ == "__main__":
    main()
