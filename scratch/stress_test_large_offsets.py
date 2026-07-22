import os
import sys
import numpy as np

sys.path.append("/home/jsm/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import get_both_arm_config, get_head_config, compute_fk, prepare_q_full
from core.calibration_optimizer import QPCalibrationOptimizer, se3_exp
from scratch.analyze_v12_dataset import OfflineRobot

def load_offline_robot():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        return None
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def run_large_offset_stress_test():
    robot = load_offline_robot()
    if robot is None:
        print("Failed to load robot model.")
        return

    model = robot.model()
    dyn_model = robot.get_dynamics()
    
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())

    # =========================================================================
    # SETTING LARGE MOCK GROUND-TRUTH OFFSETS (J0, J1, J2, J4 > 5.0 DEGREES!)
    # =========================================================================
    # Right Arm 7 joints (deg): [J0=6.5, J1=-5.2, J2=7.0, J3=1.23, J4=-6.8, J5=7.45, J6=5.64]
    gt_right_arm_deg = np.array([6.50, -5.20, 7.00, 1.23, -6.80, 7.45, 5.64])
    
    # Left Arm 7 joints (deg): [J0=-5.5, J1=6.0, J2=-6.2, J3=1.57, J4=7.5, J5=4.85, J6=-10.56]
    gt_left_arm_deg = np.array([-5.50, 6.00, -6.20, 1.57, 7.50, 4.85, -10.56])
    
    gt_arm_offset_rad = np.radians(np.concatenate([gt_right_arm_deg, gt_left_arm_deg]))
    gt_head_offset_rad = np.radians([0.5, -0.8])

    print("=========================================================================")
    print(" STRESS TEST: LARGE MOCK GROUND-TRUTH OFFSETS (J0, J1, J2, J4 > 5.0 DEG)")
    print("=========================================================================")
    print(f" RIGHT ARM GT (deg): {gt_right_arm_deg}")
    print(f" LEFT ARM GT  (deg): {gt_left_arm_deg}")
    print("=========================================================================")

    # Generate Synthetic Dataset with Motions
    q_arm_list = []
    q_head_list = []
    T_meas_list = []

    # Nominal joint baseline
    q_right_base = np.radians([15, 20, -10, 45, 10, -20, 10])
    q_left_base = np.radians([15, -20, 10, 45, -10, -20, -10])
    q_head_base = np.radians([0, 5])

    offsets_to_sweep = [-5.0, -2.5, 0.0, 2.5, 5.0]
    
    # Generate varied joint motion poses
    for j_idx in [0, 1, 2, 4]:
        for delta in offsets_to_sweep:
            q_r = q_right_base.copy()
            q_l = q_left_base.copy()
            
            if j_idx == 2:
                q_r[j_idx] += np.radians(delta)
                q_l[j_idx] += np.radians(-delta)
            else:
                q_r[j_idx] += np.radians(delta)
                q_l[j_idx] += np.radians(delta)
                
            q_arm_cmd = np.concatenate([q_r, q_l])
            q_head_cmd = q_head_base.copy()

            # FK with GT offsets to produce true measured marker poses
            q_full_gt = prepare_q_full(
                q_nominal=np.array(robot.get_state().position),
                arm_idx=cfg["arm_idx"],
                q_cmd=q_arm_cmd,
                q_offset=gt_arm_offset_rad,
                head_idx=head_cfg["head_idx"],
                q_head=q_head_cmd,
                q_head_offset=gt_head_offset_rad
            )
            
            # Compute FK
            state = dyn_model.make_state([cfg["base_link"], cfg["ee_links"]["right"], cfg["ee_links"]["left"]], model.robot_joint_names)
            state.set_q(q_full_gt)
            dyn_model.compute_forward_kinematics(state)
            
            T_right_fk = dyn_model.compute_transformation(state, 0, 1)
            T_left_fk = dyn_model.compute_transformation(state, 0, 2)
            
            # Marker transforms
            T_meas_right = np.linalg.inv(cfg["mount_to_cam_nom"]) @ T_right_fk @ cfg["ee_to_marker_nom"]["right"]
            T_meas_left = np.linalg.inv(cfg["mount_to_cam_nom"]) @ T_left_fk @ cfg["ee_to_marker_nom"]["left"]

            q_arm_list.append(q_arm_cmd)
            q_head_list.append(q_head_cmd)
            T_meas_list.append([T_meas_right, T_meas_left])

    print(f"Generated {len(q_arm_list)} synthetic sample poses for stress test.")

    # Staged Step 1 offsets
    staged_joint_offsets = {
        "right": {"joint3": gt_right_arm_deg[3], "joint5": gt_right_arm_deg[5], "joint6": gt_right_arm_deg[6]},
        "left": {"joint3": gt_left_arm_deg[3], "joint5": gt_left_arm_deg[5], "joint6": gt_left_arm_deg[6]}
    }

    # Run 3-Stage Relinearized Optimization Workflow
    opt1 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-5,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_joint_offsets,
    )
    q_arm_off, q_head_off, xi_cam, _, _ = opt1.optimize(q_arm_list, q_head_list, T_meas_list)

    opt2 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=False,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_joint_offsets,
    )
    q_arm_off, q_head_off, _, _, _ = opt2.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    opt3 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-9,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_joint_offsets,
    )
    q_arm_final, q_head_final, xi_cam_final, _, _ = opt3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    r_est_deg = np.rad2deg(q_arm_final[:7])
    l_est_deg = np.rad2deg(q_arm_final[7:14])

    print("\n" + "="*75)
    print(" STRESS TEST RESULTS: GROUND-TRUTH vs QP 3-STAGE RELINEARIZED SOLUTION")
    print("="*75)
    print("RIGHT ARM JOINTS:")
    for j in range(7):
        print(f"  J{j} : Ground-Truth = {gt_right_arm_deg[j]:+6.2f}°  <--->  Estimated = {r_est_deg[j]:+6.4f}°   (Diff = {abs(gt_right_arm_deg[j] - r_est_deg[j]):.4f}°)")

    print("\nLEFT ARM JOINTS:")
    for j in range(7):
        print(f"  J{j} : Ground-Truth = {gt_left_arm_deg[j]:+6.2f}°  <--->  Estimated = {l_est_deg[j]:+6.4f}°   (Diff = {abs(gt_left_arm_deg[j] - l_est_deg[j]):.4f}°)")
    print("="*75)

if __name__ == "__main__":
    run_large_offset_stress_test()
