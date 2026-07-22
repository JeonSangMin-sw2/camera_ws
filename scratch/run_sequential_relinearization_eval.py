import os
import sys
import numpy as np

sys.path.append("/home/jsm/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer
from scratch.analyze_v12_dataset import OfflineRobot

def load_offline_robot():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        return None
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def run_relinearized_eval():
    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260722_175935.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260722_173120.npz"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    robot = load_offline_robot()
    if robot is None:
        print("Offline robot load failed.")
        return

    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())

    staged_joint_offsets = {
        "right": {"joint3": -1.2302416979621213, "joint5": -7.4496923496490615, "joint6": -5.641026190394765},
        "left": {"joint3": -1.6685655234061672, "joint5": -4.848966338794833, "joint6": 10.556777124959687}
    }

    print("=========================================================================")
    print(f"EVALUATING STEP 2 SEQUENTIAL RELINEARIZATION ON REAL DATASET")
    print("=========================================================================")

    # Stage 1: Global Rough Init (eps = 1e-5)
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

    # Stage 2: Joint Priority Refinement with Kinematics Relinearization (Camera Extrinsics Locked, eps = 1e-6)
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
        optimize_camera=False, # CAMERA LOCKED! Joint Priority!
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

    # Stage 3: Final Integration with Updated Kinematics Relinearization (eps = 1e-9)
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
    q_arm_final, q_head_final, xi_cam_final, mount_cam_new, head_cam_new = opt3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_off,
        q_head_offset_init=q_head_off,
        xi_mount_cam_init=xi_cam,
    )

    r_deg = np.rad2deg(q_arm_final[:7])
    l_deg = np.rad2deg(q_arm_final[7:14])
    h_deg = np.rad2deg(q_head_final)

    print("\n" + "="*70)
    print(" 3-STAGE RELINEARIZED OPTIMIZATION RESULTS ON REAL DATASET")
    print("="*70)
    print("--- RIGHT ARM JOINT OFFSETS (deg) ---")
    print(f"  J1 (Shoulder Pitch) : {r_deg[0]:+.4f} deg  (Target: ~ 2.0 deg)")
    print(f"  J2 (Shoulder Roll)  : {r_deg[1]:+.4f} deg")
    print(f"  J3 (Shoulder Yaw)   : {r_deg[2]:+.4f} deg")
    print(f"  J4 (Elbow - Step1)  : {r_deg[3]:+.4f} deg  (Step1 Actual: +1.2302 deg)")
    print(f"  J5 (Wrist Yaw 1)    : {r_deg[4]:+.4f} deg")
    print(f"  J6 (Wrist Pitch)    : {r_deg[5]:+.4f} deg  (Step1 Actual: +7.4497 deg)")
    print(f"  J7 (Wrist Yaw 2)    : {r_deg[6]:+.4f} deg  (Step1 Actual: +5.6410 deg)")

    print("\n--- LEFT ARM JOINT OFFSETS (deg) ---")
    print(f"  J1 (Shoulder Pitch) : {l_deg[0]:+.4f} deg")
    print(f"  J2 (Shoulder Roll)  : {l_deg[1]:+.4f} deg")
    print(f"  J3 (Shoulder Yaw)   : {l_deg[2]:+.4f} deg")
    print(f"  J4 (Elbow - Step1)  : {l_deg[3]:+.4f} deg  (Step1 Actual: +1.5686 deg)")
    print(f"  J5 (Wrist Yaw 1)    : {l_deg[4]:+.4f} deg")
    print(f"  J6 (Wrist Pitch)    : {l_deg[5]:+.4f} deg  (Step1 Actual: +4.8480 deg)")
    print(f"  J7 (Wrist Yaw 2)    : {l_deg[6]:+.4f} deg  (Step1 Actual: -10.6568 deg)")

    print("\n--- HEAD JOINT OFFSETS (deg) ---")
    print(f"  Head Pan            : {h_deg[0]:+.4f} deg")
    print(f"  Head Tilt           : {h_deg[1]:+.4f} deg")
    print("="*70)

if __name__ == "__main__":
    run_relinearized_eval()
