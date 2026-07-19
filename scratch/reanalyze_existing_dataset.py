import os
import sys
import numpy as np
import yaml

sys.path.append("/home/jsm/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer
from scratch.analyze_v12_dataset import OfflineRobot

def load_offline_robot():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def main():
    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260716_205134.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} does not exist.")
        return

    print("======================================================")
    print(f"RE-ANALYZING REAL ROBOT DATASET: {os.path.basename(dataset_path)}")
    print("======================================================")

    q_arm, q_head, T_meas = load_npz_dataset(dataset_path)
    robot = load_offline_robot()

    model = robot.model()
    # Torso calibration pose angles
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    with open("/home/jsm/camera_ws/config/setting.yaml", "r") as f:
        camera_cfg = yaml.safe_load(f).get("camera", {})

    ee_to_marker_orig = {
        "left": camera_cfg["Tf_to_marker_left"],
        "right": camera_cfg["Tf_to_marker_right"]
    }

    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())
    active_arms = ["right", "left"]
    ee_links = cfg["ee_links"]

    # 1. Old Way: Camera Optimized (optimize_camera=True, optimize_head=True)
    optimizer_old = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        lambda_cam_pos=0.0,
        lambda_cam_rot=0.0,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    print("\n--- RUN 1: OLD METHOD (CAMERA FREE, OPTIMIZE HEAD) ---")
    q_arm_offset_old, q_head_offset_old, xi_cam_old, _, _ = optimizer_old.optimize(q_arm, q_head, T_meas)

    # 2. New Way: Camera Locked (optimize_camera=False, optimize_head=True)
    optimizer_new = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=False,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    print("\n--- RUN 2: NEW METHOD (CAMERA LOCKED to CAD, OPTIMIZE HEAD) ---")
    q_arm_offset_new, q_head_offset_new, xi_cam_new, _, _ = optimizer_new.optimize(q_arm, q_head, T_meas)

    print("\n=======================================================")
    print("COMPARATIVE CALIBRATION RESULTS ON REAL DATA")
    print("=======================================================")
    print(f"--- Head Offsets ---")
    print(f"  Old Method (Cam Free)   : Pan: {np.rad2deg(q_head_offset_old[0]):+.4f} deg, Tilt: {np.rad2deg(q_head_offset_old[1]):+.4f} deg")
    print(f"  New Method (Cam Locked) : Pan: {np.rad2deg(q_head_offset_new[0]):+.4f} deg, Tilt: {np.rad2deg(q_head_offset_new[1]):+.4f} deg")
    
    print(f"\n--- Camera Mount to Camera Extrinsics ---")
    print(f"  CAD Nominal             : [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]")
    cam_rpy_old = xi_cam_old[:3]
    cam_xyz_old = xi_cam_old[3:] * 1000.0
    print(f"  Old Method (Cam Free)   : Pos: X={cam_xyz_old[0]:+.2f}, Y={cam_xyz_old[1]:+.2f}, Z={cam_xyz_old[2]:+.2f} mm")
    print(f"                            Rot: R={cam_rpy_old[0]:+.2f}, P={cam_rpy_old[1]:+.2f}, Y={cam_rpy_old[2]:+.2f} deg")
    
    cam_rpy_new = xi_cam_new[:3]
    cam_xyz_new = xi_cam_new[3:] * 1000.0
    print(f"  New Method (Cam Locked) : Pos: X={cam_xyz_new[0]:+.2f}, Y={cam_xyz_new[1]:+.2f}, Z={cam_xyz_new[2]:+.2f} mm")
    print(f"                            Rot: R={cam_rpy_new[0]:+.2f}, P={cam_rpy_new[1]:+.2f}, Y={cam_rpy_new[2]:+.2f} deg")

if __name__ == "__main__":
    main()
