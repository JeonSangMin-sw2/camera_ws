import os
import sys
import numpy as np
import yaml

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer
from scratch.analyze_v12_dataset import OfflineRobot, load_offline_robot

def test_fix():
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260812_021103.npz"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    q_arm, q_head, T_meas = load_npz_dataset(dataset_path)
    robot = load_offline_robot()
    
    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
        marker_cfg = yaml.safe_load(f).get("marker", {})
    
    ee_to_marker_orig = {
        "left": marker_cfg["Tf_to_marker_left"],
        "right": marker_cfg["Tf_to_marker_right"]
    }
    
    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())
    
    # We will simulate the exact Stage 3 optimization parameters
    # Let's run with optimize_camera = True (to check get_calibrated_head_base_to_cam)
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
    )
    
    print("\nRunning optimization on dataset...")
    q_arm_off, q_head_off, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(q_arm, q_head, T_meas)
    
    print("\n==============================================")
    print("           VERIFICATION RESULTS")
    print("==============================================")
    print(f"Optimized xi_cam (perturbation):")
    print(f"  rpy: {np.round(xi_cam[:3], 4)} deg")
    print(f"  xyz: {np.round(xi_cam[3:] * 1000.0, 4)} mm")
    print("\nNew mount_to_cam:")
    print(f"  {mount_to_cam_new}")
    print("\nNew head_base_to_cam (Should be very close to nominal if xi_cam is small/zero!):")
    print(f"  {head_base_to_cam_new}")
    print("==============================================")

if __name__ == "__main__":
    test_fix()
