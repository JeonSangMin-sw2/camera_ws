import numpy as np
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import create_robot
import os

def check_first_sample():
    npz_path = "/home/rainbow/camera_ws/result/dataset_20260507_155725.npz"
    data = np.load(npz_path)
    q_arm = data["q_arm"][0]
    q_head = data["q_head"][0] if "q_head" in data else None
    T_meas_pair = data["marker"][0]

    # Mock robot to load model
    robot = create_robot("127.0.0.1")
    model = robot.model()
    
    # Setup nominals (copied from setting.yaml / typical config)
    mount_to_cam_nom = [0.047, 0.009, 0.057, -90.0, -1.0, -90.0]
    ee_to_marker_nom = {
        "right": [0.00000, -0.07794, -0.06671, -89.30, 0.54, 178.62],
        "left": [0.00000, 0.07760, -0.06768, -89.48, 0.20, -1.6]
    }
    ee_links = {"right": "ee_right", "left": "ee_left"}
    arm_idx = np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]])
    head_idx = model.head_idx
    
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=mount_to_cam_nom,
        ee_to_marker_nom=ee_to_marker_nom,
        ndof=22,
        head_idx=head_idx
    )
    
    # Evaluate without any offsets
    q_arm_off = np.zeros(14)
    q_head_off = np.zeros(2)
    xi_cam = np.zeros(6)
    
    for side in ["right", "left"]:
        side_idx = 0 if side == "right" else 1
        _, _, T_marker_nom, T_model = optimizer.evaluate_sample(
            q_arm, q_head, side, q_arm_off, q_head_off, xi_cam
        )
        
        T_meas = T_meas_pair[side_idx]
        
        print(f"\n--- {side.upper()} ARM ---")
        print("T_model (Translation):", np.round(T_model[:3, 3], 4))
        print("T_meas  (Translation):", np.round(T_meas[:3, 3], 4))
        
        # Calculate rotation error
        R_err = T_model[:3, :3].T @ T_meas[:3, :3]
        cos_theta = (np.trace(R_err) - 1.0) / 2.0
        angle_err = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        print(f"Orientation Error: {angle_err:.2f} degrees")

if __name__ == "__main__":
    check_first_sample()
