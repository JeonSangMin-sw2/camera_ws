import sys
import os
import json
import glob
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk as rby
from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, prepare_q_full, compute_fk
from core.calibration_core import get_both_arm_config, get_head_config

def test_decoupling():
    # 1. Instantiate simulated robot
    robot = rby.create_robot("127.0.0.1", "m")
    # Since we cannot connect, let's mock it but with real kinematics from the saved dataset if possible,
    # or let's just inspect the math.
    # Wait, can we mock robot model and dyn_model by wrapping a dummy?
    # Actually, we can load the actual dataset we just collected!
    # The dataset file: /home/rainbow/camera_ws/result/result_step2/dataset_20260720_114630.npz
    # It has the actual q_arm, q_head, and marker poses collected with the new motion plan!
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260720_114630.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found!")
        return
        
    data = np.load(dataset_path)
    q_arm_list = data["q_arm"]
    q_head_list = data["q_head"]
    T_meas_list = data["marker"]
    
    print(f"Loaded dataset: q_arm={q_arm_list.shape}, q_head={q_head_list.shape}, T_meas={T_meas_list.shape}")
    
    # Wait, the simulated measurements in the dataset were generated using the true bracket poses:
    # Right nominal: [0.0, -0.054, -0.048, 90.0, 0.0, 180.0] + GT bracket: pos [0.0005, 0.0, 0.002], rpy [-0.1, -0.1, 0.05]
    # Let's compute the true bracket values:
    # Right true bracket:
    # Pos: [0.0005, -0.054, -0.046]
    # RPY: [89.9, -0.1, 180.05] (approx)
    
    # Let's run the optimizer on the loaded dataset, but with TWO cases:
    # Case A: Using the memory bracket values (which has mismatch)
    # Case B: Using the EXACT ground truth bracket values (which has zero mismatch)
    
    # Let's print out the results and compare them!
    # Wait, to run the optimizer, we need the robot and dyn_model.
    # Since we cannot call robot.connect(), can we mock the connect to return True?
    # No, the C++ library's connect() actually tries to establish a gRPC channel.
    # But wait, is the simulated robot server currently listening on port 50051?
    # Yes! In our previous ss command, we saw:
    # tcp LISTEN 0 4096 0.0.0.0:50051
    # So the simulated robot server IS running!
    # The reason it failed in our previous script is that the UI was connected.
    # If the UI is currently idle, maybe we can connect?
    # Let's try!
    try:
        robot = rby.create_robot("127.0.0.1", "m")
        if robot.connect():
            print("Connected to robot successfully!")
        else:
            print("Failed to connect to robot.")
            return
    except Exception as e:
        print(f"Exception connecting: {e}")
        return
        
    try:
        model = robot.model()
        cfg = get_both_arm_config(model, version="1.2")
        head_cfg = get_head_config(model)
        
        # Ground Truth bracket values
        # Nominals:
        # right: [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
        # left: [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
        # MOCK_GT_OFFSETS bracket:
        # right: pos [0.0005, 0.0, 0.002], rpy [-0.1, -0.1, 0.05]
        # left: pos [0.001, 0.0005, -0.002], rpy [0.1, 0.1, 0.0]
        
        # Let's build the exact true bracket values:
        # We can construct the 4x4 matrix, then convert back to vector if needed, or check how make_transform does it.
        # Let's construct it:
        # T_nominal @ T_offset
        T_r_nom = make_transform([0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
        # T_offset is:
        T_r_off = make_transform([0.0005, 0.0, 0.002, -0.1, -0.1, 0.05])
        T_r_gt = T_r_off @ T_r_nom # Wait, in get_simulated_marker_pose:
        # marker_pos_gt = np.array(nominal_pos) + np.array(bracket_pos_gt)
        # R_ee_m_gt = R_bracket_offset @ R_ee_m_ideal
        # So:
        # T_ee_to_marker_gt = [R_bracket_offset @ R_ee_m_ideal, nominal_pos + bracket_pos]
        # Let's convert this T_ee_to_marker_gt to [x, y, z, roll, pitch, yaw] ZYX euler:
        from scipy.spatial.transform import Rotation as R_scipy
        
        def get_vector_from_T(T):
            pos = T[:3, 3]
            rpy = R_scipy.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)
            # euler returns [yaw, pitch, roll]
            return [pos[0], pos[1], pos[2], rpy[2], rpy[1], rpy[0]]
            
        T_r_gt_mat = np.eye(4)
        T_r_gt_mat[:3, :3] = R_scipy.from_euler('ZYX', [0.05, -0.1, -0.1], degrees=True).as_matrix() @ R_scipy.from_euler('ZYX', [180.0, 0.0, 90.0], degrees=True).as_matrix()
        T_r_gt_mat[:3, 3] = [0.0005, -0.054, -0.046]
        r_gt_vec = get_vector_from_T(T_r_gt_mat)
        
        T_l_gt_mat = np.eye(4)
        T_l_gt_mat[:3, :3] = R_scipy.from_euler('ZYX', [0.0, 0.1, 0.1], degrees=True).as_matrix() @ R_scipy.from_euler('ZYX', [0.0, 0.0, 90.0], degrees=True).as_matrix()
        T_l_gt_mat[:3, 3] = [0.001, 0.0545, -0.050]
        l_gt_vec = get_vector_from_T(T_l_gt_mat)
        
        print(f"Calculated GT vector right: {r_gt_vec}")
        print(f"Calculated GT vector left: {l_gt_vec}")
        
        # Case A: Memory brackets
        ee_to_marker_nom_mem = {
            "right": [0.0, -0.0538, -0.0454, 90.0, 0.1, 180.0],
            "left": [0.0, 0.0545, -0.0502, 90.04, 0.11, 0.0]
        }
        
        # Case B: GT brackets
        ee_to_marker_nom_gt = {
            "right": r_gt_vec,
            "left": l_gt_vec
        }
        
        for name, ee_to_marker_nom in [("Case A (Memory Brackets)", ee_to_marker_nom_mem), ("Case B (GT Brackets)", ee_to_marker_nom_gt)]:
            print(f"\n=== Running Optimizer for {name} ===")
            
            optimizer = QPCalibrationOptimizer(
                robot=robot,
                arm_idx=cfg["arm_idx"],
                ee_links=cfg["ee_links"],
                mount_to_cam_nom=cfg["mount_to_cam_nom"],
                head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
                ee_to_marker_nom=ee_to_marker_nom,
                head_idx=head_cfg["head_idx"],
                lambda_cam_pos=1.0,
                lambda_cam_rot=1.0,
                use_sag=False,
                optimize_head=True,
                optimize_camera=False,
                active_arms=["right", "left"],
                estimate_measurement_noise=True,
                apply_joint_offset_limits=False, # No joint offset limits to see true convergence!
            )
            
            q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
                q_arm_list,
                q_head_list,
                T_meas_list,
            )
            
            r_est = np.rad2deg(q_arm_offset[:7])
            l_est = np.rad2deg(q_arm_offset[7:])
            h_est = np.rad2deg(q_head_offset)
            
            MOCK_GT_OFFSETS = {
                "right": np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3]),
                "left": np.array([-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5]),
                "head": np.array([0.8, -1.5])
            }
            
            r_err = r_est - MOCK_GT_OFFSETS["right"]
            l_err = l_est - MOCK_GT_OFFSETS["left"]
            h_err = h_est - MOCK_GT_OFFSETS["head"]
            
            print(f"Right arm joint offset: {r_est}")
            print(f"Right arm errors: {r_err}")
            print(f"Right max error: {np.max(np.abs(r_err)):.6f} deg")
            print(f"Left arm joint offset: {l_est}")
            print(f"Left arm errors: {l_err}")
            print(f"Left max error: {np.max(np.abs(l_err)):.6f} deg")
            print(f"Head joint offset: {h_est}")
            print(f"Head errors: {h_err}")
            print(f"Head max error: {np.max(np.abs(h_err)):.6f} deg")
            
    finally:
        robot.disconnect()

if __name__ == "__main__":
    test_decoupling()
