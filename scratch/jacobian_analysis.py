import numpy as np
import os
import sys

sys.path.insert(0, '/home/rainbow/camera_ws')
from core.calibration_optimizer import QPCalibrationOptimizer
import rby1_sdk as rby
from core.calibration_core import get_both_arm_config, get_head_config

def main():
    dataset_path = '/home/rainbow/camera_ws/result/result_real/result_step2/dataset_20260716_205134.npz'
    data = np.load(dataset_path, allow_pickle=True)
    q_arm_list = data['q_arm']
    q_head_list = data['q_head'] if 'q_head' in data else None
    marker_list = data['marker']
    
    print(f"Loaded dataset: {q_arm_list.shape} samples.")
    
    try:
        robot = rby.create_robot("127.0.0.1:50051", "m")
        if not robot.connect():
            print("Failed to connect to robot!")
            return
            
        cfg = get_both_arm_config(robot.model(), version="1.2")
        head_cfg = get_head_config(robot.model())
        
        # Instantiate optimizer with both optimize_head and optimize_camera
        opt = QPCalibrationOptimizer(
            robot=robot,
            arm_idx=cfg["arm_idx"],
            ee_links=cfg["ee_links"],
            mount_to_cam_nom=cfg["mount_to_cam_nom"],
            head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
            ee_to_marker_nom=cfg["ee_to_marker_nom"],
            head_idx=head_cfg["head_idx"],
            lambda_cam_pos=0.0,
            lambda_cam_rot=0.0,
            use_sag=False,
            optimize_head=True,
            optimize_camera=True,
            active_arms=["right", "left"],
            estimate_measurement_noise=True,
            apply_joint_offset_limits=False
        )
        
        # We want to build the total Jacobian matrix J_total
        # total_dim = 14 (arms) + 2 (head) + 6 (camera) = 22
        # For each sample, we have 2 arms active. So 2 arms * 6 residuals = 12 residuals per sample.
        # Total rows = 201 * 12 = 2412
        N = len(q_arm_list)
        J_list = []
        
        q_arm_offset = np.zeros(len(opt.arm_idx))
        q_head_offset = np.zeros(len(opt.head_idx))
        xi_mount_cam = np.zeros(6)
        
        for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_list, marker_list):
            for side_idx, arm_side in enumerate(["right", "left"]):
                T_meas = T_meas_pair[side_idx]
                Jb_joint, _, T_ee_to_marker, T_model = opt.evaluate_sample(
                    q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam
                )
                J = opt.build_jacobian(
                    q_arm, q_head, arm_side, q_arm_offset, q_head_offset, xi_mount_cam,
                    Jb_joint, T_ee_to_marker, T_model
                )
                J_list.append(J)
                
        J_total = np.vstack(J_list) # Shape (2412, 22)
        print(f"J_total shape: {J_total.shape}")
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(J_total, full_matrices=False)
        print("\n--- Singular Values of Joint + Camera Jacobian (All 22 columns) ---")
        for i, val in enumerate(S):
            print(f"S[{i:02d}]: {val:.5f}")
            
        # Check condition number
        cond_num = S[0] / S[-1]
        print(f"\nCondition number (all 22 params): {cond_num:.2f}")
        
        # Let's check condition number if we ONLY optimize head (first 16 columns)
        J_head_only = J_total[:, :16]
        S_ho = np.linalg.svd(J_head_only, compute_uv=False)
        print(f"Condition number (Head only, 16 params): {S_ho[0]/S_ho[-1]:.2f}")
        
        # Let's check condition number if we ONLY optimize camera (14 arm + 6 camera = 20 columns)
        # To do this, we slice out the head columns (index 14, 15)
        J_cam_only = np.delete(J_total, [14, 15], axis=1)
        S_co = np.linalg.svd(J_cam_only, compute_uv=False)
        print(f"Condition number (Camera only, 20 params): {S_co[0]/S_co[-1]:.2f}")
        
        # Compute correlation (cosine similarity) between head columns and camera orientation columns
        # Index mapping in J_total:
        # 0..13: Arm offsets
        # 14: Head pan offset
        # 15: Head tilt offset
        # 16, 17, 18: Camera rotation offsets (rx, ry, rz)
        # 19, 20, 21: Camera translation offsets (tx, ty, tz)
        
        def cos_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
        print("\n--- Column Correlation (Cosine Similarity) ---")
        print(f"Head Pan (col 14) vs Camera rx (col 16): {cos_sim(J_total[:, 14], J_total[:, 16]):.4f}")
        print(f"Head Pan (col 14) vs Camera ry (col 17): {cos_sim(J_total[:, 14], J_total[:, 17]):.4f}")
        print(f"Head Pan (col 14) vs Camera rz (col 18): {cos_sim(J_total[:, 14], J_total[:, 18]):.4f}")
        
        print(f"Head Tilt (col 15) vs Camera rx (col 16): {cos_sim(J_total[:, 15], J_total[:, 16]):.4f}")
        print(f"Head Tilt (col 15) vs Camera ry (col 17): {cos_sim(J_total[:, 15], J_total[:, 17]):.4f}")
        print(f"Head Tilt (col 15) vs Camera rz (col 18): {cos_sim(J_total[:, 15], J_total[:, 18]):.4f}")
        
        robot.disconnect()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
