import numpy as np
import os
import sys

sys.path.insert(0, '/home/rainbow/camera_ws')
from core.calibration_optimizer import QPCalibrationOptimizer
import rby1_sdk as rby
from core.calibration_core import get_both_arm_config, get_head_config

def main():
    dataset_path = '/home/rainbow/camera_ws/result/result_real/result_step2/dataset_20260716_153458.npz'
    data = np.load(dataset_path, allow_pickle=True)
    q_arm_list = data['q_arm']
    q_head_list = data['q_head'] if 'q_head' in data else None
    marker_list = data['marker']
    
    try:
        robot = rby.create_robot("127.0.0.1:50051", "m")
        robot.connect()
            
        cfg = get_both_arm_config(robot.model(), version="1.2")
        head_cfg = get_head_config(robot.model())
        
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
                # Since J handles both arms but sets 0 for the inactive arm, we can just use J.
                J_list.append(J)
                
        J_total = np.vstack(J_list)
        
        def cos_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
        print("\n--- Elbow (J3) Correlation Analysis ---")
        # J3 Right is index 2. J3 Left is index 9.
        # Head Pan=14, Head Tilt=15
        # Camera rx=16, ry=17, rz=18, tx=19, ty=20, tz=21
        print("Right Elbow (col 2) vs Camera:")
        for i, name in zip(range(16, 22), ["rx", "ry", "rz", "tx", "ty", "tz"]):
            print(f"  vs Camera {name} (col {i}): {cos_sim(J_total[:, 2], J_total[:, i]):.4f}")
            
        print("Right Elbow (col 2) vs Head:")
        print(f"  vs Head Pan  (col 14): {cos_sim(J_total[:, 2], J_total[:, 14]):.4f}")
        print(f"  vs Head Tilt (col 15): {cos_sim(J_total[:, 2], J_total[:, 15]):.4f}")
        
        print("\nLeft Elbow (col 9) vs Camera:")
        for i, name in zip(range(16, 22), ["rx", "ry", "rz", "tx", "ty", "tz"]):
            print(f"  vs Camera {name} (col {i}): {cos_sim(J_total[:, 9], J_total[:, i]):.4f}")
            
        robot.disconnect()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
