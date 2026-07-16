import numpy as np
import os
import sys

sys.path.insert(0, '/home/rainbow/camera_ws')
from core.calibration_optimizer import QPCalibrationOptimizer, CalibrationOptimizer
import rby1_sdk as rby
from core.calibration_core import get_both_arm_config, get_head_config

def main():
    dataset_path = '/home/rainbow/camera_ws/result/result_real/result_step2/dataset_20260716_153458.npz'
    data = np.load(dataset_path, allow_pickle=True)
    q_arm_list = data['q_arm']
    q_head_list = data['q_head'] if 'q_head' in data else None
    marker_list = data['marker']
    
    print(f"Loaded dataset: {q_arm_list.shape} samples.")
    
    try:
        robot = rby.create_robot("127.0.0.1:50051", ".*")
        cfg = get_both_arm_config(robot.model(), version="1.2")
        head_cfg = get_head_config(robot.model())
        
        opt_cam = QPCalibrationOptimizer(
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
        
        res = opt_cam.optimize(q_arm_list, q_head_list, marker_list)
        q_arm_off, q_head_off, xi_cam, mtc, hbtc = res
        
        print("\n--- Optimization Finished ---")
        print(f"Right arm off (deg): {np.rad2deg(q_arm_off[:7])}")
        print(f"Left arm off (deg) : {np.rad2deg(q_arm_off[7:])}")
        print(f"Head off (deg)     : {np.rad2deg(q_head_off)}")
        print(f"xi_cam             : {xi_cam}")
        print(f"mount_to_cam_new   : {mtc}")
        
    except Exception as e:
        print(f"Failed to run optimizer: {e}")

if __name__ == "__main__":
    main()
