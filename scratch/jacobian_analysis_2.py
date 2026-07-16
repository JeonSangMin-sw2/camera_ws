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

    try:
        robot = rby.create_robot("127.0.0.1:50051", "m")
        if not robot.connect(): return
        cfg = get_both_arm_config(robot.model(), version="1.2")
        head_cfg = get_head_config(robot.model())
        opt = QPCalibrationOptimizer(
            robot=robot, arm_idx=cfg["arm_idx"], ee_links=cfg["ee_links"],
            mount_to_cam_nom=cfg["mount_to_cam_nom"], head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
            ee_to_marker_nom=cfg["ee_to_marker_nom"], head_idx=head_cfg["head_idx"],
            lambda_cam_pos=0.0, lambda_cam_rot=0.0, use_sag=False,
            optimize_head=True, optimize_camera=True, active_arms=["right", "left"],
            estimate_measurement_noise=True, apply_joint_offset_limits=False
        )

        J_list = []
        for q_arm, q_head, T_meas_pair in zip(q_arm_list, q_head_list, marker_list):
            for side_idx, arm_side in enumerate(["right", "left"]):
                Jb_joint, _, T_ee_to_marker, T_model = opt.evaluate_sample(q_arm, q_head, arm_side, np.zeros(14), np.zeros(2), np.zeros(6))
                J = opt.build_jacobian(q_arm, q_head, arm_side, np.zeros(14), np.zeros(2), np.zeros(6), Jb_joint, T_ee_to_marker, T_model)
                J_list.append(J)
        
        J_total = np.vstack(J_list)
        
        # J_total columns: 0-13 (Arms), 14-15 (Head), 16-18 (Cam Rot), 19-21 (Cam Trans)
        J_head_camTrans = np.delete(J_total, [16, 17, 18], axis=1)
        S = np.linalg.svd(J_head_camTrans, compute_uv=False)
        print(f"Condition number (Arms + Head + Cam Trans, 19 params): {S[0]/S[-1]:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if robot: robot.disconnect()

if __name__ == "__main__":
    main()
