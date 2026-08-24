import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rby1_sdk as rby
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config
from core.calibration.CalibratorBase import BaseCalibrator

# Load dataset
dataset_path = '/home/rainbow/camera_ws/result/result_step2/dataset_20260820_171236.npz'
data = np.load(dataset_path)
q_arm_list = data['q_arm']
q_head_list = data['q_head']
T_meas_list = data['marker']

robot = rby.create_robot('127.0.0.1:50051', 'm')
robot.connect(1)
model = robot.model()
both_cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

staged_offsets = {
    'right': {'joint3': -0.4937, 'joint5': -5.4194, 'joint6': -2.3251},
    'left': {'joint3': -0.6893, 'joint5': 2.9815, 'joint6': -3.4657}
}

mock_gt = BaseCalibrator.MOCK_GT_OFFSETS
is_v13 = False

def run_3stage_eval(lambda_cam_rot_val, label):
    print("\n" + "="*70)
    print(f"  {label} (lambda_cam_rot = {lambda_cam_rot_val})")
    print("="*70)
    
    # Stage 1
    opt1 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=both_cfg['arm_idx'],
        ee_links=both_cfg['ee_links'],
        mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
        head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
        ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
        head_idx=head_cfg['head_idx'],
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=lambda_cam_rot_val,
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=['right', 'left'],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets,
    )
    q_arm_1, q_head_1, xi_cam_1, _, _ = opt1.optimize(q_arm_list, q_head_list, T_meas_list)
    
    # Stage 2
    opt2 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=both_cfg['arm_idx'],
        ee_links=both_cfg['ee_links'],
        mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
        head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
        ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
        head_idx=head_cfg['head_idx'],
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=lambda_cam_rot_val,
        use_sag=False,
        optimize_head=True,
        optimize_camera=False,
        active_arms=['right', 'left'],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets,
    )
    q_arm_2, q_head_2, _, _, _ = opt2.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_1,
        q_head_offset_init=q_head_1,
        xi_mount_cam_init=xi_cam_1,
    )
    
    # Stage 3
    opt3 = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=both_cfg['arm_idx'],
        ee_links=both_cfg['ee_links'],
        mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
        head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
        ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
        head_idx=head_cfg['head_idx'],
        eps=1e-7,
        lambda_cam_pos=1.0,
        lambda_cam_rot=lambda_cam_rot_val,
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=['right', 'left'],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets,
    )
    q_arm_3, q_head_3, xi_cam_3, mount_new, _ = opt3.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_2,
        q_head_offset_init=q_head_2,
        xi_mount_cam_init=xi_cam_1,
    )
    
    r_calc = np.rad2deg(q_arm_3[:7])
    l_calc = np.rad2deg(q_arm_3[7:])
    h_calc = np.rad2deg(q_head_3)
    
    r_gt = [
        mock_gt["right"]["joint0"], mock_gt["right"]["joint1"], mock_gt["right"]["joint2"],
        mock_gt["right"]["joint3"], mock_gt["right"]["joint4"], mock_gt["right"]["joint5_v12"],
        mock_gt["right"]["joint6"]
    ]
    l_gt = [
        mock_gt["left"]["joint0"], mock_gt["left"]["joint1"], mock_gt["left"]["joint2"],
        mock_gt["left"]["joint3"], mock_gt["left"]["joint4"], mock_gt["left"]["joint5_v12"],
        mock_gt["left"]["joint6"]
    ]
    h_gt = [mock_gt["head"]["pan"], mock_gt["head"]["tilt"]]
    
    print("\n [RIGHT ARM JOINTS]")
    for i in range(7):
        diff = abs(r_calc[i] - r_gt[i])
        print(f"   J{i}: Calc = {r_calc[i]:+8.4f}° | GT = {r_gt[i]:+8.4f}° | Error = {diff:6.4f}°")
        
    print("\n [LEFT ARM JOINTS]")
    for i in range(7):
        diff = abs(l_calc[i] - l_gt[i])
        print(f"   J{i}: Calc = {l_calc[i]:+8.4f}° | GT = {l_gt[i]:+8.4f}° | Error = {diff:6.4f}°")
        
    print("\n [HEAD JOINTS]")
    print(f"   Pan:  Calc = {h_calc[0]:+8.4f}° | GT = {h_gt[0]:+8.4f}° | Error = {abs(h_calc[0] - h_gt[0]):6.4f}°")
    print(f"   Tilt: Calc = {h_calc[1]:+8.4f}° | GT = {h_gt[1]:+8.4f}° | Error = {abs(h_calc[1] - h_gt[1]):6.4f}°")
    
    c_rot_deg = np.rad2deg(xi_cam_3[:3])
    c_pos_mm = xi_cam_3[3:] * 1000.0
    print("\n [CAMERA EXTRINSICS]")
    print(f"   Rot Perturbation RPY (deg): [{c_rot_deg[0]:+.4f}, {c_rot_deg[1]:+.4f}, {c_rot_deg[2]:+.4f}]")
    print(f"   Pos Perturbation XYZ (mm) : [{c_pos_mm[0]:+.3f}, {c_pos_mm[1]:+.3f}, {c_pos_mm[2]:+.3f}]")
    print(f"   mount_to_cam_new           : {mount_new}")

run_3stage_eval(1.0, "BASELINE RUN (Weak Camera Regularization: lambda_cam_rot = 1.0)")
run_3stage_eval(1e6, "LOCKED CAMERA ROTATION RUN (lambda_cam_rot = 1e6)")
