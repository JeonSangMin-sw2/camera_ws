import numpy as np
import json
import os
import sys

sys.path.append('/home/rainbow/camera_ws')
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_arm_config, get_head_config, get_both_arm_config
import rby1_sdk

robot = rby1_sdk.create_robot("127.0.0.1", "m")
npz = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_082005.npz')

q_arm_list = npz['q']
q_head_list = npz['q_head']
T_meas_list = npz['marker']

cfg = get_both_arm_config(robot.model(), version="1.3")
head_cfg = get_head_config(robot.model())

# Test with 7b963e4 parameters (Y=30mm, J5=+1.87)
ee_to_marker_759 = {
    'right': [0.0611, 0.030, -0.0037, 90.708, -2.698, -89.659],
    'left': [0.0591, 0.030, -0.0041, 89.820, 0.166, -89.645]
}
joint_offsets_759 = {
    'right': {'joint3': -2.204, 'joint5': 1.869, 'joint6': -1.301},
    'left': {'joint3': -1.916, 'joint5': 0.236, 'joint6': 1.556}
}

# Run Stage 1 & 2
cfg_r = get_arm_config(robot.model(), "right", version="1.3")
opt_r = QPCalibrationOptimizer(
    robot=robot, arm_idx=cfg_r["arm_idx"], ee_links={"right": cfg_r["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"], ee_to_marker_nom={"right": ee_to_marker_759["right"]},
    active_arms=["right"], optimize_arm=True, optimize_head=True, optimize_camera=True,
    head_idx=head_cfg["head_idx"], lambda_cam_pos=1.0, lambda_cam_rot=1e6,
    estimate_measurement_noise=True, apply_joint_offset_limits=True, joint_offsets_to_apply=joint_offsets_759,
    camera_pos_bound_m=0.005, camera_rot_bound_rad=np.radians(2.0), eps=1e-7, max_iter=50
)
qr, hr, xir, m2c_r, _ = opt_r.optimize(q_arm_list[:, :7], q_head_list, T_meas_list[:, 0])

cfg_l = get_arm_config(robot.model(), "left", version="1.3")
opt_l = QPCalibrationOptimizer(
    robot=robot, arm_idx=cfg_l["arm_idx"], ee_links={"left": cfg_l["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"], ee_to_marker_nom={"left": ee_to_marker_759["left"]},
    active_arms=["left"], optimize_arm=True, optimize_head=True, optimize_camera=True,
    head_idx=head_cfg["head_idx"], lambda_cam_pos=1.0, lambda_cam_rot=1e6,
    estimate_measurement_noise=True, apply_joint_offset_limits=True, joint_offsets_to_apply=joint_offsets_759,
    camera_pos_bound_m=0.005, camera_rot_bound_rad=np.radians(2.0), eps=1e-7, max_iter=50
)
ql, hl, xil, m2c_l, _ = opt_l.optimize(q_arm_list[:, 7:], q_head_list, T_meas_list[:, 1])

# Stage 3 Dual Arm with anchored head/camera
opt_both = QPCalibrationOptimizer(
    robot=robot, arm_idx=cfg["arm_idx"], ee_links=cfg["ee_links"],
    mount_to_cam_nom=m2c_r, ee_to_marker_nom=ee_to_marker_759,
    active_arms=["right", "left"], optimize_arm=True, optimize_head=False, optimize_camera=False,
    head_idx=head_cfg["head_idx"], lambda_cam_pos=1.0, lambda_cam_rot=1e6,
    estimate_measurement_noise=True, apply_joint_offset_limits=True, joint_offsets_to_apply=joint_offsets_759,
    eps=1e-7, max_iter=50
)
h_avg = 0.5 * (hr + hl)
q_both, _, _, _, _ = opt_both.optimize(q_arm_list, h_avg, T_meas_list)

print("=== REPRODUCED RESULTS WITH 7b963e4 PARAMS ===")
print("Right arm (deg):", np.round(-np.degrees(q_both[:7]), 4))
print("Left arm (deg) :", np.round(-np.degrees(q_both[7:]), 4))
print("Head (deg)     :", np.round(np.degrees(h_avg), 4))
