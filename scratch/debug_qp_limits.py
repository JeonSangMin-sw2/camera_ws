import numpy as np
import sys
import os
import rby1_sdk as rby

sys.path.append("/home/rainbow/camera_ws")
from core.calibration_optimizer import QPCalibrationOptimizer, prepare_q_full
from core.calibration_core import get_arm_config, get_head_config, load_npz_dataset, load_camera_nominals

def get_both_arm_config(model, version="1.2"):
    camera_nominals = load_camera_nominals(version=version)
    return {
        "arm_idx": np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]]),
        "ee_links": {
            "right": "ee_right",
            "left": "ee_left",
        },
        "mount_to_cam_nom": camera_nominals["mount_to_cam_nom"],
        "head_base_to_cam_nom": camera_nominals["head_base_to_cam_nom"],
        "camera_mount_link": camera_nominals["camera_mount_link"],
        "ee_to_marker_nom": {
            "right": camera_nominals["ee_to_marker_right"],
            "left": camera_nominals["ee_to_marker_left"],
        },
    }

npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260812_190908.npz"
q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)

print("Dataset Loaded:", len(q_arm_list))

# Connect to the simulation server
robot = rby.create_robot("127.0.0.1:50051", "m")
if not robot.connect():
    print("[ERROR] Failed to connect to robot at 127.0.0.1:50051")
    sys.exit(1)

model = robot.model()
head_cfg = get_head_config(model)
camera_nominals = load_camera_nominals()
active_arms = ["right", "left"]
cfg = get_both_arm_config(model)
ee_links = cfg["ee_links"]
ee_to_marker_nom = cfg["ee_to_marker_nom"]

# Stage 1
print("\n--- Running Stage 1 ---")
optimizer_st1 = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=ee_links,
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_nom,
    head_idx=head_cfg["head_idx"],
    eps=1e-6,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    estimate_measurement_noise=True,
)
q_arm_offset, q_head_offset, xi_cam, _, _ = optimizer_st1.optimize(
    q_arm_list, q_head_list, T_meas_list
)
print("Stage 1 Head Offset (deg):", np.round(np.rad2deg(q_head_offset), 3))
print("Stage 1 xi_cam (deg/mm):", np.round(xi_cam, 4))

# Stage 2
print("\n--- Running Stage 2 ---")
optimizer_st2 = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=ee_links,
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_nom,
    head_idx=head_cfg["head_idx"],
    eps=1e-6,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    use_sag=False,
    optimize_head=True,
    optimize_camera=False,
    active_arms=active_arms,
    estimate_measurement_noise=True,
)
q_arm_offset, q_head_offset, _, _, _ = optimizer_st2.optimize(
    q_arm_list, q_head_list, T_meas_list,
    q_arm_offset_init=q_arm_offset,
    q_head_offset_init=q_head_offset,
    xi_mount_cam_init=xi_cam,
)
print("Stage 2 Head Offset (deg):", np.round(np.rad2deg(q_head_offset), 3))

# Stage 3
print("\n--- Running Stage 3 ---")
optimizer_st3 = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=ee_links,
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_nom,
    head_idx=head_cfg["head_idx"],
    eps=1e-7,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    estimate_measurement_noise=True,
)
q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer_st3.optimize(
    q_arm_list, q_head_list, T_meas_list,
    q_arm_offset_init=q_arm_offset,
    q_head_offset_init=q_head_offset,
    xi_mount_cam_init=xi_cam,
)
print("Stage 3 Head Offset (deg):", np.round(np.rad2deg(q_head_offset), 3))
print("Stage 3 xi_cam (deg/mm):", np.round(xi_cam, 4))
