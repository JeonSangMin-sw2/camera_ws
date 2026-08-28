import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')

from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_arm_config, get_head_config

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# Load real dataset
data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_065350.npz', allow_pickle=True)
indices = np.linspace(0, len(data['q_arm']) - 1, 20, dtype=int)
q_arm_list = data['q_arm'][indices]
q_head_list = data['q_head'][indices]
T_meas_list = data['marker'][indices]

# Step 1 calibrated values
joint_offsets_step1 = {
    'right': {'joint3': -2.2039956, 'joint5': 1.8687617, 'joint6': -1.3008088},
    'left': {'joint3': -1.9160720, 'joint5': 0.2359542, 'joint6': 1.5559680}
}

class OfflineRobotWrapper:
    def __init__(self):
        import rby1_sdk
        self._real_robot = rby1_sdk.create_robot("127.0.0.1", "m")
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(32)
        return State()

robot = OfflineRobotWrapper()
cfg = get_both_arm_config(robot.model(), version="1.3")
head_cfg = get_head_config(robot.model())

ee_to_marker_nom = {
    "right": [0.061105, 0.030, -0.003718, 90.708, -2.6978, -89.659],
    "left": [0.059119, 0.030, -0.004135, 89.820, 0.1662, -89.644]
}

# Run 3-Stage Optimizer
print("=== RUNNING STAGE 1 (RIGHT ARM) ===")
cfg_r = get_arm_config(robot.model(), "right", version="1.3")
opt_r = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_r["arm_idx"],
    ee_links={"right": cfg_r["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom={"right": ee_to_marker_nom["right"]},
    active_arms=["right"],
    optimize_arm=True,
    optimize_head=True,
    optimize_camera=True,
    head_idx=head_cfg["head_idx"],
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=joint_offsets_step1,
    camera_pos_bound_m=0.005,
    camera_rot_bound_rad=2.0 * D2R,
    eps=1e-7,
    max_iter=15,
)
qr, hr, xir, mount_r, head_r = opt_r.optimize(q_arm_list[:, :7], q_head_list, T_meas_list[:, 0])

print("Stage 1 Right Arm Offsets (deg):", np.round(qr, 4))
print("Stage 1 Head Offsets (deg)     :", np.round(hr, 4))

print("\n=== RUNNING STAGE 2 (LEFT ARM) ===")
cfg_l = get_arm_config(robot.model(), "left", version="1.3")
opt_l = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_l["arm_idx"],
    ee_links={"left": cfg_l["ee_link"]},
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom={"left": ee_to_marker_nom["left"]},
    active_arms=["left"],
    optimize_arm=True,
    optimize_head=True,
    optimize_camera=True,
    head_idx=head_cfg["head_idx"],
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=joint_offsets_step1,
    camera_pos_bound_m=0.005,
    camera_rot_bound_rad=2.0 * D2R,
    eps=1e-7,
    max_iter=15,
)
ql, hl, xil, mount_l, head_l = opt_l.optimize(q_arm_list[:, 7:], q_head_list, T_meas_list[:, 1])

print("Stage 2 Left Arm Offsets (deg):", np.round(ql, 4))
print("Stage 2 Head Offsets (deg)    :", np.round(hl, 4))

print("\n=== RUNNING STAGE 3 (DUAL ARM) ===")
q_head_anchor = 0.5 * (hr + hl)
xi_cam_anchor = 0.5 * (xir + xil)
opt_st3 = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg["arm_idx"],
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_nom,
    active_arms=["right", "left"],
    optimize_arm=True,
    optimize_head=False,
    optimize_camera=False,
    head_idx=head_cfg["head_idx"],
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=joint_offsets_step1,
    camera_pos_bound_m=0.005,
    camera_rot_bound_rad=2.0 * D2R,
    eps=1e-7,
    max_iter=50,
)
q_arm_init = np.concatenate([qr, ql])
q_arm_offset, _, _, mount_new, head_new = opt_st3.optimize(
    q_arm_list, q_head_list, T_meas_list,
    q_arm_offset_init=q_arm_init,
    q_head_offset_init=q_head_anchor,
    xi_mount_cam_init=xi_cam_anchor,
)

print("\n========================================================")
print("              OPTIMIZED STEP 2 OFFSETS                  ")
print("========================================================")
print("Right Arm [J0~J6]:", np.round(q_arm_offset[:7], 4))
print("Left Arm  [J0~J6]:", np.round(q_arm_offset[7:], 4))
print("Head      [Pan, Tilt]:", np.round(q_head_anchor, 4))
print("========================================================")
