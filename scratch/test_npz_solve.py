import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config, get_arm_config, load_npz_dataset
import rby1_sdk as rby

class SingleRobotHolder:
    _instance = None
    @classmethod
    def get_robot(cls):
        if cls._instance is None:
            cls._instance = rby.create_robot("127.0.0.1", "m")
        return cls._instance

class MockRobot:
    def __init__(self):
        self._real_robot = SingleRobotHolder.get_robot()
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(20)
        return State()

mock_robot = MockRobot()
model = mock_robot.model()

npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260729_163337.npz"
q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)

gt_r = np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3])
gt_h = np.array([0.8, -1.6])

staged_offsets = {
    "right": {"joint3": -0.4955, "joint5": -5.4697, "joint6": -2.3879}
}

# Single Arm NPZ Solve (Right Arm Only)
q_arm_single = q_arm_list[:, :7]
T_meas_single = T_meas_list[:, 0] if T_meas_list.ndim == 4 else T_meas_list
head_cfg = get_head_config(model)
cfg_single = get_arm_config(model, "right", version="1.2")
ee_links_single = {"right": cfg_single["ee_link"]}
ee_to_marker_single = {"right": cfg_single["ee_to_marker_nom"]}

opt_single_st1 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=cfg_single["arm_idx"],
    ee_links=ee_links_single,
    mount_to_cam_nom=cfg_single["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg_single.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_single,
    head_idx=head_cfg["head_idx"],
    eps=1e-6,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=["right"],
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply={"right": staged_offsets["right"]},
)
q_arm_offset_s, q_head_offset_s, xi_cam_s, _, _ = opt_single_st1.optimize(q_arm_single, q_head_list, T_meas_single)

opt_single_st3 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=cfg_single["arm_idx"],
    ee_links=ee_links_single,
    mount_to_cam_nom=cfg_single["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg_single.get("head_base_to_cam_nom"),
    ee_to_marker_nom=ee_to_marker_single,
    head_idx=head_cfg["head_idx"],
    eps=1e-7,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=["right"],
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply={"right": staged_offsets["right"]},
)
q_arm_offset_s, q_head_offset_s, xi_cam_s, mount_to_cam_new_s, _ = opt_single_st3.optimize(
    q_arm_single, q_head_list, T_meas_single,
    q_arm_offset_init=q_arm_offset_s,
    q_head_offset_init=q_head_offset_s,
    xi_mount_cam_init=xi_cam_s
)

calc_r_s = np.rad2deg(q_arm_offset_s)
calc_h_s = np.rad2deg(q_head_offset_s)

err_r_s = np.abs(calc_r_s - gt_r)
err_h_s = np.abs(calc_h_s - gt_h)

print("\n==========================================================================")
print(" SINGLE-ARM NPZ DATASET OPTIMIZATION VERIFICATION")
print("==========================================================================")
print(" [RIGHT ARM]")
for i in range(7):
    status = "[< 0.1° PASS]" if err_r_s[i] < 0.1 else "[FAIL]"
    print(f"   J{i}: GT = {gt_r[i]:+6.2f}° | Calc = {calc_r_s[i]:+8.4f}° | Error = {err_r_s[i]:6.4f}° {status}")

print("\n [HEAD]")
print(f"   Pan : GT = {gt_h[0]:+6.2f}° | Calc = {calc_h_s[0]:+8.4f}° | Error = {err_h_s[0]:6.4f}° {'[< 0.1° PASS]' if err_h_s[0] < 0.1 else '[FAIL]'}")
print(f"   Tilt: GT = {gt_h[1]:+6.2f}° | Calc = {calc_h_s[1]:+8.4f}° | Error = {err_h_s[1]:6.4f}° {'[< 0.1° PASS]' if err_h_s[1] < 0.1 else '[FAIL]'}")
print("==========================================================================\n")
print(f"Max Right Arm Error: {np.max(err_r_s):.4f}°")
print(f"Max Head Error     : {np.max(err_h_s):.4f}°")
