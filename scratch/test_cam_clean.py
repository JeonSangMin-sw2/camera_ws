import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config
import rby1_sdk as rby

class MockRobot:
    def __init__(self):
        self._real_robot = rby.create_robot("127.0.0.1", "a")
        self.joint_offsets = {"right": {}, "left": {}}
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(20)
        return State()

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260907_193815.npz"
data = np.load(dataset_path, allow_pickle=True)

q_arm_list = data["q_arm"]
q_head_list = data["q_head"]
T_meas_list = data["marker"]

print(f"Loaded {len(q_arm_list)} samples.")

robot = MockRobot()
model = robot.model()
cfg_both = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

# Actual bracket values from user run
ee_to_marker = {
    "right": [0.0, -0.05393608, -0.04602495, 90.14223228, 0.11118854, 180.0],
    "left":  [0.0,  0.05445050, -0.05002385, 90.30354097, 0.09732171, 0.0]
}

corrupted_mount = [0.047, 0.009, 0.057, -87.9089, -0.169, -89.9471]
clean_mount = [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]

for name, mount in [("Corrupted (from user run)", corrupted_mount), ("Clean (CAD nominal)", clean_mount)]:
    print(f"\n========================================================")
    print(f"Testing with: {name}")
    print(f"Mount to Cam: {mount}")
    print(f"========================================================")
    opt = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg_both["arm_idx"],
        ee_links=cfg_both["ee_links"],
        mount_to_cam_nom=mount,
        head_base_to_cam_nom=cfg_both.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker,
        active_arms=["right", "left"],
        optimize_arm=True,
        optimize_head=True,
        optimize_camera=False,
        head_idx=head_cfg["head_idx"],
        use_head_kinematics=True,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1.0,
        use_sag=False,
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply={
            "right": {"joint3": -0.5015, "joint5": -5.4419, "joint6": -2.3778},
            "left": {"joint3": -0.6851, "joint5": 2.9807, "joint6": -3.5203}
        },
        camera_pos_bound_m=0.010,
        camera_rot_bound_rad=3.0 * np.pi / 180.0,
        eps=1e-7,
        max_iter=50,
    )
    res = opt.optimize(q_arm_list, q_head_list, T_meas_list)
    q_arm_offset = res[0]
    q_head_offset = res[1]
    
    r_est = np.degrees(q_arm_offset[:7])
    l_est = np.degrees(q_arm_offset[7:])
    h_est = np.degrees(q_head_offset)
    
    gt_r = np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3])
    gt_l = np.array([-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5])
    gt_h = np.array([0.8, -1.5])
    
    r_err = np.abs(r_est - gt_r)
    l_err = np.abs(l_est - gt_l)
    h_err = np.abs(h_est - gt_h)
    
    print(f"\n[RESULTS for {name}]")
    print(f"Head Pan:  Est={h_est[0]:+.4f}°, GT={gt_h[0]:+.4f}°, Err={h_err[0]:.4f}°")
    print(f"Head Tilt: Est={h_est[1]:+.4f}°, GT={gt_h[1]:+.4f}°, Err={h_err[1]:.4f}°")
    print(f"Right Arm Joint Errors: {[round(x, 4) for x in r_err]}")
    print(f"Left Arm Joint Errors:  {[round(x, 4) for x in l_err]}")
    print(f"Max Right Error: {np.max(r_err):.4f}°")
    print(f"Max Left Error:  {np.max(l_err):.4f}°")
