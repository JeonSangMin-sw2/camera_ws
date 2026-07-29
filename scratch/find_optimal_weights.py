import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config, load_npz_dataset
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

npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260729_171441.npz"
q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)

gt_r = np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3])
gt_l = np.array([-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5])
gt_h = np.array([0.8, -1.6])

staged_offsets = {
    "right": {"joint3": -0.4955, "joint5": -5.4697, "joint6": -2.3879},
    "left":  {"joint3": -0.6914, "joint5": 2.9833,  "joint6": -3.5370}
}

cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

print(f"Loaded NPZ: q_arm_list shape = {q_arm_list.shape}")

class CustomOptimizer(QPCalibrationOptimizer):
    margin = 0.05
    def get_joint_offset_limits(self):
        q_lower, q_upper = super().get_joint_offset_limits()
        D2R = np.pi / 180.0
        if getattr(self, 'apply_joint_offset_limits', False) and getattr(self, 'joint_offsets_to_apply', None) is not None:
            jo = self.joint_offsets_to_apply
            r_j3 = jo.get("right", {}).get("joint3", 0.0)
            l_j3 = jo.get("left", {}).get("joint3", 0.0)
            r_j6 = jo.get("right", {}).get("joint6", 0.0)
            l_j6 = jo.get("left", {}).get("joint6", 0.0)

            q_lower[3] = min((-r_j3 - self.margin) * D2R, (-r_j3 + self.margin) * D2R)
            q_upper[3] = max((-r_j3 - self.margin) * D2R, (-r_j3 + self.margin) * D2R)

            q_lower[10] = min((-l_j3 - self.margin) * D2R, (-l_j3 + self.margin) * D2R)
            q_upper[10] = max((-l_j3 - self.margin) * D2R, (-l_j3 + self.margin) * D2R)

            q_lower[6] = min((-r_j6 - self.margin) * D2R, (-r_j6 + self.margin) * D2R)
            q_upper[6] = max((-r_j6 - self.margin) * D2R, (-r_j6 + self.margin) * D2R)

            q_lower[13] = min((-l_j6 - self.margin) * D2R, (-l_j6 + self.margin) * D2R)
            q_upper[13] = max((-l_j6 - self.margin) * D2R, (-l_j6 + self.margin) * D2R)

        return q_lower, q_upper

for j36_margin_deg in [0.01, 0.02, 0.03, 0.05]:
    opt = CustomOptimizer(
        robot=mock_robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets,
    )
    opt.margin = j36_margin_deg
    q_arm_offset, q_head_offset, xi_cam, _, _ = opt.optimize(q_arm_list, q_head_list, T_meas_list)

    calc_r = np.rad2deg(q_arm_offset[:7])
    calc_l = np.rad2deg(q_arm_offset[7:])
    calc_h = np.rad2deg(q_head_offset)

    err_r = np.abs(calc_r - gt_r)
    err_l = np.abs(calc_l - gt_l)
    err_h = np.abs(calc_h - gt_h)

    max_e = max(np.max(err_r), np.max(err_l), np.max(err_h))
    sys.stdout.write(f"\n--- Margin: {j36_margin_deg}° ---\n")
    sys.stdout.write(f"Max Error Overall: {max_e:.4f}°\n")
    sys.stdout.write(f"Right Arm Errors: J0={err_r[0]:.4f}°, J1={err_r[1]:.4f}°, J2={err_r[2]:.4f}°, J3={err_r[3]:.4f}°, J4={err_r[4]:.4f}°, J5={err_r[5]:.4f}°, J6={err_r[6]:.4f}°\n")
    sys.stdout.write(f"Left Arm Errors : J0={err_l[0]:.4f}°, J1={err_l[1]:.4f}°, J2={err_l[2]:.4f}°, J3={err_l[3]:.4f}°, J4={err_l[4]:.4f}°, J5={err_l[5]:.4f}°, J6={err_l[6]:.4f}°\n")
    sys.stdout.write(f"Head Errors     : Pan={err_h[0]:.4f}°, Tilt={err_h[1]:.4f}°\n")
    sys.stdout.flush()
