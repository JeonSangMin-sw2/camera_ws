import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config

# Load recorded dataset
data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260820_171236.npz')
q_arm_list = data['q_arm']
q_head_list = data['q_head']
T_meas_list = data['marker']

class State:
    def __init__(self):
        self.position = np.zeros(20)

class MockRobot:
    def __init__(self):
        import rby1_sdk as rby
        self._real_robot = rby.create_robot("127.0.0.1", "m")
        self._state = State()
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        return self._state

mock_robot = MockRobot()
model = mock_robot.model()
both_cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

staged_offsets = {
    'right': {'joint3': -0.4937, 'joint5': -5.4194, 'joint6': -2.3251},
    'left': {'joint3': -0.6893, 'joint5': 2.9815, 'joint6': -3.4657}
}

print("=== TEST 1: lambda_cam_rot = 1.0 (Default) ===")
opt1 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=both_cfg['arm_idx'],
    ee_links=both_cfg['ee_links'],
    mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
    head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
    ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
    head_idx=head_cfg['head_idx'],
    eps=1e-7,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=['right', 'left'],
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=staged_offsets,
)
q_arm_1, q_head_1, xi_cam_1, _, _ = opt1.optimize(q_arm_list, q_head_list, T_meas_list)
h1 = np.rad2deg(q_head_1)
c1 = np.rad2deg(xi_cam_1[:3])
print(f"Head Pan: {h1[0]:+.4f}° (GT: +0.8000°), Head Tilt: {h1[1]:+.4f}° (GT: -1.5000°)")
print(f"Cam Rot RPY: {c1} deg")

print("\n=== TEST 2: Lock Camera Rotation via lambda_cam_rot = 1e6 ===")
opt2 = QPCalibrationOptimizer(
    robot=mock_robot,
    arm_idx=both_cfg['arm_idx'],
    ee_links=both_cfg['ee_links'],
    mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
    head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
    ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
    head_idx=head_cfg['head_idx'],
    eps=1e-7,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1e6, # Strong penalty locking camera rotation to nominal CAD
    use_sag=False,
    optimize_head=True,
    optimize_camera=True,
    active_arms=['right', 'left'],
    estimate_measurement_noise=True,
    apply_joint_offset_limits=True,
    joint_offsets_to_apply=staged_offsets,
)
q_arm_2, q_head_2, xi_cam_2, _, _ = opt2.optimize(q_arm_list, q_head_list, T_meas_list)
h2 = np.rad2deg(q_head_2)
c2 = np.rad2deg(xi_cam_2[:3])
print(f"Head Pan: {h2[0]:+.4f}° (GT: +0.8000°), Head Tilt: {h2[1]:+.4f}° (GT: -1.5000°)")
print(f"Cam Rot RPY: {c2} deg")
print(f"Cam Pos XYZ (mm): {xi_cam_2[3:] * 1000.0} mm")
