import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import rby1_sdk as rby
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config

data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260820_112521.npz')
q_arm_list = data['q_arm']
q_head_list = data['q_head']
T_meas_list = data['marker']

class MockRobot:
    def __init__(self):
        self._real_robot = rby.create_robot("127.0.0.1", "m")
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(26)
        return State()

robot = MockRobot()
model = robot.model()
both_cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

staged_offsets = {
    'right': {'joint3': -0.4933, 'joint5': -5.4195, 'joint6': -2.3771},
    'left': {'joint3': -0.6898, 'joint5': 2.9815, 'joint6': -3.5191}
}

out = []
for l_cam in [0.0, 1.0, 10.0, 100.0, 1000.0, 1e5]:
    try:
        opt = QPCalibrationOptimizer(
            robot=robot,
            arm_idx=both_cfg['arm_idx'],
            ee_links=both_cfg['ee_links'],
            mount_to_cam_nom=both_cfg['mount_to_cam_nom'],
            head_base_to_cam_nom=both_cfg.get('head_base_to_cam_nom'),
            ee_to_marker_nom=both_cfg['ee_to_marker_nom'],
            head_idx=head_cfg['head_idx'],
            eps=1e-6,
            lambda_cam_pos=1.0,
            lambda_cam_rot=l_cam,
            use_sag=False,
            optimize_head=True,
            optimize_camera=True,
            active_arms=['right', 'left'],
            estimate_measurement_noise=True,
            apply_joint_offset_limits=True,
            joint_offsets_to_apply=staged_offsets,
        )
        q_arm_offset, q_head_offset, xi_cam, _, _ = opt.optimize(q_arm_list, q_head_list, T_meas_list)
        h_deg = np.rad2deg(q_head_offset)
        c_roll = np.rad2deg(xi_cam[0])
        s = f"lambda_cam_rot = {l_cam:8.1f} | Head Tilt: {h_deg[1]:+7.4f}° (GT -1.5°) | Cam Roll: {c_roll:+7.4f}° (GT 0.0°)"
        out.append(s)
    except Exception as e:
        out.append(f"Error for lambda {l_cam}: {e}")

with open("/home/rainbow/camera_ws/scratch/tilt_results.txt", "w") as f:
    f.write("\n".join(out) + "\n")

