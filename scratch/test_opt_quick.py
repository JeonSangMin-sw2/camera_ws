import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')

from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_arm_config, get_head_config

data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_065350.npz', allow_pickle=True)
q_arm_list = data['q_arm'][:3]
q_head_list = data['q_head'][:3]
T_meas_list = data['marker'][:3]

import rby1_sdk
real_robot = rby1_sdk.create_robot("127.0.0.1", "m")

class OfflineRobotWrapper:
    def model(self):
        return real_robot.model()
    def get_dynamics(self):
        return real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(26)
        return State()

robot = OfflineRobotWrapper()
cfg_r = get_arm_config(robot.model(), "right", version="1.3")
head_cfg = get_head_config(robot.model())

ee_to_marker_nom = {
    "right": [0.061105, 0.030, -0.003718, 90.708, -2.6978, -89.659],
}

opt_r = QPCalibrationOptimizer(
    robot=robot,
    arm_idx=cfg_r["arm_idx"],
    ee_links={"right": cfg_r["ee_link"]},
    mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
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
    apply_joint_offset_limits=False,
    max_iter=3,
)

print("Starting optimize...", flush=True)
try:
    qr, hr, xir, _, _ = opt_r.optimize(q_arm_list[:, :7], q_head_list, T_meas_list[:, 0])
    print("Finished!", flush=True)
    print("qr (deg):", np.degrees(qr), flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
