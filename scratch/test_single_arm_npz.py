import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform

class MockRobot:
    def model(self):
        import rby1_sdk as rby
        return rby.create_robot("127.0.0.1", "m").model()
    def get_dynamics(self):
        import rby1_sdk as rby
        return rby.create_robot("127.0.0.1", "m").get_dynamics()

mock_robot = MockRobot()
marker_st = SimulatedMarkerTransform(mock_robot, {}, "1.2")
mw = UnifiedCalibrationApp(marker_st=marker_st, robot=mock_robot, ui_only=True)
mw.robot_version = "1.2"

npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260729_163337.npz"
data = np.load(npz_path, allow_pickle=True)

# Truncate to single-arm (7 joints) to simulate single-arm NPZ dataset
q_arm_list_single = data["q_arm_list"][:, :7]
q_head_list = data["q_head_list"]
T_meas_list = data["T_meas_list"]

print(f"Testing Single-Arm NPZ dataset (q_arm_list shape = {q_arm_list_single.shape})")

mw.run_optimizer(
    active_arms=["right", "left"], # UI has 2 active arms, but NPZ only has 1 arm ('right')
    optimize_head=True,
    optimize_camera=True,
    q_arm_list=q_arm_list_single,
    q_head_list=q_head_list,
    T_meas_list=T_meas_list,
    result_path="/home/rainbow/camera_ws/result/result_step2/test_single_arm_output.json",
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    solver_type="QP Solver",
    use_sag=False
)
