import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

class MockRobot:
    def __init__(self):
        import rby1_sdk as rby
        self._real_robot = rby.create_robot("127.0.0.1", "m")
        self.joint_offsets = {"right": {}, "left": {}}
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(20)
        return State()

mock_robot = MockRobot()
marker_st = SimulatedMarkerTransform(mock_robot, {}, "1.2")
mw = UnifiedCalibrationApp(marker_st=marker_st, robot=mock_robot, ui_only=True)
mw.step2_mode_sel.setCurrentText("sim")

print("--- Step 1: Running Auto Motion Data Collection (sim mode) ---")
mw.auto_ready_done = True
from main_ui import build_incremental_motion_plan
mw.auto_motion_plan = build_incremental_motion_plan(mock_robot, mw.dyn_model, mw.auto_config, ["right", "left"])

for i in range(len(mw.auto_motion_plan)):
    mw.run_auto_motion_step_blocking()

print(f"[COLLECTION] Samples recorded: {len(mw.shared_arm_q_list)}")

q_arm_list = np.array(mw.shared_arm_q_list)
q_head_list = np.array(mw.shared_head_q_list)
T_meas_list = np.array(mw.shared_T_list)

print("\n--- Step 2: Running Hand-Eye Optimization ---")
def log_print(msg):
    if "Result saved" in msg or "RESULT" in msg or "Right arm joint offset" in msg or "Left arm joint offset" in msg or "mount_to_cam_new" in msg or "BASE LINE COMPARISON" in msg or "Diff =" in msg:
        print(msg)

mw.log_msg = log_print

mw.run_optimizer(
    active_arms=["right", "left"],
    optimize_head=True,
    optimize_camera=True,
    q_arm_list=q_arm_list,
    q_head_list=q_head_list,
    T_meas_list=T_meas_list,
    result_path="result/result_step2/verify_sim_gt.json",
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    solver_type="QP Solver",
    use_sag=False,
)

print("\n[VERIFICATION DONE]")
