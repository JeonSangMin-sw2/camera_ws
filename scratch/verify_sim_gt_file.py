import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform, build_incremental_motion_plan

class State:
    def __init__(self):
        self.position = np.zeros(20)

class MockRobot:
    def __init__(self):
        import rby1_sdk as rby
        self._real_robot = rby.create_robot("127.0.0.1", "m")
        self.joint_offsets = {"right": {}, "left": {}}
        self._state = State()
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        return self._state

mock_robot = MockRobot()
marker_st = SimulatedMarkerTransform(mock_robot, {}, "1.2")
mw = UnifiedCalibrationApp(marker_st=marker_st, robot=mock_robot, ui_only=True)
mw.step2_mode_sel.setCurrentText("sim")

mw.auto_ready_done = True
mw.auto_motion_plan = build_incremental_motion_plan(mock_robot, mw.dyn_model, mw.auto_config, ["right", "left"])

for step in mw.auto_motion_plan:
    if "q_arm" in step:
        mock_robot._state.position[mw.model.right_arm_idx[:7]] = step["q_arm"][:7]
        mock_robot._state.position[mw.model.left_arm_idx[:7]] = step["q_arm"][7:]
    if "q_head" in step and len(mw.model.head_idx) >= 2:
        mock_robot._state.position[mw.model.head_idx[:2]] = step["q_head"]
    mw.run_auto_motion_step_blocking()

q_arm_list = np.array(mw.shared_arm_q_list)
q_head_list = np.array(mw.shared_head_q_list)
T_meas_list = np.array(mw.shared_T_list)

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
