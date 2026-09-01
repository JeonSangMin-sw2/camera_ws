import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_ui import UnifiedCalibrationApp, SimulatedMarkerTransform, build_incremental_motion_plan
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

class MockRobot:
    def __init__(self):
        import rby1_sdk as rby
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

mock_robot = MockRobot()
marker_st = SimulatedMarkerTransform(mock_robot, {}, "1.2", include_head_motion=False)
mw = UnifiedCalibrationApp(marker_st=marker_st, robot=mock_robot, ui_only=True)
mw.chk_servo_head.setChecked(False)
mw.include_head_motion = False
mw.step2_mode_sel.setCurrentText("sim")
mw.apply_joint_offset_flag = True

# Stage Step 1 GT offsets
mw.joint_offsets_store["right"]["joint6"] = -2.38
mw.joint_offsets_store["right"]["joint5"] = -5.42
mw.joint_offsets_store["right"]["joint3"] = -0.50
mw.joint_offsets_store["left"]["joint6"] = -3.52
mw.joint_offsets_store["left"]["joint5"] = 2.98
mw.joint_offsets_store["left"]["joint3"] = -0.69

print("--- Running Step 2 Auto Motion (Headless simulation) ---")
mw.auto_ready_done = True
mw.auto_motion_plan = build_incremental_motion_plan(mock_robot, mw.dyn_model, mw.auto_config, ["right", "left"], include_head_motion=False)

while mw.head_move_count < len(mw.auto_motion_plan):
    mw.run_auto_motion_step()
    app.processEvents()

print(f"[COLLECTION] Samples recorded: {len(mw.shared_arm_q_list)}")

q_arm_list = np.array(mw.shared_arm_q_list)
q_head_list = None
T_meas_list = np.array(mw.shared_T_list)

print("\n--- Running Step 2 QP Optimization ---")
mw.log_signal_safe.connect(print)

result_path = os.path.abspath("result/result_step2/verify_fixed_sim_gt.json")
mw.run_optimizer(
    active_arms=["right", "left"],
    optimize_head=False,
    optimize_camera=False,
    q_arm_list=q_arm_list,
    q_head_list=q_head_list,
    T_meas_list=T_meas_list,
    result_path=result_path,
    lambda_cam_pos=1.0,
    lambda_cam_rot=1.0,
    solver_type="QP Solver",
    use_sag=False,
)

print("\n[TEST COMPLETED SUCCESSFULLY]")
