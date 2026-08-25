import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import rby1_sdk as rby
from core.robot_motion import build_incremental_motion_plan, AutoCollectionConfig

class MockState:
    def __init__(self, n):
        self.position = np.zeros(n)

class MockRobot:
    def __init__(self):
        self._robot = rby.create_robot('127.0.0.1', 'a')
    def model(self):
        return self._robot.model()
    def get_dynamics(self):
        return self._robot.get_dynamics()
    def get_state(self):
        return MockState(len(self.model().robot_joint_names))

robot = MockRobot()
dyn_model = robot.get_dynamics()
config = AutoCollectionConfig()
config.max_x = 2.0

log_path = "/home/rainbow/camera_ws/scratch/test_out.txt"
try:
    plan = build_incremental_motion_plan(robot, dyn_model, config, active_arms=['right', 'left'], include_head_motion=True)
    with open(log_path, "w") as f:
        f.write(f"SUCCESS: Total plan steps generated: {len(plan)}\n")
        j0_tilt_steps = [s for s in plan if 'head_tilt_offset_deg' in s and s.get('joint_idx') == 0]
        f.write(f"J0 x Head Tilt Cross Grid steps count: {len(j0_tilt_steps)}\n")
        for s in j0_tilt_steps[:8]:
            f.write(f"  - {s['desc']}\n")
except Exception as e:
    import traceback
    with open(log_path, "w") as f:
        f.write(f"ERROR: {e}\n{traceback.format_exc()}\n")
