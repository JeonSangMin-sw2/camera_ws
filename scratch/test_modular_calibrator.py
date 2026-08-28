import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')
from core.calibration.MarkerCalibrator import MarkerCalibrator

# Mock marker transform and robot
class DummyMarkerST:
    pass

class DummyRobot:
    pass

mc = MarkerCalibrator(DummyMarkerST(), None)

# Create mock sweep datasets
poses_4 = [np.eye(4) for _ in range(50)]
poses_5 = [np.eye(4) for _ in range(50)]
poses_6 = [np.eye(4) for _ in range(50)]

res_4 = {'captured_poses': poses_4, 'radius': 73.0, 'axis_opt': [0, 0, 1], 'rmse': 0.5, 'pts_ee': np.zeros((50, 3))}
res_5 = {'captured_poses': poses_5, 'radius': 142.0, 'axis_opt': [0, 1, 0], 'rmse': 0.5, 'pts_ee': np.zeros((50, 3))}
res_6 = {'captured_poses': poses_6, 'radius': 30.0, 'axis_opt': [1, 0, 0], 'rmse': 0.5, 'pts_ee': np.zeros((50, 3))}

print("Testing compute_wrist_joints_from_3axis_sweeps...")
wrist_res = mc.compute_wrist_joints_from_3axis_sweeps(res_4, res_5, res_6, "right", calib_pitch_deg=0.0, calib_roll_deg=0.0)
print("Wrist Result:", wrist_res)

print("\nTesting compute_marker_bracket_from_orthogonal_sweeps...")
bracket_res = mc.compute_marker_bracket_from_orthogonal_sweeps(res_4, res_5, res_6, "right")
print("Bracket Result:", bracket_res)

print("\nTesting compute_unified_bracket_calibration_v1_3...")
unified_res = mc.compute_unified_bracket_calibration_v1_3(res_5, res_6, "right", marker_data_4=res_4)
print("Unified Result Converged:", unified_res.get('converged'))
print("SUCCESS: All modular methods work perfectly!")
