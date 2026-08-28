import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append('/home/rainbow/camera_ws')

from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator
from main_ui import SimulatedMarkerTransform

# Let's create a simulated robot and simulated marker transform
import rby1_sdk
robot = rby1_sdk.create_robot("127.0.0.1", "m")
dyn_model = robot.get_dynamics()
model = robot.model()

arm_side = "right"
arm_idx = model.right_arm_idx

# Ground truth from BaseCalibrator
mock_gt = BaseCalibrator.MOCK_GT_OFFSETS[arm_side]
print(f"Ground Truth for {arm_side.upper()} ARM:")
print(f"  * J6 Offset: {mock_gt['joint6']:+.4f}°")
print(f"  * J5 Offset: {mock_gt['joint5_v13']:+.4f}°")
print(f"  * Bracket RPY: {mock_gt['bracket_rpy']}")

# Let's test FK at ready pose:
ready_poses = {
    "right": [-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0]
}
q_cmd = np.zeros(32)
q_cmd[arm_idx] = np.radians(ready_poses[arm_side])

# Actual robot joint angles in simulation:
q_act = np.array(q_cmd)
q_act[arm_idx[5]] += np.radians(mock_gt['joint5_v13'])
q_act[arm_idx[6]] += np.radians(mock_gt['joint6'])

# End effector frame in torso
T_t5_to_ee_act = BaseCalibrator.compute_fk(robot, dyn_model, q_act, f"ee_{arm_side}", "link_torso_5")

# Marker in torso frame:
bracket_offset_vec = list(mock_gt['bracket_pos']) + list(mock_gt['bracket_rpy'])
T_bracket_offset = BaseCalibrator.make_transform(bracket_offset_vec)

nominal_template = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES["1.3"][arm_side]
T_nominal = BaseCalibrator.make_transform(nominal_template)

T_ee_to_marker_true = T_nominal @ T_bracket_offset
T_t5_to_marker = T_t5_to_ee_act @ T_ee_to_marker_true

print("\n--- TRUE FLANGE TO MARKER TRANSFORMATION ---")
rpy_true = R_scipy.from_matrix(T_ee_to_marker_true[:3, :3]).as_euler('ZYX', degrees=True)
print(f"True Bracket RPY on Flange: Roll={rpy_true[2]:.4f}°, Pitch={rpy_true[1]:.4f}°, Yaw={rpy_true[0]:.4f}°")

# If FK is evaluated with commanded angle q_cmd (which lacks the joint offsets):
T_t5_to_ee_cmd = BaseCalibrator.compute_fk(robot, dyn_model, q_cmd, f"ee_{arm_side}", "link_torso_5")
T_ee_to_marker_measured = np.linalg.inv(T_t5_to_ee_cmd) @ T_t5_to_marker

rpy_meas = R_scipy.from_matrix(T_ee_to_marker_measured[:3, :3]).as_euler('ZYX', degrees=True)
print(f"Apparent Bracket RPY (with J5, J6 uncalibrated): Roll={rpy_meas[2]:.4f}°, Pitch={rpy_meas[1]:.4f}°, Yaw={rpy_meas[0]:.4f}°")

# Rotation difference between apparent and nominal:
R_diff = T_ee_to_marker_measured[:3, :3] @ T_nominal[:3, :3].T
yaw_d, pitch_d, roll_d = R_scipy.from_matrix(R_diff).as_euler('ZYX', degrees=True)
print(f"\nDiscrepancy: Roll Diff={roll_d:.4f}°, Pitch Diff={pitch_d:.4f}°, Yaw Diff={yaw_d:.4f}°")
print(f"Notice: Roll Diff ({roll_d:.4f}°) = J6 Offset ({mock_gt['joint6']:.2f}°) + Bracket Roll ({mock_gt['bracket_rpy'][0]:.2f}°) = {mock_gt['joint6'] + mock_gt['bracket_rpy'][0]:.4f}°!")
