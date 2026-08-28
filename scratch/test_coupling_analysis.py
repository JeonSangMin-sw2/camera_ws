import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Let's inspect the math of extract_axis_from_rotations and SVD frame construction

# Ideal bracket orientation
nominal_rpy = [90.0, 0.0, -90.0]
R_ee_m_ideal = R_scipy.from_euler('ZYX', [-90.0, 0.0, 90.0], degrees=True).as_matrix()

# Ground Truth Offsets
j6_gt = np.radians(2.30)
j5_gt = np.radians(-2.10)
bracket_rpy_gt = np.radians([-0.10, -0.10, 0.05]) # R, P, Y in degrees

# Actual Bracket Rotation on the Flange:
# In Flange frame, the marker is rotated by R_bracket = R_bracket_offset @ R_ee_m_ideal
R_bracket_offset = R_scipy.from_euler('ZYX', [bracket_rpy_gt[2], bracket_rpy_gt[1], bracket_rpy_gt[0]]).as_matrix()
R_ee_m_actual = R_bracket_offset @ R_ee_m_ideal

# When Joint 6 rotates around X_ee:
# The rotation axis in EE frame is [1, 0, 0]
# In Marker frame, this axis is:
n6_marker = R_ee_m_actual.T @ np.array([1.0, 0.0, 0.0])

# When Joint 5 rotates around Y (with J6 offset j6_gt):
# The rotation axis in EE frame is R_x(j6_gt) @ [0, 1, 0]
n5_ee = R_scipy.from_euler('X', j6_gt).as_matrix() @ np.array([0.0, 1.0, 0.0])
n5_marker = R_ee_m_actual.T @ n5_ee

# When Joint 4 rotates around Z (with J5 offset j5_gt and J6 offset j6_gt):
n4_ee = R_scipy.from_euler('X', j6_gt).as_matrix() @ R_scipy.from_euler('Y', j5_gt).as_matrix() @ np.array([0.0, 0.0, 1.0])
n4_marker = R_ee_m_actual.T @ n4_ee

print("=== EXACT GEOMETRY WITH BRACKET TILT AND JOINT OFFSETS ===")
print("n6_marker:", n6_marker)
print("n5_marker:", n5_marker)
print("n4_marker:", n4_marker)

# 1. Check Angle between J4 and J6 in space:
dot_46 = np.dot(n4_marker, n6_marker)
ang_46 = np.degrees(np.arccos(np.clip(dot_46, -1.0, 1.0)))
print(f"Angle between J4 and J6: {ang_46:.4f}° (Deviation from 90°: {ang_46 - 90.0:.4f}°)")
# Notice: n4_ee . n6_ee = ([0, 0, 1]^T @ R_y(j5)^T @ R_x(j6)^T) @ [1, 0, 0]
# = [0, 0, 1] @ R_y(-j5) @ [1, 0, 0] = sin(j5_gt)!
# So angle between J4 and J6 in space depends ONLY on j5_gt (-2.10°)!
# It is completely independent of bracket tilt!
# arcsin(sin(-2.10°)) = -2.10°!
print(f"Theoretical Pitch Error: {np.degrees(j5_gt):.4f}° -> Measured: {90.0 - ang_46:.4f}°")

# 2. Check Roll calculation:
# In MarkerCalibrator:
x_col = n6_marker / np.linalg.norm(n6_marker)
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])

ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
ref_y /= np.linalg.norm(ref_y)
ref_z = np.cross(x_col, ref_y)

diff_angle_6 = np.arctan2(np.dot(n5_marker, ref_z), np.dot(n5_marker, ref_y))
print(f"\nMeasured Roll Diff Angle: {np.degrees(diff_angle_6):.4f}°")
print(f"Ground Truth J6 Offset   : {np.degrees(j6_gt):.4f}°")
print(f"Ground Truth Bracket Roll: {np.degrees(bracket_rpy_gt[0]):.4f}°")
print(f"Sum (J6 + Bracket Roll)  : {np.degrees(j6_gt + bracket_rpy_gt[0]):.4f}°")
