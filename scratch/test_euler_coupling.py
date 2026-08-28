import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Nominal bracket for v1.3
# nominal_rpy = [Roll=90, Pitch=0, Yaw=-90]
yaw_nom = -90.0
pitch_nom = 0.0
roll_nom = 90.0

R_nom = R_scipy.from_euler('ZYX', [yaw_nom, pitch_nom, roll_nom], degrees=True).as_matrix()
print("R_nom (Flange to Marker):\n", np.round(R_nom, 4))

# Now let's see what happens if the Flange rotates around X_ee by delta (Joint 6 rotation):
# R_flange_rotated = R_nom @ Rot_X(delta) or Rot_X_flange(delta) @ R_nom?
# The Flange frame is the base/reference frame, Marker frame is the child frame.
# T_flange_to_marker = R_ee_m
# When Flange rotates around its own X_ee by angle delta, the marker transform in new flange frame is:
# R_new = Rot_X(-delta) @ R_nom   (or R_nom for marker relative to flange when flange rotated)
# But in marker frame, what axis corresponds to X_ee?
# X_ee in Marker frame is: x_ee_m = R_nom.T @ [1, 0, 0]
x_ee_m = R_nom.T @ np.array([1.0, 0.0, 0.0])
y_ee_m = R_nom.T @ np.array([0.0, 1.0, 0.0])
z_ee_m = R_nom.T @ np.array([0.0, 0.0, 1.0])

print("\nFlange Axes expressed in Marker Frame:")
print("X_ee in Marker (J6 Roll axis) :", np.round(x_ee_m, 4))
print("Y_ee in Marker (J5 Pitch axis):", np.round(y_ee_m, 4))
print("Z_ee in Marker (J4 Yaw axis)  :", np.round(z_ee_m, 4))

# Now let's perturb each Euler angle (Yaw, Pitch, Roll) by +5 degrees and see how R changes in Flange Frame:
# In Flange frame, R = R_z(yaw) * R_y(pitch) * R_x(roll)
# If we change Roll by +5 deg:
R_roll = R_scipy.from_euler('ZYX', [yaw_nom, pitch_nom, roll_nom + 5.0], degrees=True).as_matrix()
# Relative rotation in Flange frame: R_roll @ R_nom.T
rotvec_roll = R_scipy.from_matrix(R_roll @ R_nom.T).as_rotvec()
print("\nPerturbing Roll (+5 deg) produces Flange-frame rotation axis:", np.round(rotvec_roll / np.linalg.norm(rotvec_roll), 4), f"angle={np.degrees(np.linalg.norm(rotvec_roll)):.2f}°")

# If we change Pitch by +5 deg:
R_pitch = R_scipy.from_euler('ZYX', [yaw_nom, pitch_nom + 5.0, roll_nom], degrees=True).as_matrix()
rotvec_pitch = R_scipy.from_matrix(R_pitch @ R_nom.T).as_rotvec()
print("Perturbing Pitch (+5 deg) produces Flange-frame rotation axis:", np.round(rotvec_pitch / np.linalg.norm(rotvec_pitch), 4), f"angle={np.degrees(np.linalg.norm(rotvec_pitch)):.2f}°")

# If we change Yaw by +5 deg:
R_yaw = R_scipy.from_euler('ZYX', [yaw_nom + 5.0, pitch_nom, roll_nom], degrees=True).as_matrix()
rotvec_yaw = R_scipy.from_matrix(R_yaw @ R_nom.T).as_rotvec()
print("Perturbing Yaw (+5 deg) produces Flange-frame rotation axis:", np.round(rotvec_yaw / np.linalg.norm(rotvec_yaw), 4), f"angle={np.degrees(np.linalg.norm(rotvec_yaw)):.2f}°")
