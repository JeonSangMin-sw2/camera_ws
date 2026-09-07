import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

# 11 Tilt sweep points from user's log
captured_tilt_angles = [-8.0, -6.4, -4.8, -3.2, -1.6, 0.0, 1.6, 3.2, 4.8, 6.4, 8.0]
pts_tilt_cam = np.array([
    [ 87.4,  +20.5, 198.1],
    [ 87.4,  +16.3, 197.5],
    [ 87.5,   +9.3, 196.1],
    [ 87.5,   +2.6, 194.8],
    [ 87.3,   -4.1, 193.0],
    [ 87.4,  -11.0, 191.2],
    [ 87.7,  -17.5, 189.6],
    [ 87.5,  -23.9, 187.2],
    [ 87.3,  -30.5, 184.7],
    [ 87.4,  -36.9, 182.2],
    [ 87.6,  -43.3, 179.7]
]) / 1000.0

# 11 Pan sweep points from user's log
captured_pan_angles = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
pts_pan_cam = np.array([
    [ 48.2,  -10.9, 200.8],
    [ 53.3,  -11.2, 200.0],
    [ 61.9,  -11.0, 198.3],
    [ 70.2,  -11.1, 196.2],
    [ 78.6,  -11.3, 193.9],
    [ 86.9,  -11.3, 191.4],
    [ 95.1,  -11.2, 188.5],
    [103.1,  -11.4, 185.5],
    [111.3,  -11.4, 181.7],
    [119.1,  -11.5, 178.2],
    [126.7,  -11.8, 174.3]
]) / 1000.0

print("pts_tilt_cam shape:", pts_tilt_cam.shape)
print("pts_pan_cam shape:", pts_pan_cam.shape)

# Let's fit a circle to the tilt sweep in camera frame
# In camera coordinates, Tilt rotates about X axis:
# (Y - Y_c)^2 + (Z - Z_c)^2 = R^2
Y = pts_tilt_cam[:, 1]
Z = pts_tilt_cam[:, 2]

# Circle fit in Y-Z plane
# 2*Y*Y_c + 2*Z*Z_c + (R^2 - Y_c^2 - Z_c^2) = Y^2 + Z^2
A = np.column_stack([2*Y, 2*Z, np.ones_like(Y)])
b = Y**2 + Z**2
sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
Y_c, Z_c, C = sol
R = np.sqrt(C + Y_c**2 + Z_c**2)

print(f"Tilt Circle Center in Cam frame: Y_c = {Y_c*1000:.2f} mm, Z_c = {Z_c*1000:.2f} mm, Radius R = {R*1000:.2f} mm")

# In robot kinematics, where is the head tilt axis relative to the camera?
# Nominal mount_to_cam is: [0.047, 0.009, 0.057, -90, 0, -90]
# Camera optical frame: X right, Y down, Z forward.
# In Camera optical frame, the tilt axis is along -X.
# The vector from camera optical center to head tilt joint axis:
# Mount_to_cam translation: [0.047, 0.009, 0.057] (in mount frame).
# Head tilt joint is at link_head_2 origin.
# So in camera frame, where is the head tilt axis?
