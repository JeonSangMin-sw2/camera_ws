import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

R_right = R_scipy.from_euler('ZYX', [180.0, 0.0, 90.0], degrees=True).as_matrix()
R_left = R_scipy.from_euler('ZYX', [0.0, 0.0, 90.0], degrees=True).as_matrix()

print("R_right.T:")
print(np.round(R_right.T, 2))
print("y_right (R_right.T @ [0,1,0]):", np.round(R_right.T @ [0,1,0], 2))
print("z_right (R_right.T @ [0,0,1]):", np.round(R_right.T @ [0,0,1], 2))
print("x_right (R_right.T @ [1,0,0]):", np.round(R_right.T @ [1,0,0], 2))

print("\nR_left.T:")
print(np.round(R_left.T, 2))
print("y_left (R_left.T @ [0,1,0]):", np.round(R_left.T @ [0,1,0], 2))
print("z_left (R_left.T @ [0,0,1]):", np.round(R_left.T @ [0,0,1], 2))
print("x_left (R_left.T @ [1,0,0]):", np.round(R_left.T @ [1,0,0], 2))

