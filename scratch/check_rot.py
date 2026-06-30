import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

R = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
print("R_rob_to_cam (which is R_cam_to_torso) is:")
print(np.round(R, 4))
print("Transpose R_rob_to_cam.T is:")
print(np.round(R.T, 4))
