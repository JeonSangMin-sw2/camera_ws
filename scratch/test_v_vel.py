import sys
import numpy as np

c_A = np.array([112.066, -3.663, 405.844])
c_B = np.array([109.622, -7.106, 404.989])

# Assume some reasonable values for a_cand, p_cand
a_cand_cam = np.array([0.0, 1.0, 0.0]) # Y axis
p_cand_cam = np.array([0.0, 0.0, 200.0]) # J4 origin

v = np.cross(a_cand_cam, c_A - p_cand_cam)
delta = np.dot(c_B - c_A, v) / np.linalg.norm(v)**2
opt = -np.degrees(delta)
print(f"Optimal Offset: {opt:.4f} deg")
