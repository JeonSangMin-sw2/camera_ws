import numpy as np
import cv2

R_right_ee = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) # R.T
n5_act = R_right_ee @ np.array([-np.sin(-2.35*np.pi/180), 0, np.cos(-2.35*np.pi/180)])
z_axis = np.array([0, 1, 0])
ref_y = np.array([0, 0, 1])
ref_x = np.array([1, 0, 0])
n5_proj = n5_act - np.dot(n5_act, z_axis)*z_axis
print(np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y))))
