import numpy as np
from scipy.spatial.transform import Rotation as R

def make_transform(vec):
    pos = vec[:3]
    rpy = vec[3:6]
    rot = R.from_euler('ZYX', [rpy[2], rpy[1], rpy[0]], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

# Nominal right arm v1.2
nom_val = [0.0, -0.054, -0.048, 90.0, 0.0, 180.0]
T_nom = make_transform(nom_val)

# GT right arm bracket offset
gt_pos = [0.0005, 0.0, 0.002]
gt_rpy = [-0.1, -0.1, 0.05]
T_bracket = make_transform(list(gt_pos) + list(gt_rpy))

# In simulation: T_ee_actual_to_marker = T_bracket @ T_nom
T_cal = T_bracket @ T_nom

# Let's extract the calculated bracket offset: T_bracket_calc = T_cal @ inv(T_nom)
T_bracket_calc = T_cal @ np.linalg.inv(T_nom)
calc_pos_offset = T_bracket_calc[:3, 3]
calc_rot_offset = R.from_matrix(T_bracket_calc[:3, :3]).as_euler('ZYX', degrees=True)[::-1]

# Print differences
print("Right Arm:")
print("GT Bracket Pos:", gt_pos)
print("Calc Bracket Pos Offset:", calc_pos_offset)
print("GT Bracket RPY:", gt_rpy)
print("Calc Bracket RPY Offset:", calc_rot_offset)
print("Nominal RPY:", nom_val[3:6])
print("Calibrated RPY:", calc_rpy)
print("Difference (Cal - Nom):", calc_rpy - nom_val[3:6])
print("GT Bracket RPY:", gt_rpy)

# Nominal left arm v1.2
nom_val_l = [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
T_nom_l = make_transform(nom_val_l)

# GT left arm bracket offset
gt_pos_l = [0.001, 0.0005, -0.002]
gt_rpy_l = [0.1, 0.1, 0.0]
T_bracket_l = make_transform(list(gt_pos_l) + list(gt_rpy_l))

# In simulation: T_ee_actual_to_marker = T_bracket @ T_nom
T_cal_l = T_bracket_l @ T_nom_l

# Let's extract position and euler ZYX from T_cal
calc_pos_l = T_cal_l[:3, 3]
calc_rpy_l = R.from_matrix(T_cal_l[:3, :3]).as_euler('ZYX', degrees=True)[::-1] # ZYX to XYZ(RPY)

print("\nLeft Arm:")
print("Nominal RPY:", nom_val_l[3:6])
print("Calibrated RPY:", calc_rpy_l)
print("Difference (Cal - Nom):", calc_rpy_l - nom_val_l[3:6])
print("GT Bracket RPY:", gt_rpy_l)
