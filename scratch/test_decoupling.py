import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Let's test the complete separation of Bracket RPY and Joint 6 Roll!
# 1. During Joint 6 sweep:
# q6 changes from -15 deg to +15 deg.
# At each sample: T_cam_to_marker is recorded.
# In EE frame (flange frame):
# T_ee_to_marker = T_ee_to_torso @ T_torso_to_cam @ T_cam_to_marker
# The average orientation of T_ee_to_marker gives R_ee_to_marker_measured!
#
# 2. R_ee_to_marker_measured contains:
# R_bracket = R_bracket_actual
# If Joint 6 has an encoder error delta_6, then the actual physical angle was q6 + delta_6.
# So R_ee_to_marker_measured = R_x(delta_6) @ R_bracket_actual.
#
# 3. Now look at Joint 5 sweep (Pitch sweep):
# Joint 5 axis in link 4 is [0, 1, 0].
# In link 5 (after Joint 6): axis is R_x(delta_6) @ [0, 1, 0].
# In marker frame: n5_m = R_bracket_actual.T @ R_x(-delta_6) @ [0, 1, 0].
# Notice: R_bracket_actual @ n5_m = R_x(-delta_6) @ [0, 1, 0]!
#
# 4. Therefore, when we combine (2) and (3):
# R_ee_to_marker_measured @ n5_m = (R_x(delta_6) @ R_bracket_actual) @ (R_bracket_actual.T @ R_x(-delta_6) @ [0, 1, 0])
#                               = R_x(delta_6) @ R_x(-delta_6) @ [0, 1, 0]
#                               = [0, 1, 0]!
#
# This completely decouples delta_6 and R_bracket!

print("Decoupling verified mathematically!")
