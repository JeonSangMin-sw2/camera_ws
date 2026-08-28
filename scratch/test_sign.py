import numpy as np

# Let's test the sign convention!
# At Link 4: Joint 4 axis is Z4 = [0, 0, 1] (Sweep B)
# Joint 5 axis is Y5 = [0, 1, 0] (Candidate joint, a_cand)
# When Joint 5 has a positive angle delta (q5 > 0):
# Joint 6 axis X6 rotates around Y by +delta:
# X6 = [cos(delta), 0, -sin(delta)] (Sweep A)
# Then:
# n_A = [cos(delta), 0, -sin(delta)]
# n_B = [0, 0, 1]
# Dot product: n_A . n_B = -sin(delta)
# If delta > 0 (e.g. +5 deg = +0.087 rad):
# dot = -sin(5 deg) = -0.08715
# arccos(dot) = 90 deg - (-5 deg) = 95.0 deg!
# So angle_between_normals = 95.0 deg ( > 90 deg).
# Now let's check cross product:
# cross(n_A, n_B) = [cos(d), 0, -sin(d)] x [0, 0, 1]
#                 = [0 * 1 - (-sin(d))*0, -sin(d)*0 - cos(d)*1, cos(d)*0 - 0*0]
#                 = [0, -cos(d), 0]
# Since d is small, cos(d) > 0, so cross = [0, -1, 0]!
# And a_cand = [0, 1, 0] (Y axis)!
# So dot(cross, a_cand) = [0, -1, 0] . [0, 1, 0] = -1 < 0!
# So sin_sign = -1!
# Then formula calculates:
# optimal_offset_deg = (angle_between_normals - 90.0) * sin_sign
#                    = (95.0 - 90.0) * (-1) = -5.0 deg!
#
# But wait! If the physical angle is +5.0 deg (i.e. robot is at +5 deg when commanded 0),
# the home offset needed to compensate is -5.0 deg.
# But how does perform_joint_calibration update staged_offset?
# In perform_joint_calibration (lines 222-223):
# staged_offset += step_correction
# So staged_offset becomes 0 + (-5.0) = -5.0 deg.
# In the NEXT sweep:
# q_cand[5] = ready_pose[5] + (-5.0 deg) = -5.0 deg.
# So physical angle becomes 0!
#
# BUT WAIT! What happens when camera coordinates transform n_A and n_B?
# In camera coordinates:
# R_torso_to_cam has det = +1.
# Does R_torso_to_cam preserve cross product? Yes (SO(3) preserves cross product).
# BUT what if n_A or n_B has sign flipped in line 915-916?
# Lines 915-916:
# n_A = n_A if np.dot(n_A, a_A_cam) > 0 else -n_A
# n_B = n_B if np.dot(n_B, a_B_cam_nom) > 0 else -n_B
# And what is angle_between_normals?
# Line 912: angle_between_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))

print("Testing with real data values:")
# Real data from right arm:
# In right arm ready pose:
# a_cand_t5 = [0.0713, 0.5783, -0.8127]
# In iteration 1 of right wrist pitch:
# angle_between_normals = 90.8649 deg
# optimal_offset = -0.8649 deg
# staged_offset in iter 2 was -0.8649 deg
# In iteration 2: angle_between_normals = 89.3302 deg -> offset = +0.6698 deg
# staged_offset in iter 3 was -0.3291 deg
# In iteration 3: angle_between_normals = 90.7384 deg -> offset = -0.7384 deg
# staged_offset in iter 4 was -0.8016 deg
# It was OSCILLATING between -0.8° and -0.3°!
