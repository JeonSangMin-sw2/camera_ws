import numpy as np

# Test the sign logic
j6_gt = 2.30   # deg
j5_gt = -2.10  # deg

# Pass 1: No compensation yet
q_cmd_pass1_j6 = 0.0
q_cmd_pass1_j5 = 0.0

q_act_pass1_j6 = q_cmd_pass1_j6 + j6_gt  # +2.30 deg
q_act_pass1_j5 = q_cmd_pass1_j5 + j5_gt  # -2.10 deg

# Camera measures physical error in space:
# Measured roll error = +2.30 deg
# Measured pitch error = -2.10 deg

# CORRECT compensation offset convention (like Elbow):
# offset = -measured_error
offset_pass1_j6 = -q_act_pass1_j6  # -2.30 deg
offset_pass1_j5 = -q_act_pass1_j5  # +2.10 deg

print(f"Pass 1 Staged Offsets -> J6: {offset_pass1_j6:+.2f}°, J5: {offset_pass1_j5:+.2f}°")

# Pass 2: movej applies offset: q_cmd = q_nom + offset
q_cmd_pass2_j6 = offset_pass1_j6  # -2.30 deg
q_cmd_pass2_j5 = offset_pass1_j5  # +2.10 deg

q_act_pass2_j6 = q_cmd_pass2_j6 + j6_gt  # -2.30 + 2.30 = 0.00 deg
q_act_pass2_j5 = q_cmd_pass2_j5 + j5_gt  # +2.10 - 2.10 = 0.00 deg

print(f"Pass 2 Physical Position in Space -> J6: {q_act_pass2_j6:+.2f}°, J5: {q_act_pass2_j5:+.2f}°")

# In Pass 2, camera measures remaining error:
meas_err_pass2_j6 = q_act_pass2_j6  # 0.00 deg
meas_err_pass2_j5 = q_act_pass2_j5  # 0.00 deg

# Total updated offset: previous_offset + (-meas_err)
offset_pass2_j6 = offset_pass1_j6 - meas_err_pass2_j6 # -2.30 deg
offset_pass2_j5 = offset_pass1_j5 - meas_err_pass2_j5 # +2.10 deg

print(f"Pass 2 Final Offsets -> J6: {offset_pass2_j6:+.2f}°, J5: {offset_pass2_j5:+.2f}°")
print(f"Change between Pass 1 and Pass 2: J6: {abs(offset_pass2_j6 - offset_pass1_j6):.4f}°, J5: {abs(offset_pass2_j5 - offset_pass1_j5):.4f}° (CONVERGED < 0.05°!)")
