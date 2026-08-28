import numpy as np

# Ground Truth from setting / simulation
gt_j6 = 2.30   # deg
gt_j5 = -2.10  # deg

print("=== SIMULATING PASS 1 AND PASS 2 WITH CORRECT SIGN CONVENTION ===")

# Pass 1: Staged offsets are 0.0
staged_j6_p1 = 0.0
staged_j5_p1 = 0.0

# In Pass 1, robot moves to ready pose with staged offsets (0.0):
# Physical position in space:
q_phys_j6_p1 = 0.0 + gt_j6  # +2.30°
q_phys_j5_p1 = 0.0 + gt_j5  # -2.10°

# Camera sweeps and measures physical deviation in space:
# diff_angle_6 = -2.30° (since actual is shifted by +2.30°)
# pitch_deviation_deg = +2.10° (since actual is shifted by -2.10°)
meas_roll_corr_p1 = -2.30  # deg
meas_pitch_corr_p1 = +2.10 # deg

# compute_unified_bracket_calibration_v1_3 output:
opt_j6_p1 = staged_j6_p1 + meas_roll_corr_p1  # -2.30°
opt_j5_p1 = staged_j5_p1 + meas_pitch_corr_p1 # +2.10°

print(f"\n[PASS 1 RESULT]")
print(f"  * Recommended J6 (Roll) Offset : {opt_j6_p1:+.4f}° (Target compensation: -2.30°)")
print(f"  * Recommended J5 (Pitch) Offset: {opt_j5_p1:+.4f}° (Target compensation: +2.10°)")

# Pass 2: movej moves robot with staged offsets applied:
# q_cmd = q_target + opt_offset
# Physical position in space during Pass 2:
q_phys_j6_p2 = opt_j6_p1 + gt_j6  # -2.30 + 2.30 = 0.00°
q_phys_j5_p2 = opt_j5_p1 + gt_j5  # +2.10 - 2.10 = 0.00°

print(f"\n[PASS 2 PHYSICAL STATE ON ROBOT]")
print(f"  * Remaining Physical Roll Error in Space : {q_phys_j6_p2:+.4f}°")
print(f"  * Remaining Physical Pitch Error in Space: {q_phys_j5_p2:+.4f}°")

# Camera sweeps in Pass 2 and measures remaining physical deviation:
meas_roll_corr_p2 = -q_phys_j6_p2   # 0.00°
meas_pitch_corr_p2 = -q_phys_j5_p2  # 0.00°

# compute_unified_bracket_calibration_v1_3 output in Pass 2:
opt_j6_p2 = opt_j6_p1 + meas_roll_corr_p2  # -2.30 + 0.00 = -2.30°
opt_j5_p2 = opt_j5_p1 + meas_pitch_corr_p2 # +2.10 + 0.00 = +2.10°

print(f"\n[PASS 2 VERIFICATION RESULT]")
print(f"  * Recommended J6 (Roll) Offset : {opt_j6_p2:+.4f}°")
print(f"  * Recommended J5 (Pitch) Offset: {opt_j5_p2:+.4f}°")
print(f"  * Step Change (Pass 1 -> Pass 2): J6: {abs(opt_j6_p2 - opt_j6_p1):.4f}°, J5: {abs(opt_j5_p2 - opt_j5_p1):.4f}°")
print(f"  * Convergence: {'PASSED (<0.05°)' if max(abs(opt_j6_p2 - opt_j6_p1), abs(opt_j5_p2 - opt_j5_p1)) < 0.05 else 'FAILED'}")
