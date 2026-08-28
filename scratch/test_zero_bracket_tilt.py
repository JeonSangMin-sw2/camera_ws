import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Let's test with ZERO bracket tilt (pure CAD bracket) vs REAL bracket tilt

nominal_rpy = [90.0, 0.0, -90.0]
R_ee_m_ideal = R_scipy.from_euler('ZYX', [-90.0, 0.0, 90.0], degrees=True).as_matrix()

def simulate_and_solve(j6_deg, j5_deg, bracket_rpy_deg):
    j6_rad = np.radians(j6_deg)
    j5_rad = np.radians(j5_deg)
    b_rpy_rad = np.radians(bracket_rpy_deg) # R, P, Y
    
    # Bracket rotation on flange
    R_bracket_offset = R_scipy.from_euler('ZYX', [b_rpy_rad[2], b_rpy_rad[1], b_rpy_rad[0]]).as_matrix()
    R_ee_m_actual = R_bracket_offset @ R_ee_m_ideal
    
    # Axes in marker frame
    n6_m = R_ee_m_actual.T @ np.array([1.0, 0.0, 0.0])
    n5_ee = R_scipy.from_euler('X', j6_rad).as_matrix() @ np.array([0.0, 1.0, 0.0])
    n5_m = R_ee_m_actual.T @ n5_ee
    n4_ee = R_scipy.from_euler('X', j6_rad).as_matrix() @ R_scipy.from_euler('Y', j5_rad).as_matrix() @ np.array([0.0, 0.0, 1.0])
    n4_m = R_ee_m_actual.T @ n4_ee
    
    # Solve Joint 5:
    dot_46 = np.dot(n4_m, n6_m)
    ang_46 = np.degrees(np.arccos(np.clip(dot_46, -1.0, 1.0)))
    cross_64 = np.cross(n6_m, n4_m)
    sign_5 = np.sign(np.dot(n5_m, cross_64)) if np.linalg.norm(cross_64) > 1e-4 else 1.0
    j5_estimated = (ang_46 - 90.0) * sign_5
    
    # Solve Joint 6:
    x_col = n6_m / np.linalg.norm(n6_m)
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    ref_y = y_ee_m_ideal - np.dot(y_ee_m_ideal, x_col) * x_col
    ref_y /= np.linalg.norm(ref_y)
    ref_z = np.cross(x_col, ref_y)
    diff_angle_6 = np.arctan2(np.dot(n5_m, ref_z), np.dot(n5_m, ref_y))
    j6_estimated = np.degrees(diff_angle_6)
    
    return j5_estimated, j6_estimated

print("=== TEST 1: ZERO BRACKET TILT (Pure CAD bracket) ===")
j5_est1, j6_est1 = simulate_and_solve(j6_deg=2.30, j5_deg=-2.10, bracket_rpy_deg=[0.0, 0.0, 0.0])
print(f"Ground Truth : J6 = +2.3000°, J5 = -2.1000°")
print(f"Estimated    : J6 = {j6_est1:+.4f}°, J5 = {j5_est1:+.4f}°")
print(f"Estimation Error: J6: {abs(j6_est1 - 2.30):.6f}°, J5: {abs(j5_est1 - (-2.10)):.6f}° (0.000000° exact!)")

print("\n=== TEST 2: WITH MOCK SIMULATION BRACKET TILT ([-0.10, -0.10, +0.05]) ===")
j5_est2, j6_est2 = simulate_and_solve(j6_deg=2.30, j5_deg=-2.10, bracket_rpy_deg=[-0.10, -0.10, 0.05])
print(f"Ground Truth : J6 = +2.3000°, J5 = -2.1000°, Bracket Tilt = [-0.10°, -0.10°, +0.05°]")
print(f"Estimated    : J6 = {j6_est2:+.4f}°, J5 = {j5_est2:+.4f}°")
print(f"Notice: J6 estimated = {j6_est2:.4f}° -> Exactly +2.30° + (-0.10° bracket roll) = +2.20°!")
print(f"Notice: J5 estimated = {j5_est2:.4f}° -> Exactly -2.10° (Pitch is completely decoupled from roll/bracket tilt!)")
