import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

# Let's test the 3-axis simultaneous solver on synthetic data with known ground truth!
def simulate_and_solve():
    # Ground Truth:
    gt_d5 = np.radians(1.5)    # +1.5 deg Pitch offset
    gt_d6 = np.radians(-0.8)   # -0.8 deg Roll offset
    gt_xe = 0.068              # 68 mm (nom 67 mm)
    gt_ye = 0.001              # 1 mm (nom 0 mm)
    gt_ze = -0.002             # -2 mm (nom 0 mm)
    gt_rpy = [90.5, 1.2, -89.0] # Bracket RPY error vs ideal [90, 0, -90]
    
    R_ee_m_gt = R_scipy.from_euler('ZYX', [gt_rpy[2], gt_rpy[1], gt_rpy[0]], degrees=True).as_matrix()
    R_m_ee_gt = R_ee_m_gt.T
    
    # 1. Measurement of n6 in marker frame (Joint 6 sweep at q6=0, q5=0)
    n6_ee = np.array([1.0, 0.0, 0.0])
    n6_m_meas = R_m_ee_gt @ n6_ee
    
    # 2. Measurement of n5 in marker frame (Joint 5 sweep at q6=0 -> physical angle is gt_d6)
    n5_ee = R_scipy.from_euler('X', gt_d6).as_matrix() @ np.array([0.0, 1.0, 0.0])
    n5_m_meas = R_m_ee_gt @ n5_ee
    
    # 3. Measurement of n4 in marker frame (Joint 4 sweep at q6=0, q5=0 -> physical angles gt_d6, gt_d5)
    n4_ee = R_scipy.from_euler('X', gt_d6).as_matrix() @ R_scipy.from_euler('Y', gt_d5).as_matrix() @ np.array([0.0, 0.0, 1.0])
    n4_m_meas = R_m_ee_gt @ n4_ee
    
    # Radii
    L_nom = 0.125
    Z_shifted = gt_ze - L_nom
    r6_meas = np.sqrt(gt_ye**2 + Z_shifted**2)
    Z_prime = gt_ye * np.sin(gt_d6) + Z_shifted * np.cos(gt_d6)
    Y_prime = gt_ye * np.cos(gt_d6) - Z_shifted * np.sin(gt_d6)
    r5_meas = np.sqrt(gt_xe**2 + Z_prime**2)
    r4_meas = np.sqrt((gt_xe * np.cos(gt_d5) + Z_prime * np.sin(gt_d5))**2 + Y_prime**2)
    
    # Add noise
    noise_ang = 0.001 # rad
    
    # SOLVER:
    # State: [yaw_err, pitch_err, roll_err, d5, d6, xe, ye, ze]
    R_ideal = R_scipy.from_euler('ZYX', [-90.0, 0.0, 90.0], degrees=True).as_matrix()
    
    def residuals(x):
        y_e, p_e, r_e, d5_est, d6_est, xe_est, ye_est, ze_est = x
        R_off = R_scipy.from_euler('ZYX', [y_e, p_e, r_e]).as_matrix()
        R_em = R_off @ R_ideal
        R_me = R_em.T
        
        n6_p = R_me @ np.array([1.0, 0.0, 0.0])
        n5_p = R_me @ R_scipy.from_euler('X', d6_est).as_matrix() @ np.array([0.0, 1.0, 0.0])
        n4_p = R_me @ R_scipy.from_euler('X', d6_est).as_matrix() @ R_scipy.from_euler('Y', d5_est).as_matrix() @ np.array([0.0, 0.0, 1.0])
        
        Zs = ze_est - L_nom
        r6_p = np.sqrt(ye_est**2 + Zs**2)
        Zp = ye_est * np.sin(d6_est) + Zs * np.cos(d6_est)
        Yp = ye_est * np.cos(d6_est) - Zs * np.sin(d6_est)
        r5_p = np.sqrt(xe_est**2 + Zp**2)
        r4_p = np.sqrt((xe_est * np.cos(d5_est) + Zp * np.sin(d5_est))**2 + Yp**2)
        
        res = []
        res.extend(n6_m_meas - n6_p)
        res.extend(n5_m_meas - n5_p)
        res.extend(n4_m_meas - n4_p)
        res.append((r6_meas - r6_p) * 10.0)
        res.append((r5_meas - r5_p) * 10.0)
        res.append((r4_meas - r4_p) * 10.0)
        return np.array(res)

    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.067, 0.0, 0.0]
    opt = least_squares(residuals, x0, method='lm')
    
    y_e, p_e, r_e, d5_est, d6_est, xe_est, ye_est, ze_est = opt.x
    R_em_solved = R_scipy.from_euler('ZYX', [y_e, p_e, r_e]).as_matrix() @ R_ideal
    rpy_solved = R_scipy.from_matrix(R_em_solved).as_euler('ZYX', degrees=True)
    
    print("=== SIMULTANEOUS SOLVER RECOVERY TEST ===")
    print(f"Ground Truth Joint 5 Offset : {np.degrees(gt_d5):.4f}° | Solved: {np.degrees(d5_est):.4f}°")
    print(f"Ground Truth Joint 6 Offset : {np.degrees(gt_d6):.4f}° | Solved: {np.degrees(d6_est):.4f}°")
    print(f"Ground Truth Bracket Pos    : [{gt_xe*1000:.2f}, {gt_ye*1000:.2f}, {gt_ze*1000:.2f}] mm")
    print(f"Solved Bracket Pos          : [{xe_est*1000:.2f}, {ye_est*1000:.2f}, {ze_est*1000:.2f}] mm")
    print(f"Ground Truth Bracket RPY    : {gt_rpy}")
    print(f"Solved Bracket RPY          : [{rpy_solved[2]:.2f}, {rpy_solved[1]:.2f}, {rpy_solved[0]:.2f}]")

simulate_and_solve()
