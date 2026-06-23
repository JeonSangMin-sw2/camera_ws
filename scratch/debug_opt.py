import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import qpsolvers

def debug_v13_bracket_calibration_and_optimization(arm_side, x_e_gt, y_e_gt, z_e_gt, d5_gt_deg, d6_gt_deg):
    nominal_rpy = [90.0, 0.0, -90.0]
    R_ee_m_ideal = R_scipy.from_euler('ZYX', [nominal_rpy[2], nominal_rpy[1], nominal_rpy[0]], degrees=True).as_matrix()
    
    # Ground truth values in METERS
    P_ee_gt = np.array([x_e_gt, y_e_gt, z_e_gt]) / 1000.0
    T_ee_to_marker_gt = np.eye(4)
    T_ee_to_marker_gt[:3, :3] = R_ee_m_ideal
    T_ee_to_marker_gt[:3, 3] = P_ee_gt

    d5_gt_rad = np.radians(d5_gt_deg)
    d6_gt_rad = np.radians(d6_gt_deg)

    # Corrected Actual kinematics in METERS
    def mock_compute_fk_actual(q):
        q4, q5, q6 = q[4], q[5], q[6]
        q5_act = q5 + d5_gt_rad
        q6_act = q6 + d6_gt_rad
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5_act).as_matrix()
        R6 = R_scipy.from_euler('X', q6_act).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R4 @ R5 @ R6
        T[:3, 3] = R4 @ R5 @ [0.0, 0.0, 0.3]
        return T

    L_5_ee = 0.3 # meters
    r6_gt = np.sqrt((y_e_gt/1000.0)**2 + (z_e_gt/1000.0)**2)
    p_j6_0 = R_scipy.from_euler('X', d6_gt_rad).as_matrix() @ P_ee_gt + [0.0, 0.0, L_5_ee]
    r5_gt = np.sqrt(p_j6_0[0]**2 + p_j6_0[2]**2)
    p_j5_0 = R_scipy.from_euler('Y', d5_gt_rad).as_matrix() @ p_j6_0
    r4_gt = np.sqrt(p_j5_0[0]**2 + p_j5_0[1]**2)

    T_t5_to_cam = np.eye(4)
    T_t5_to_cam[:3, :3] = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    
    poses_4 = []
    q_fulls_4 = []
    for q4_deg in np.linspace(-10.0, 10.0, 30):
        q = np.zeros(20)
        q[4] = np.radians(q4_deg)
        T_act = mock_compute_fk_actual(q)
        poses_4.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_4.append(q)

    poses_5 = []
    q_fulls_5 = []
    for q5_deg in np.linspace(-10.0, 10.0, 30):
        q = np.zeros(20)
        q[5] = np.radians(q5_deg)
        T_act = mock_compute_fk_actual(q)
        poses_5.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_5.append(q)

    poses_6 = []
    q_fulls_6 = []
    for q6_deg in np.linspace(-20.0, 20.0, 30):
        q = np.zeros(20)
        q[6] = np.radians(q6_deg)
        T_act = mock_compute_fk_actual(q)
        poses_6.append(np.linalg.inv(T_t5_to_cam) @ T_act @ T_ee_to_marker_gt)
        q_fulls_6.append(q)

    # Reconstruct ideal/actual axes
    def extract_axis_from_rotations(poses, ideal_axis):
        if len(poses) < 2: return ideal_axis
        mid_idx = len(poses) // 2
        R_ref = poses[mid_idx][:3, :3]
        axes = []
        for i, T in enumerate(poses):
            if i == mid_idx: continue
            R_rel = R_ref.T @ T[:3, :3] 
            rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > np.radians(1.0):
                axis = rotvec / angle
                if np.dot(axis, ideal_axis) < 0: axis = -axis
                axes.append(axis)
        if len(axes) > 0:
            avg_axis = np.mean(axes, axis=0)
            return avg_axis / np.linalg.norm(avg_axis)
        return ideal_axis

    x_ee_m_ideal = R_ee_m_ideal.T @ np.array([1.0, 0.0, 0.0])
    y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 1.0, 0.0])
    z_ee_m_ideal = R_ee_m_ideal.T @ np.array([0.0, 0.0, 1.0])

    n6_marker_actual = extract_axis_from_rotations(poses_6, x_ee_m_ideal)
    n5_marker_actual = extract_axis_from_rotations(poses_5, y_ee_m_ideal)
    n4_marker_actual = extract_axis_from_rotations(poses_4, z_ee_m_ideal)

    radius_6 = r6_gt # METERS
    radius_5 = r5_gt # METERS
    radius_4 = r4_gt # METERS

    x_nom = 0.095 # meters
    y_nom = 0.0
    z_nom = -0.005 # meters

    # Stage 1: roll_diff
    R_list_6 = []
    R_t5_to_cam = T_t5_to_cam[:3, :3]
    for q_full, T_cam_to_marker in zip(q_fulls_6, poses_6):
        q4, q5, q6 = q_full[4], q_full[5], q_full[6]
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5).as_matrix()
        R6 = R_scipy.from_euler('X', q6).as_matrix()
        T_t5_to_ee = np.eye(4)
        T_t5_to_ee[:3, :3] = R4 @ R5 @ R6
        T_t5_to_ee[:3, 3] = R4 @ R5 @ [0.0, 0.0, 0.3]
        
        R_ee_to_t5 = T_t5_to_ee[:3, :3].T
        R_cam_to_marker = T_cam_to_marker[:3, :3]
        R_ee_to_marker = R_ee_to_t5 @ R_t5_to_cam @ R_cam_to_marker
        R_list_6.append(R_ee_to_marker)

    M = np.mean(R_list_6, axis=0)
    U, S, Vt = np.linalg.svd(M)
    R_ee_m_measured = U @ Vt
    if np.linalg.det(R_ee_m_measured) < 0:
        U[:, 2] *= -1
        R_ee_m_measured = U @ Vt

    R_diff = R_ee_m_measured @ R_ee_m_ideal.T
    yaw_diff, pitch_diff, roll_diff = R_scipy.from_matrix(R_diff).as_euler('ZYX', degrees=True)
    opt_delta_6_deg = (roll_diff + 180.0) % 360.0 - 180.0
    d6_init = np.radians(opt_delta_6_deg)

    # Stage 2: QP optimization
    x_state = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom, y_nom, z_nom], dtype=float)
    x_target = np.array([0.0, 0.0, 0.0, 0.0, d6_init, x_nom, y_nom, z_nom], dtype=float)
    w_reg = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])

    # Tighten bounds on Y (index 6) to nominal +/- 3.0 mm
    x_min = np.array([
        -np.radians(30.0), -np.radians(30.0), 0.0,
        -np.radians(15.0), d6_init - np.radians(15.0),
        x_nom - 0.03, y_nom - 0.003, z_nom - 0.06
    ])
    x_max = np.array([
        np.radians(30.0), np.radians(30.0), 0.0,
        np.radians(15.0), d6_init + np.radians(15.0),
        x_nom + 0.03, y_nom + 0.003, z_nom + 0.06
    ])

    def eval_residuals(x):
        y_off, p_off, r_off, d5_val, d6_val, xe, ye, ze = x
        R_off = R_scipy.from_euler('ZYX', [y_off, p_off, r_off]).as_matrix()
        R_em = R_off @ R_ee_m_ideal
        
        n6_p = R_em.T @ np.array([1.0, 0.0, 0.0])
        n5_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ np.array([0.0, 1.0, 0.0])
        
        r6_p = np.sqrt(ye**2 + ze**2)
        Z_p = ye * np.sin(d6_val) + ze * np.cos(d6_val) + L_5_ee
        r5_p = np.sqrt(xe**2 + Z_p**2)
        
        res = []
        res.extend(n6_marker_actual - n6_p)
        res.extend(n5_marker_actual - n5_p)
        
        if n4_marker_actual is not None:
            n4_p = R_em.T @ R_scipy.from_euler('X', -d6_val).as_matrix() @ R_scipy.from_euler('Y', -d5_val).as_matrix() @ np.array([0.0, 0.0, 1.0])
            res.extend(n4_marker_actual - n4_p)
            
        res.append(radius_6 - r6_p)
        res.append(radius_5 - r5_p)
        if n4_marker_actual is not None:
            Y_p = ye * np.cos(d6_val) - ze * np.sin(d6_val)
            r4_p = np.sqrt((xe * np.cos(d5_val) + Z_p * np.sin(d5_val))**2 + Y_p**2)
            res.append(radius_4 - r4_p)
            
        for idx in range(len(x)):
            res.append(w_reg[idx] * (x[idx] - x_target[idx]))
        return np.array(res, dtype=float)

    max_iter = 100
    eps_converge = 1e-8
    qp_reg = 1e-8

    for iteration in range(max_iter):
        f_vals = eval_residuals(x_state)
        f_norm = np.linalg.norm(f_vals)
        
        # Numeric Jacobian
        eps_jac = 1e-7
        J = np.zeros((len(f_vals), len(x_state)))
        for j in range(len(x_state)):
            x_plus = x_state.copy()
            x_plus[j] += eps_jac
            f_plus = eval_residuals(x_plus)
            x_minus = x_state.copy()
            x_minus[j] -= eps_jac
            f_minus = eval_residuals(x_minus)
            J[:, j] = (f_plus - f_minus) / (2.0 * eps_jac)
            
        H = J.T @ J
        g = J.T @ f_vals
        
        P = H + qp_reg * np.eye(len(x_state))
        q = g
        
        dx_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        lb = np.maximum(-dx_max, x_min - x_state)
        ub = np.minimum(dx_max, x_max - x_state)
        
        dx = qpsolvers.solve_qp(P, q, lb=lb, ub=ub, solver='osqp')
        
        x_state += dx
        if np.linalg.norm(dx) < eps_converge:
            break

    yaw_off_opt, pitch_off_opt, roll_off_opt, d5_opt, d6_opt, xe_opt, ye_opt, ze_opt = x_state
    print(f"\nFinal Results:")
    print(f"ye_opt: {ye_opt*1000.0:.4f} mm (gt: {y_e_gt:.4f})")
    print(f"ze_opt: {ze_opt*1000.0:.4f} mm (gt: {z_e_gt:.4f})")

if __name__ == "__main__":
    debug_v13_bracket_calibration_and_optimization("right", x_e_gt=74.8, y_e_gt=0.0, z_e_gt=-50.1, d5_gt_deg=-3.8, d6_gt_deg=5.4)
