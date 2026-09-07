import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

import rby1_sdk as rby
from core.calibration.CalibratorBase import BaseCalibrator
from main_ui import SimulatedMarkerTransform

class MockRobot:
    def __init__(self):
        self._real_robot = rby.create_robot("127.0.0.1:50051", "m")
        self._real_robot.connect(1)
        self._pos = np.zeros(20)
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = self._pos
        return State()
    def set_position(self, pos):
        self._pos = np.array(pos)

def test_decoupled_solution():
    print("=== [TEST] Full Decoupled Head-Camera Calibration with Mock Offsets ===")
    robot = MockRobot()
    cam_cfg = {"mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]}
    sim_marker = SimulatedMarkerTransform(robot, cam_cfg, robot_version="1.2")
    
    dyn_model = robot.get_dynamics()
    model = robot.model()
    head_idx = list(model.head_idx)
    
    # Dual-arm ready pose (check_calib from ready_poses.yaml)
    import yaml
    with open(os.path.join(BASE_DIR, "config", "ready_poses.yaml")) as f:
        r_poses = yaml.safe_load(f)
    r_arm_q = np.radians(r_poses["v1.2"]["check_calib"]["right_arm"])
    l_arm_q = np.radians(r_poses["v1.2"]["check_calib"]["left_arm"])
    q_ready = np.zeros(len(model.robot_joint_names))
    q_ready[model.right_arm_idx[:7]] = r_arm_q
    q_ready[model.left_arm_idx[:7]] = l_arm_q
    robot.set_position(q_ready)
    
    # 1. Sample both markers at Ready Pose (head = [0, 0])
    m_res_r = sim_marker.get_marker_transform(side="right")
    m_res_l = sim_marker.get_marker_transform(side="left")
    obs_r_0 = np.array(m_res_r[0])[:3, 3] if m_res_r else None
    obs_l_0 = np.array(m_res_l[0])[:3, 3] if m_res_l else None
    
    # 2. Sweep Tilt (Right marker)
    tilt_angles_deg = np.linspace(-10.0, 10.0, 11)
    pts_tilt_cam = []
    for t_deg in tilt_angles_deg:
        q_cur = q_ready.copy()
        q_cur[head_idx[1]] = np.radians(t_deg)
        robot.set_position(q_cur)
        pts_tilt_cam.append(np.array(sim_marker.get_marker_transform(side="right")[0])[:3, 3])
        
    # 3. Sweep Pan (Right marker)
    pan_angles_deg = np.linspace(-15.0, 15.0, 11)
    pts_pan_cam = []
    for p_deg in pan_angles_deg:
        q_cur = q_ready.copy()
        q_cur[head_idx[0]] = np.radians(p_deg)
        robot.set_position(q_cur)
        pts_pan_cam.append(np.array(sim_marker.get_marker_transform(side="right")[0])[:3, 3])
        
    pts_tilt_cam = np.array(pts_tilt_cam)
    pts_pan_cam = np.array(pts_pan_cam)
    
    # 4. Decoupled Solver
    nom_mount = cam_cfg["mount_to_cam"]
    R_nom = R_scipy.from_euler('ZYX', [nom_mount[5], nom_mount[4], nom_mount[3]], degrees=True).as_matrix()
    T_nom = np.eye(4)
    T_nom[:3, :3] = R_nom
    T_nom[:3, 3] = nom_mount[:3]
    
    state = dyn_model.make_state(["link_torso_5", "link_head_2"], model.robot_joint_names)
    q_base = q_ready.copy()
    
    # Initial guess for marker in torso5
    q_eval0 = q_base.copy()
    q_eval0[head_idx[0]] = 0.0
    q_eval0[head_idx[1]] = 0.0
    state.set_q(q_eval0)
    dyn_model.compute_forward_kinematics(state)
    T_t5_cam0 = dyn_model.compute_transformation(state, 0, 1) @ T_nom
    p_m_init = (T_t5_cam0 @ np.append(pts_tilt_cam[len(pts_tilt_cam)//2], 1.0))[:3]
    
    def residual_fn(params):
        dq_tilt = params[0]
        d_rpy = params[1:4]
        P_m = params[4:7]
        
        R_d = R_scipy.from_euler('xyz', d_rpy).as_matrix()
        T_d = np.eye(4)
        T_d[:3, :3] = R_d
        T_mount = T_nom @ T_d
        
        residuals = []
        # Tilt sweep
        for t_deg, p_obs in zip(tilt_angles_deg, pts_tilt_cam):
            q_eval = q_base.copy()
            q_eval[head_idx[0]] = 0.0
            q_eval[head_idx[1]] = np.radians(t_deg) + dq_tilt
            state.set_q(q_eval)
            dyn_model.compute_forward_kinematics(state)
            T_t5_cam = dyn_model.compute_transformation(state, 0, 1) @ T_mount
            p_pred = (np.linalg.inv(T_t5_cam) @ np.append(P_m, 1.0))[:3]
            residuals.extend(p_obs - p_pred)
            
        # Pan sweep
        for p_deg, p_obs in zip(pan_angles_deg, pts_pan_cam):
            q_eval = q_base.copy()
            q_eval[head_idx[0]] = np.radians(p_deg)
            q_eval[head_idx[1]] = dq_tilt
            state.set_q(q_eval)
            dyn_model.compute_forward_kinematics(state)
            T_t5_cam = dyn_model.compute_transformation(state, 0, 1) @ T_mount
            p_pred = (np.linalg.inv(T_t5_cam) @ np.append(P_m, 1.0))[:3]
            residuals.extend(p_obs - p_pred)
            
        return np.array(residuals)

    init_guess = np.array([0.0, 0.0, 0.0, 0.0, p_m_init[0], p_m_init[1], p_m_init[2]])
    bounds = (
        [-np.radians(15), -np.radians(8), -np.radians(8), -np.radians(8), 0.1, -0.4, -0.4],
        [ np.radians(15),  np.radians(8),  np.radians(8),  np.radians(8), 0.7,  0.4,  0.4]
    )

    sol = least_squares(residual_fn, init_guess, bounds=bounds, method="trf")
    assert sol.success, "Optimization failed!"
    
    head_tilt_offset_deg = float(np.degrees(sol.x[0]))
    opt_d_rpy = sol.x[1:4]
    
    R_d_opt = R_scipy.from_euler('xyz', opt_d_rpy).as_matrix()
    T_mount_calib = T_nom @ np.block([[R_d_opt, np.zeros((3,1))], [0,0,0,1]])
    
    from core.calibration_optimizer import rot_to_euler_zyx
    rpy_calib_deg = rot_to_euler_zyx(T_mount_calib[:3, :3]) * 180.0 / np.pi
    
    # Neck base position in link_torso_5:
    q_zero = np.zeros(len(model.robot_joint_names))
    state_neck = dyn_model.make_state(["link_torso_5", "link_head_1"], model.robot_joint_names)
    state_neck.set_q(q_zero)
    dyn_model.compute_forward_kinematics(state_neck)
    p_neck = dyn_model.compute_transformation(state_neck, 0, 1)[:3, 3]
    
    q_eval_0 = q_base.copy()
    q_eval_0[head_idx[0]] = 0.0
    q_eval_0[head_idx[1]] = np.radians(head_tilt_offset_deg)
    state.set_q(q_eval_0)
    dyn_model.compute_forward_kinematics(state)
    T_t5_cam_0 = dyn_model.compute_transformation(state, 0, 1) @ T_mount_calib
    
    P_r_t5 = (T_t5_cam_0 @ np.append(obs_r_0, 1.0))[:3]
    P_l_t5 = (T_t5_cam_0 @ np.append(obs_l_0, 1.0))[:3]
    P_mid_t5 = (P_r_t5 + P_l_t5) / 2.0
    v_mid = P_mid_t5 - p_neck
    head_pan_offset_deg = float(-np.degrees(np.arctan2(v_mid[1], v_mid[0])))
    
    gt_pan = BaseCalibrator.MOCK_GT_OFFSETS["head"]["pan"]
    gt_tilt = BaseCalibrator.MOCK_GT_OFFSETS["head"]["tilt"]
    
    print("\n--- Decoupled Calibration Results ---")
    print(f"Calibrated Camera RPY (deg): {np.round(rpy_calib_deg, 3)} (Nominal: [-90, 0, -90])")
    print(f"Camera Δ RPY (deg)         : {np.round(rpy_calib_deg - np.array([-90.0, 0.0, -90.0]), 3)}")
    print(f"Head Pan  Offset (deg)     : {head_pan_offset_deg:+.4f}° (Ground Truth: {gt_pan:+.4f}°)")
    print(f"Head Tilt Offset (deg)     : {head_tilt_offset_deg:+.4f}° (Ground Truth: {gt_tilt:+.4f}°)")
    rmse_3d = np.sqrt(np.mean(sol.fun**2)) * 1000.0
    print(f"3D Marker Reprojection RMSE: {rmse_3d:.4f} mm")
    
    assert abs(head_tilt_offset_deg - gt_tilt) < 0.2, f"Tilt offset error too large: {head_tilt_offset_deg}"
    assert abs(head_pan_offset_deg - gt_pan) < 0.5, f"Pan offset error too large: {head_pan_offset_deg}"
    assert np.all(np.abs(rpy_calib_deg - np.array([-90.0, 0.0, -90.0])) < 0.25), "Camera mount error too large!"
    print("\n>>> DECOUPLING V2 TEST PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    test_decoupled_solution()
