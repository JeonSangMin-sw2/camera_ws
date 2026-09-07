import os
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

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
        self.joint_offsets = {"right": {}, "left": {}}
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

def test_decoupling_with_sim():
    print("=== [TEST] Decoupled Head & Camera Extrinsic Calibration ===")
    robot = MockRobot()
    cam_cfg = {"mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]}
    
    # Ground truth in BaseCalibrator.MOCK_GT_OFFSETS["head"]:
    # pan: 0.8 deg, tilt: -1.5 deg
    # SimulatedMarkerTransform applies this to q_actual_head!
    sim_marker = SimulatedMarkerTransform(robot, cam_cfg, robot_version="1.2")
    
    dyn_model = robot.get_dynamics()
    model = robot.model()
    head_idx = list(model.head_idx)
    
    # Dual-arm ready pose (check_calib from ready_poses.yaml)
    q_ready = np.zeros(len(model.robot_joint_names))
    r_arm_q = np.radians([-24.67, -39.97, 11.25, -107.95, -123.90, 62.21, 58.40])
    l_arm_q = np.radians([-24.67, 39.98, -11.24, -107.95, 123.90, 62.21, -58.40])
    q_ready[model.right_arm_idx[:7]] = r_arm_q
    q_ready[model.left_arm_idx[:7]] = l_arm_q
    robot.set_position(q_ready)
    
    # 1. Compute P_marker_t5 for right arm (assuming Step 1 arm joint offsets are known/calibrated)
    ee_name = "ee_right"
    q_arm_calib = q_ready.copy()
    gt_arm = BaseCalibrator.MOCK_GT_OFFSETS["right"]
    arm_idx = model.right_arm_idx
    q_arm_calib[arm_idx[0]] += np.radians(gt_arm["joint0"])
    q_arm_calib[arm_idx[1]] += np.radians(gt_arm["joint1"])
    q_arm_calib[arm_idx[2]] += np.radians(gt_arm["joint2"])
    q_arm_calib[arm_idx[3]] += np.radians(gt_arm["joint3"])
    q_arm_calib[arm_idx[4]] += np.radians(gt_arm["joint4"])
    q_arm_calib[arm_idx[5]] += np.radians(gt_arm["joint5_v12"])
    q_arm_calib[arm_idx[6]] += np.radians(gt_arm["joint6"])
    
    T_t5_to_ee = BaseCalibrator.compute_fk(robot, dyn_model, q_arm_calib, ee_name, "link_torso_5")
    tf_vec = cam_cfg.get("Tf_to_marker_right")
    if tf_vec is None:
        tf_vec = BaseCalibrator.NOMINAL_BRACKET_TEMPLATES["1.2"]["right"]
    T_ee_to_marker = BaseCalibrator.make_transform(tf_vec)
    
    # In SimulatedMarkerTransform, bracket offset is also added:
    gt_bracket = BaseCalibrator.MOCK_GT_OFFSETS["right"]
    bracket_offset_vec = list(gt_bracket["bracket_pos"]) + list(gt_bracket["bracket_rpy"])
    T_bracket_offset = BaseCalibrator.make_transform(bracket_offset_vec)
    
    P_marker_t5 = (T_t5_to_ee @ T_bracket_offset @ T_ee_to_marker)[:3, 3]
    print(f"Anchored Marker Position in Torso5: {np.round(P_marker_t5 * 1000, 2)} mm")
    
    # 2. Simulate Head Tilt & Pan Sweeps
    tilt_angles = np.linspace(-8.0, 8.0, 11)
    pts_tilt_cam = []
    
    for t_deg in tilt_angles:
        q_cur = q_ready.copy()
        q_cur[head_idx[0]] = 0.0
        q_cur[head_idx[1]] = np.radians(t_deg)
        robot.set_position(q_cur)
        
        m_res = sim_marker.get_marker_transform(side="right")
        T_c2m = np.array(m_res[0])
        pts_tilt_cam.append(T_c2m[:3, 3])
        
    pan_angles = np.linspace(-10.0, 10.0, 11)
    pts_pan_cam = []
    
    for p_deg in pan_angles:
        q_cur = q_ready.copy()
        q_cur[head_idx[0]] = np.radians(p_deg)
        q_cur[head_idx[1]] = 0.0
        robot.set_position(q_cur)
        
        m_res = sim_marker.get_marker_transform(side="right")
        T_c2m = np.array(m_res[0])
        pts_pan_cam.append(T_c2m[:3, 3])
        
    pts_tilt_cam = np.array(pts_tilt_cam)
    pts_pan_cam = np.array(pts_pan_cam)
    
    print(f"Collected {len(pts_tilt_cam)} tilt points, {len(pts_pan_cam)} pan points.")
    
    # 3. Solve Decoupled Optimization
    nom_mount = cam_cfg["mount_to_cam"]
    R_nom = R_scipy.from_euler('ZYX', [nom_mount[5], nom_mount[4], nom_mount[3]], degrees=True).as_matrix()
    T_nom = np.eye(4)
    T_nom[:3, :3] = R_nom
    T_nom[:3, 3] = nom_mount[:3]
    
    state = dyn_model.make_state(["link_torso_5", "link_head_2"], model.robot_joint_names)
    
    def residual_fn(params):
        dq_pan, dq_tilt = params[0], params[1]
        d_rpy = params[2:5]
        
        R_d = R_scipy.from_euler('xyz', d_rpy).as_matrix()
        T_d = np.eye(4)
        T_d[:3, :3] = R_d
        T_mount = T_nom @ T_d
        
        residuals = []
        
        # Tilt sweep
        for t_deg, p_obs in zip(tilt_angles, pts_tilt_cam):
            q_eval = q_ready.copy()
            q_eval[head_idx[0]] = dq_pan
            q_eval[head_idx[1]] = np.radians(t_deg) + dq_tilt
            state.set_q(q_eval)
            dyn_model.compute_forward_kinematics(state)
            T_t5_head = dyn_model.compute_transformation(state, 0, 1)
            
            T_t5_cam = T_t5_head @ T_mount
            p_pred = (np.linalg.inv(T_t5_cam) @ np.append(P_marker_t5, 1.0))[:3]
            residuals.extend(p_obs - p_pred)
            
        # Pan sweep
        for p_deg, p_obs in zip(pan_angles, pts_pan_cam):
            q_eval = q_ready.copy()
            q_eval[head_idx[0]] = np.radians(p_deg) + dq_pan
            q_eval[head_idx[1]] = dq_tilt
            state.set_q(q_eval)
            dyn_model.compute_forward_kinematics(state)
            T_t5_head = dyn_model.compute_transformation(state, 0, 1)
            
            T_t5_cam = T_t5_head @ T_mount
            p_pred = (np.linalg.inv(T_t5_cam) @ np.append(P_marker_t5, 1.0))[:3]
            residuals.extend(p_obs - p_pred)
            
        return np.array(residuals)

    init_guess = np.zeros(5)
    bounds = (
        [-np.radians(10), -np.radians(10), -np.radians(5), -np.radians(5), -np.radians(5)],
        [ np.radians(10),  np.radians(10),  np.radians(5),  np.radians(5),  np.radians(5)]
    )
    
    sol = least_squares(residual_fn, init_guess, bounds=bounds, method="trf")
    
    print("\n--- Optimization Results ---")
    print(f"Success: {sol.success}, Cost: {sol.cost:.4e}")
    opt_pan_deg = np.degrees(sol.x[0])
    opt_tilt_deg = np.degrees(sol.x[1])
    opt_d_rpy = sol.x[2:5]
    
    gt_pan = BaseCalibrator.MOCK_GT_OFFSETS["head"]["pan"]
    gt_tilt = BaseCalibrator.MOCK_GT_OFFSETS["head"]["tilt"]
    
    print(f"Head Pan Offset  : {opt_pan_deg:+.4f}° (Ground Truth: {gt_pan:+.4f}°, Error: {abs(opt_pan_deg - gt_pan):.4f}°)")
    print(f"Head Tilt Offset : {opt_tilt_deg:+.4f}° (Ground Truth: {gt_tilt:+.4f}°, Error: {abs(opt_tilt_deg - gt_tilt):.4f}°)")
    print(f"Camera Mount RPY Δ: {np.round(opt_d_rpy, 4)}° (Ground Truth: [0, 0, 0]°)")
    
    res_vec = sol.fun
    rmse_mm = np.sqrt(np.mean(res_vec**2)) * 1000.0
    print(f"3D Reprojection RMSE: {rmse_mm:.4f} mm")
    
    assert abs(opt_pan_deg - gt_pan) < 0.05, "Pan offset error too large!"
    assert abs(opt_tilt_deg - gt_tilt) < 0.05, "Tilt offset error too large!"
    assert np.all(np.abs(opt_d_rpy) < 0.05), "Camera delta too large!"
    print("\n>>> ALL DECOUPLING TESTS PASSED WITH FLYING COLORS! <<<")

if __name__ == "__main__":
    test_decoupling_with_sim()
