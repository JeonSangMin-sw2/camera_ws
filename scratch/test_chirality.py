import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.mock_robot import PureMockRobot
from core.calibration.JointCalibrator import JointCalibrator

def check_chirality(arm, mode):
    robot = PureMockRobot(arm_side=arm, model_name="v12")
    calib = JointCalibrator(marker_st=None, robot=robot)
    
    # Get nominal axes
    R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()
    
    if mode == "wrist_pitch":
        cand_joint = 5
        sweep_joint_A = 4
        sweep_joint_B = 6
        a_cand_local = np.array([0.0, 1.0, 0.0])
        a_A_local = np.array([0.0, 0.0, 1.0])
        a_B_local = np.array([0.0, 0.0, 1.0])
    else:
        cand_joint = 3
        sweep_joint_A = 2
        sweep_joint_B = 4
        a_cand_local = np.array([0.0, 1.0, 0.0])
        a_A_local = np.array([0.0, 0.0, 1.0])
        a_B_local = np.array([0.0, 0.0, 1.0])
        
    # Torso to camera
    a_cand_cam = R_rob_to_cam.T @ a_cand_local
    a_A_cam = R_rob_to_cam.T @ a_A_local
    a_B_cam_nom = R_rob_to_cam.T @ a_B_local
    
    a_cand_cam /= np.linalg.norm(a_cand_cam)
    a_A_cam /= np.linalg.norm(a_A_cam)
    a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)
    
    a_A_proj = a_A_cam - np.dot(a_A_cam, a_cand_cam) * a_cand_cam
    if np.linalg.norm(a_A_proj) > 1e-6:
        a_A_proj /= np.linalg.norm(a_A_proj)
        
    R_epsilon = R_scipy.from_rotvec(np.radians(1.0) * a_cand_cam).as_matrix()
    a_B_cam_eps = R_epsilon @ a_B_cam_nom
    a_B_proj_eps = a_B_cam_eps - np.dot(a_B_cam_eps, a_cand_cam) * a_cand_cam
    if np.linalg.norm(a_B_proj_eps) > 1e-6:
        a_B_proj_eps /= np.linalg.norm(a_B_proj_eps)
        
    cross_nominal = np.cross(a_A_proj, a_B_proj_eps)
    dot_val = np.dot(cross_nominal, a_cand_cam)
    chirality_sign = np.sign(dot_val)
    
    print(f"[{arm.upper()} - {mode}] dot_val: {dot_val:.6f}, chirality_sign: {chirality_sign:.1f}")

check_chirality("right", "wrist_pitch")
check_chirality("left", "wrist_pitch")
check_chirality("right", "elbow")
check_chirality("left", "elbow")
