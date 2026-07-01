import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.mock_robot import PureMockRobot
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.CalibratorBase import BaseCalibrator

def check_chirality_fk(arm, mode):
    robot = PureMockRobot(arm_side=arm, model_name="v12")
    calib = JointCalibrator(marker_st=None, robot=robot)
    
    # Ready pose
    ready_mode = "elbow" if mode == "elbow" else "wrist_pitch"
    version_key = "v1.2"
    q_ready_arm = calib.get_ready_pose(version_key, "joint", ready_mode, arm)
    
    model = robot.model()
    arm_idx = model.left_arm_idx if arm == "left" else model.right_arm_idx
    q_ready = np.zeros(20)
    q_ready[arm_idx] = q_ready_arm
    
    # Config
    if mode == "wrist_pitch":
        cand_joint = 5
        sweep_joint_A = 4
        sweep_joint_B = 6
    else:
        cand_joint = 3
        sweep_joint_A = 2
        sweep_joint_B = 4
        
    dyn_model = robot.get_dynamics()
    
    parent_cand = f"link_torso_5" if cand_joint == 0 else f"link_{arm}_arm_{cand_joint - 1}"
    parent_A = f"link_torso_5" if sweep_joint_A == 0 else f"link_{arm}_arm_{sweep_joint_A - 1}"
    parent_B = f"link_torso_5" if sweep_joint_B == 0 else f"link_{arm}_arm_{sweep_joint_B - 1}"

    T_t5_to_parent_cand = BaseCalibrator.compute_fk(robot, dyn_model, q_ready, parent_cand, "link_torso_5")
    T_t5_to_parent_A = BaseCalibrator.compute_fk(robot, dyn_model, q_ready, parent_A, "link_torso_5")
    T_t5_to_parent_B = BaseCalibrator.compute_fk(robot, dyn_model, q_ready, parent_B, "link_torso_5")

    R_t5_to_parent_cand = T_t5_to_parent_cand[:3, :3]
    R_t5_to_parent_A = T_t5_to_parent_A[:3, :3]
    R_t5_to_parent_B = T_t5_to_parent_B[:3, :3]

    # Rotate nominal axes from local parent link frames to torso frame
    # Wrist Pitch (cand=5) is Y-axis, sweep A (4) is Z-axis, sweep B (6) is Z-axis.
    # Elbow (cand=3) is Y-axis, sweep A (2) is Z-axis, sweep B (4) is Z-axis.
    a_cand_t5 = R_t5_to_parent_cand @ np.array([0.0, 1.0, 0.0])
    a_A_t5 = R_t5_to_parent_A @ np.array([0.0, 0.0, 1.0])
    a_B_t5 = R_t5_to_parent_B @ np.array([0.0, 0.0, 1.0])

    R_rob_to_cam = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()

    a_cand_cam = R_rob_to_cam.T @ a_cand_t5
    a_A_cam = R_rob_to_cam.T @ a_A_t5
    a_B_cam_nom = R_rob_to_cam.T @ a_B_t5
    
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
    
    print(f"[{arm.upper()} - {mode}] FK dot_val: {dot_val:.6f}, chirality_sign: {chirality_sign:.1f}")

check_chirality_fk("right", "wrist_pitch")
check_chirality_fk("left", "wrist_pitch")
check_chirality_fk("right", "elbow")
check_chirality_fk("left", "elbow")
