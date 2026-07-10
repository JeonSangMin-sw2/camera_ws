import numpy as np
import yaml
import sys

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset
from core.calibration_optimizer import make_transform, rot_to_euler_zyx

class OfflineRobot:
    def __init__(self, dyn_robot):
        self.dyn_robot = dyn_robot
        self.robot_joint_names = dyn_robot.get_joint_names()
        self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
        self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
        self.head_idx = [self.robot_joint_names.index(f"head_{i}") for i in range(2)] if "head_0" in self.robot_joint_names else []
        self.torso_idx = [self.robot_joint_names.index(f"torso_{i}") for i in range(6)] if "torso_0" in self.robot_joint_names else []

def run_fk_check():
    # Load URDF dynamics
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    robot = OfflineRobot(dyn_robot)
    
    # Load dataset
    dataset_path = "/home/rainbow/camera_data/result/result_/dataset_20260710_182629.npz"
    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    
    # Load camera configs
    with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
        camera_cfg = yaml.safe_load(f).get("camera", {})
        
    T_mount_to_cam = make_transform(camera_cfg["mount_to_cam"])
    
    ee_to_marker = {
        "right": make_transform(camera_cfg["Tf_to_marker_right"]),
        "left": make_transform(camera_cfg["Tf_to_marker_left"])
    }
    
    # Check first sample
    q_arm = q_arm_list[0]
    q_head = q_head_list[0]
    T_meas = T_meas_list[0] # Shape (2, 4, 4)
    
    # Torso state
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    
    # Prepare full joint positions
    q_full = np.zeros(dyn_robot.get_dof())
    # Set torso
    for idx, val in zip(robot.torso_idx, torso_angles):
        q_full[idx] = val
    # Set head
    for idx, val in zip(robot.head_idx, q_head):
        q_full[idx] = val
    # Set arms
    q_full[robot.right_arm_idx] = q_arm[:7]
    q_full[robot.left_arm_idx] = q_arm[7:]
    
    # Make state
    state = dyn_robot.make_state(
        ["link_head_2", "ee_right", "ee_left", "link_head_0"],
        robot.robot_joint_names
    )
    state.set_q(q_full)
    dyn_robot.compute_forward_kinematics(state)
    
    # Compute base to camera mount link
    T_base_to_mount = dyn_robot.compute_transformation(state, 3, 0) # from base to link_head_2
    T_base_to_cam = T_base_to_mount @ T_mount_to_cam
    
    print("=== FIRST SAMPLE FK ANALYSIS (v1.2) ===")
    for idx, arm in enumerate(["right", "left"]):
        T_base_to_ee = dyn_robot.compute_transformation(state, 3, idx + 1) # from base to ee_right/left
        T_model_cam_to_marker = np.linalg.inv(T_base_to_cam) @ T_base_to_ee @ ee_to_marker[arm]
        
        T_meas_arm = T_meas[idx]
        
        # Translation difference
        t_model = T_model_cam_to_marker[:3, 3]
        t_meas = T_meas_arm[:3, 3]
        t_diff = (t_model - t_meas) * 1000.0 # mm
        
        # Rotation difference
        R_diff = np.linalg.inv(T_model_cam_to_marker[:3, :3]) @ T_meas_arm[:3, :3]
        rpy_diff = np.rad2deg(rot_to_euler_zyx(R_diff))
        
        print(f"\n[{arm.upper()} ARM]:")
        print(f"  Model marker position: {np.round(t_model*1000.0, 2)} mm")
        print(f"  Meas marker position : {np.round(t_meas*1000.0, 2)} mm")
        print(f"  Translation Diff (Model - Meas): X: {t_diff[0]:+.2f}, Y: {t_diff[1]:+.2f}, Z: {t_diff[2]:+.2f} mm")
        print(f"  Rotation Diff (Model to Meas)  : R: {rpy_diff[0]:+.2f}, P: {rpy_diff[1]:+.2f}, Y: {rpy_diff[2]:+.2f} deg")

if __name__ == "__main__":
    run_fk_check()
