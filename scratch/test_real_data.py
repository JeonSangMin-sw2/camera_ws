import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rby1_sdk.dynamics as rd
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.CalibratorBase import BaseCalibrator

# Offline Robot Wrapper
class OfflineRobot:
    def __init__(self, dyn_robot):
        self.dyn_robot = dyn_robot
        self._model = self._create_model_meta()
        
    def model(self):
        return self._model
        
    def get_dynamics(self):
        return self.dyn_robot
        
    def get_state(self):
        class State:
            def __init__(self, dof):
                self.position = np.zeros(dof)
        return State(self.dyn_robot.get_dof())
        
    def _create_model_meta(self):
        class ModelMeta:
            def __init__(self, dyn_robot):
                self.robot_joint_names = dyn_robot.get_joint_names()
                self.right_arm_idx = [self.robot_joint_names.index(f"right_arm_{i}") for i in range(7)]
                self.left_arm_idx = [self.robot_joint_names.index(f"left_arm_{i}") for i in range(7)]
        return ModelMeta(self.dyn_robot)

def load_real_sweep_data(filepath, sweep_joint_idx, arm_idx, ready_pose):
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = [float(x) for x in line.split(',')]
            angle_deg = parts[0]
            
            # Construct q_full (26 elements)
            q_full = np.zeros(26)
            
            # Set right arm joints to ready pose
            for i, val in enumerate(ready_pose):
                q_full[arm_idx[i]] = val
                
            # Set the sweep joint angle
            q_full[arm_idx[sweep_joint_idx]] = ready_pose[sweep_joint_idx] + np.radians(angle_deg)
            
            # T_cam2marker starts at index 10 (16 values)
            T_flat = parts[10:26]
            T_cam_to_marker = np.array(T_flat).reshape(4, 4)
            
            dataset.append((q_full, T_cam_to_marker))
    return dataset

def run_real_calibration():
    # Load URDF dynamics
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    robot = OfflineRobot(dyn_robot)
    
    calibrator = JointCalibrator(marker_st=None, robot=robot)
    calibrator.get_robot_version = lambda: "1.3"
    
    # Configure camera settings
    calibrator.camera_config = {
        "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, 180.0], # nominal marker offset
        "mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0] # nominal mount to camera
    }
    
    arm_idx = robot.model().right_arm_idx
    ready_pose = np.radians([-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0]) # v1.3 ready pose
    initial_joint_pos = list(ready_pose)
    
    calib_dir = "/home/rainbow/camera_ws/core/calibration"
    
    # 1. Wrist Pitch v13 calibration (Sweep A = Axis 5, Sweep B = Axis 3)
    file_5 = os.path.join(calib_dir, "sweep_points_right_joint_A_axis_5.txt")
    file_3 = os.path.join(calib_dir, "sweep_points_right_joint_B_axis_3.txt")
    
    if os.path.exists(file_5) and os.path.exists(file_3):
        print("\n=== RUNNING REAL DATA WRIST_PITCH_V13 CALIBRATION ===")
        dataset_5 = load_real_sweep_data(file_5, 5, arm_idx, ready_pose)
        dataset_3 = load_real_sweep_data(file_3, 3, arm_idx, ready_pose)
        
        res_pitch = calibrator.compute_calibration_results(
            "right", "wrist_pitch_v13", dataset_5, dataset_3, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False,
            log_callback=print
        )
        print(f"Resulting Recommended Offset: {res_pitch['optimal_offset']:.6f}°")
    else:
        print("Wrist Pitch real files not found!")
        
    # 2. Wrist Roll v13 calibration (Sweep A = Axis 6, Sweep B = Axis 5)
    file_6 = os.path.join(calib_dir, "sweep_points_right_joint_A_axis_6.txt")
    file_5_b = os.path.join(calib_dir, "sweep_points_right_joint_B_axis_5.txt")
    
    if os.path.exists(file_6) and os.path.exists(file_5_b):
        print("\n=== RUNNING REAL DATA WRIST_ROLL_V13 CALIBRATION ===")
        dataset_6 = load_real_sweep_data(file_6, 6, arm_idx, ready_pose)
        dataset_5_b = load_real_sweep_data(file_5_b, 5, arm_idx, ready_pose)
        
        res_roll = calibrator.compute_calibration_results(
            "right", "wrist_roll_v13", dataset_6, dataset_5_b, initial_joint_pos,
            current_offset_deg=0.0, use_angle_based_fitting=True, save_debug=False,
            log_callback=print
        )
        print(f"Resulting Recommended Offset: {res_roll['optimal_offset']:.6f}°")
    else:
        print("Wrist Roll real files not found!")

if __name__ == "__main__":
    run_real_calibration()
