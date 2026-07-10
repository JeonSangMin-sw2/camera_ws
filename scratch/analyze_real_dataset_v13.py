import os
import sys
import numpy as np
import yaml

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

import rby1_sdk.dynamics as rd
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer

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
                self.head_idx = [self.robot_joint_names.index(f"head_{i}") for i in range(2)] if "head_0" in self.robot_joint_names else []
        return ModelMeta(self.dyn_robot)

def load_offline_robot():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot)

def run_opt(dataset_path):
    print(f"\n=======================================================")
    print(f"Running QP Optimization for dataset: {os.path.basename(dataset_path)}")
    print(f"=======================================================")
    
    # Load dataset
    q_arm, q_head, T_meas = load_npz_dataset(dataset_path)
    
    # Set up offline robot
    robot = load_offline_robot()
    
    # Initialize robot state torso joints with the actual torso angles used during collection
    model = robot.model()
    torso_angles = np.radians([0, 30, -60, 30, 0, 0])
    for idx, val in zip(model.torso_idx if hasattr(model, 'torso_idx') else [], torso_angles):
        robot.get_state().position[idx] = val

    # PASS version="1.3" to get the correct Tf_to_marker nominals!
    cfg = get_both_arm_config(robot.model(), version="1.3")
    head_cfg = get_head_config(robot.model())
        
    active_arms = ["right", "left"]
    ee_links = cfg["ee_links"]
    ee_to_marker_nom = cfg["ee_to_marker_nom"]
        
    print(f"EE to Marker Left: {ee_to_marker_nom['left']}")
    print(f"EE to Marker Right: {ee_to_marker_nom['right']}")

    # Create optimizer with non-symmetric bounds (original)
    optimizer_orig = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_nom,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=False,
        optimize_camera=False,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    
    # Let's restore the original non-symmetric limits for test
    D2R = np.pi / 180.0
    optimizer_orig.joint_offset_lower[1] = 0.0 * D2R # RSR >= 0
    optimizer_orig.joint_offset_upper[1] = 10.0 * D2R
    optimizer_orig.joint_offset_lower[8] = -10.0 * D2R # LSR <= 0
    optimizer_orig.joint_offset_upper[8] = 0.0 * D2R
    
    # Run original optimization
    print("\n--- Running with ORIGINAL limits ---")
    q_arm_offset_o, q_head_offset_o, xi_cam_o, _, _ = optimizer_orig.optimize(
        q_arm,
        q_head,
        T_meas
    )
    
    # Create optimizer with symmetric bounds
    optimizer_sym = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_nom,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=False,
        optimize_camera=False,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    # Modify boundaries to be symmetric
    optimizer_sym.joint_offset_lower[1] = -15.0 * D2R
    optimizer_sym.joint_offset_upper[1] = 15.0 * D2R
    optimizer_sym.joint_offset_lower[8] = -15.0 * D2R
    optimizer_sym.joint_offset_upper[8] = 15.0 * D2R
    
    # Run symmetric optimization
    print("\n--- Running with SYMMETRIC limits ---")
    q_arm_offset_s, q_head_offset_s, xi_cam_s, _, _ = optimizer_sym.optimize(
        q_arm,
        q_head,
        T_meas
    )
    
    joint_names = ["shoulder_yaw (J0)", "shoulder_roll (J1)", "shoulder_pitch (J2)", "elbow_pitch (J3)", "wrist_yaw (J4)", "wrist_pitch (J5)", "wrist_roll (J6)"]
    
    print("\n=== COMPARISON RESULTS ===")
    print("--- Right Arm Offsets ---")
    for name, val_o, val_s in zip(joint_names, np.rad2deg(q_arm_offset_o[:7]), np.rad2deg(q_arm_offset_s[:7])):
        print(f"  {name:20s} | Original: {val_o:+.4f} deg | Symmetric: {val_s:+.4f} deg")
        
    print("--- Left Arm Offsets ---")
    for name, val_o, val_s in zip(joint_names, np.rad2deg(q_arm_offset_o[7:]), np.rad2deg(q_arm_offset_s[7:])):
        print(f"  {name:20s} | Original: {val_o:+.4f} deg | Symmetric: {val_s:+.4f} deg")

if __name__ == "__main__":
    dataset_path = "/home/rainbow/camera_ws/result/dataset_20260709_151033.npz"
    run_opt(dataset_path)
