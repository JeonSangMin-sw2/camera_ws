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
                self.torso_idx = [self.robot_joint_names.index(f"torso_{i}") for i in range(6)] if "torso_0" in self.robot_joint_names else []
        return ModelMeta(self.dyn_robot)

def load_offline_robot():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
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
    for idx, val in zip(model.torso_idx, torso_angles):
        robot.get_state().position[idx] = val

    with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
        camera_cfg = yaml.safe_load(f).get("camera", {})
    
    ee_to_marker_orig = {
        "left": camera_cfg["Tf_to_marker_left"],
        "right": camera_cfg["Tf_to_marker_right"]
    }
    
    cfg = get_both_arm_config(robot.model(), version="1.2")
    head_cfg = get_head_config(robot.model())
    active_arms = ["right", "left"]
    ee_links = cfg["ee_links"]

    # 1. Run with Original Nominals (Camera Fixed)
    optimizer_orig = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=False,
        optimize_camera=False,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    print("\n--- RUN 1: ORIGINAL NOMINALS (CAMERA FIXED) ---")
    q_arm_offset_o, q_head_offset_o, xi_cam_o, _, _ = optimizer_orig.optimize(q_arm, q_head, T_meas)

    # 2. Run with Original Nominals (Camera Optimized, high regularization lambda=1.0)
    optimizer_cam_reg = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=False,
        optimize_camera=True,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1.0,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    print("\n--- RUN 2: CAMERA OPTIMIZED (LAMBDA = 1.0) ---")
    q_arm_offset_cr, q_head_offset_cr, xi_cam_cr, _, _ = optimizer_cam_reg.optimize(q_arm, q_head, T_meas)
    
    # 3. Run with Original Nominals (Camera Optimized, zero regularization lambda=0.0)
    optimizer_cam_free = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links=ee_links,
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=ee_to_marker_orig,
        head_idx=head_cfg["head_idx"],
        use_sag=False,
        optimize_head=False,
        optimize_camera=True,
        lambda_cam_pos=0.0,
        lambda_cam_rot=0.0,
        active_arms=active_arms,
        estimate_measurement_noise=True,
    )
    print("\n--- RUN 3: CAMERA OPTIMIZED (LAMBDA = 0.0 - FREE) ---")
    q_arm_offset_cf, q_head_offset_cf, xi_cam_cf, _, _ = optimizer_cam_free.optimize(q_arm, q_head, T_meas)
    
    joint_names = ["shoulder_yaw (J0)", "shoulder_roll (J1)", "shoulder_pitch (J2)", "elbow_pitch (J3)", "wrist_yaw (J4)", "wrist_pitch (J5)", "wrist_roll (J6)"]
    
    print("\n=== COMPARISON RESULTS ===")
    print("--- Right Arm Offsets ---")
    for name, val_o, val_cr, val_cf in zip(joint_names, np.rad2deg(q_arm_offset_o[:7]), np.rad2deg(q_arm_offset_cr[:7]), np.rad2deg(q_arm_offset_cf[:7])):
        print(f"  {name:20s} | Fixed: {val_o:+.4f} deg | Cam Reg (1.0): {val_cr:+.4f} deg | Cam Free (0.0): {val_cf:+.4f} deg")
        
    print("--- Left Arm Offsets ---")
    for name, val_o, val_cr, val_cf in zip(joint_names, np.rad2deg(q_arm_offset_o[7:]), np.rad2deg(q_arm_offset_cr[7:]), np.rad2deg(q_arm_offset_cf[7:])):
        print(f"  {name:20s} | Fixed: {val_o:+.4f} deg | Cam Reg (1.0): {val_cr:+.4f} deg | Cam Free (0.0): {val_cf:+.4f} deg")
        
    print(f"\n--- Camera Offset (Cam Reg) (rpy deg, xyz mm) ---")
    print(f"  rpy: {np.round(xi_cam_cr[:3], 4)} deg")
    print(f"  xyz: {np.round(xi_cam_cr[3:] * 1000.0, 4)} mm")
    
    print(f"\n--- Camera Offset (Cam Free) (rpy deg, xyz mm) ---")
    print(f"  rpy: {np.round(xi_cam_cf[:3], 4)} deg")
    print(f"  xyz: {np.round(xi_cam_cf[3:] * 1000.0, 4)} mm")

if __name__ == "__main__":
    dataset_path = "/home/rainbow/camera_data/result/result_/dataset_20260710_182629.npz"
    run_opt(dataset_path)
