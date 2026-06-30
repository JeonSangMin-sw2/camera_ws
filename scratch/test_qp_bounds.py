import os
import sys
import numpy as np

# Ensure workspace paths are in sys.path
sys.path.append("/home/rainbow/camera_ws")

from core.calibration.mock_robot import OfflineRobot
from core.calibration_core import load_npz_dataset, get_both_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer

def load_offline_robot():
    import rby1_sdk.dynamics as rd
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    return OfflineRobot(dyn_robot, "m", "1.2")

def run_opt(dataset_path, apply_limits, use_old_bracket, use_symmetric_bounds):
    print(f"\n=======================================================")
    print(f"Running QP Optimization for dataset: {os.path.basename(dataset_path)}")
    print(f"Apply Joint Offset Limits (apply_limits): {apply_limits}")
    print(f"Use Old Bracket Nominals: {use_old_bracket}")
    print(f"Use Symmetric Bounds for J1/J3: {use_symmetric_bounds}")
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

    cfg = get_both_arm_config(robot.model())
    head_cfg = get_head_config(robot.model())
        
    active_arms = ["right", "left"]
    ee_links = cfg["ee_links"]
    
    # Choose bracket nominals
    if use_old_bracket:
        # Old values from git diff before they were shortened
        ee_to_marker_nom = {
            "left": [0.0, 0.07748, -0.07162, 90.02, -0.00, -0.00],
            "right": [0.0, -0.04704, 0.06292, 88.97, -0.29, 182.81]
        }
    else:
        ee_to_marker_nom = cfg["ee_to_marker_nom"]
        
    print(f"EE to Marker Left: {ee_to_marker_nom['left']}")
    print(f"EE to Marker Right: {ee_to_marker_nom['right']}")
    
    joint_offsets = None
    if apply_limits:
        # Use J3 and J5 offsets (locked to config values)
        # For simplicity in this test, we load from setting.yaml
        import yaml
        with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
            data = yaml.safe_load(f)
        jo = data.get("joint_offset", {})
        joint_offsets = {
            "right": {
                "joint3": jo.get("right", {}).get("joint3", 0.0),
                "joint5": jo.get("right", {}).get("joint5", 0.0),
            },
            "left": {
                "joint3": jo.get("left", {}).get("joint3", 0.0),
                "joint5": jo.get("left", {}).get("joint5", 0.0),
            }
        }

    optimizer = QPCalibrationOptimizer(
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
        apply_joint_offset_limits=apply_limits,
        joint_offsets_to_apply=joint_offsets,
    )
    
    # Override bounds dynamically if requested
    D2R = np.pi / 180.0
    if use_symmetric_bounds:
        # Modify J1 (shoulder roll) to be symmetric [-10, 10] deg
        optimizer.joint_offset_lower[1] = -10.0 * D2R
        optimizer.joint_offset_upper[1] =  10.0 * D2R
        optimizer.joint_offset_lower[8] = -10.0 * D2R
        optimizer.joint_offset_upper[8] =  10.0 * D2R
        
        # Modify J3 (elbow pitch) to be symmetric [-10, 10] deg if not overridden by apply_limits
        if not apply_limits:
            optimizer.joint_offset_lower[3] = -10.0 * D2R
            optimizer.joint_offset_upper[3] =  10.0 * D2R
            optimizer.joint_offset_lower[10] = -10.0 * D2R
            optimizer.joint_offset_upper[10] =  10.0 * D2R
            
    # Run optimization
    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
        q_arm,
        q_head,
        T_meas
    )
    
    right_offset = np.rad2deg(q_arm_offset[:7])
    left_offset = np.rad2deg(q_arm_offset[7:])
    
    joint_names = ["shoulder_yaw (J0)", "shoulder_roll (J1)", "shoulder_pitch (J2)", "elbow_pitch (J3)", "wrist_yaw (J4)", "wrist_pitch (J5)", "wrist_roll (J6)"]
    print("--- Right Arm Offsets ---")
    for name, val in zip(joint_names, right_offset):
        print(f"  {name}: {val:+.4f} deg")
        
    print("--- Left Arm Offsets ---")
    for name, val in zip(joint_names, left_offset):
        print(f"  {name}: {val:+.4f} deg")
        
    print(f"--- Camera Extrinsics (rpy deg, xyz mm) ---")
    print(f"  rpy: {np.round(xi_cam[:3], 4)} deg")
    print(f"  xyz: {np.round(xi_cam[3:] * 1000.0, 4)} mm")

if __name__ == "__main__":
    dataset_path = "/home/rainbow/camera_ws/result/dataset_20260608_150837.npz"
        
    print("=================== TEST 1: ORIGINAL DEFAULT BOUNDS (LIMITS FALSE) ===================")
    run_opt(dataset_path, apply_limits=False, use_old_bracket=True, use_symmetric_bounds=False)
    
    print("\n=================== TEST 2: PROPOSED SYMMETRIC BOUNDS (LIMITS FALSE) ===================")
    run_opt(dataset_path, apply_limits=False, use_old_bracket=True, use_symmetric_bounds=True)
