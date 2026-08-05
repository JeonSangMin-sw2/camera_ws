import numpy as np
import json
import os
import sys
import rby1_sdk as rby
import yaml

# Add core path to import configs/kinematics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer, make_transform

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_210630.npz"
setting_path = "/home/rainbow/camera_ws/config/setting.yaml"

if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

with open(setting_path, "r") as f:
    config = yaml.safe_load(f)

# Connect to robot to get kinematics
ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    # Load dataset
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    T_meas = data["marker"]
    
    print(f"Loaded dataset: {q_arm.shape[0]} samples")
    
    # Let's inspect the measured marker positions (R_pos, L_pos) in this new dataset!
    pos_r = T_meas[:, 0, :3, 3] * 1000.0
    pos_l = T_meas[:, 1, :3, 3] * 1000.0
    dists = np.linalg.norm(pos_r - pos_l, axis=1)
    
    print("\n--- Measured Marker Positions (mm) ---")
    print(f"Mean Right Marker: {np.mean(pos_r, axis=0)}")
    print(f"Mean Left Marker:  {np.mean(pos_l, axis=0)}")
    print(f"Mean Distance:     {np.mean(dists):.2f} mm")
    print(f"Std Dev Distance:  {np.std(dists):.2f} mm")
    
finally:
    robot.disconnect()
