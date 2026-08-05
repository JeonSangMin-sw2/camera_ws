import numpy as np
import json
import os
import sys

# Add core path to import configs/kinematics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_194834.npz"

if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

data = np.load(dataset_path)
print("Keys in NPZ:", list(data.keys()))

q_arm = data["q_arm"]
q_full = data["q"]
T_meas = data["marker"]

print(f"q_arm shape: {q_arm.shape}")
print(f"q_full shape: {q_full.shape}")
print(f"T_meas shape: {T_meas.shape}")

# T_meas is shape (N, 2, 4, 4) where index 0 is right arm and index 1 is left arm.
# Let's inspect the positions of the markers in the camera frame.
num_samples = T_meas.shape[0]
distances_meas = []

print("\n--- Sample Position Analysis (mm) ---")
print(f"{'Sample':<8} | {'Right Marker (X, Y, Z)':<25} | {'Left Marker (X, Y, Z)':<25} | {'Distance (mm)':<12}")
print("-" * 80)

for i in range(num_samples):
    T_r = T_meas[i, 0]
    T_l = T_meas[i, 1]
    
    pos_r = T_r[:3, 3] * 1000.0 # to mm
    pos_l = T_l[:3, 3] * 1000.0 # to mm
    
    dist = np.linalg.norm(pos_r - pos_l)
    distances_meas.append(dist)
    
    if i < 10 or i >= num_samples - 5: # Print first 10 and last 5
        r_str = f"[{pos_r[0]:.1f}, {pos_r[1]:.1f}, {pos_r[2]:.1f}]"
        l_str = f"[{pos_l[0]:.1f}, {pos_l[1]:.1f}, {pos_l[2]:.1f}]"
        print(f"{i+1:<8} | {r_str:<25} | {l_str:<25} | {dist:.2f}")
    elif i == 10:
        print("...")

distances_meas = np.array(distances_meas)
print("\n--- Marker Distance Statistics (Measured by Camera) ---")
print(f"Mean distance: {np.mean(distances_meas):.2f} mm")
print(f"Min distance:  {np.min(distances_meas):.2f} mm")
print(f"Max distance:  {np.max(distances_meas):.2f} mm")
print(f"Std dev:       {np.std(distances_meas):.2f} mm")
print(f"Max difference: {np.max(distances_meas) - np.min(distances_meas):.2f} mm")
