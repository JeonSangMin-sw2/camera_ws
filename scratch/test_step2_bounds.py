import numpy as np
import json
import sys

sys.path.append('/home/rainbow/camera_ws')

# Load the real dataset from today's run
data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_065350.npz', allow_pickle=True)
q_arm_list = data['q_arm']
q_head_list = data['q_head']
T_meas_list = data['T_meas']

print(f"Loaded dataset: {q_arm_list.shape[0]} samples")

# Let's inspect the bounds logic in calibration_optimizer.py
jo = {
    'right': {'joint3': -2.2039956272497325, 'joint5': 1.8687617405465176, 'joint6': -1.3008088869818764},
    'left': {'joint3': -1.916072094322157, 'joint5': 0.23595425020634764, 'joint6': 1.5559680501386643}
}

print("\n--- Current Bounds in calibration_optimizer.py (WITH NEGATION) ---")
for side in ['right', 'left']:
    for j in ['joint3', 'joint5', 'joint6']:
        val = jo[side][j]
        neg_val = -val
        print(f"  {side} {j}: Step1 Val = {val:+.4f}° -> Bound Centered at: {neg_val:+.4f}°")

print("\n--- Correct Bounds (WITHOUT NEGATION) ---")
for side in ['right', 'left']:
    for j in ['joint3', 'joint5', 'joint6']:
        val = jo[side][j]
        print(f"  {side} {j}: Step1 Val = {val:+.4f}° -> Bound Centered at: {val:+.4f}°")
