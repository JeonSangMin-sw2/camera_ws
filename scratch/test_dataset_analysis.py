import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')

# Load real robot dataset
data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_072507.npz', allow_pickle=True)
q_arm_list = data['q_arm']
q_head_list = data['q_head']
T_meas_list = data['marker']

print(f"Loaded dataset: q_arm={q_arm_list.shape}, q_head={q_head_list.shape}, marker={T_meas_list.shape}")

import json
with open('/home/rainbow/camera_ws/config/home_reset_baseline.json', 'r') as f:
    baseline = json.load(f)

print("\nFactory Baseline Offsets (deg):")
print("Right:", baseline["right_arm_joint_offset_deg"])
print("Left :", baseline["left_arm_joint_offset_deg"])
