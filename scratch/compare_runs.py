import numpy as np
import json
import os

res_dir = '/home/rainbow/camera_ws/result/result_step2'

f_759 = os.path.join(res_dir, 'result_20260828_075907.json')
f_820 = os.path.join(res_dir, 'result_20260828_082005.json')

with open(f_759, 'r') as f:
    d_759 = json.load(f)

with open(f_820, 'r') as f:
    d_820 = json.load(f)

print("=== 07:59 (Commit 7b963e4) Result ===")
print("Right Arm:", np.round(d_759['right_arm_joint_offset_deg'], 4))
print("Left Arm :", np.round(d_759['left_arm_joint_offset_deg'], 4))
print("Head     :", np.round(d_759['head_joint_offset_deg'], 4))

print("\n=== 08:20 (Latest) Result ===")
print("Right Arm:", np.round(d_820['right_arm_joint_offset_deg'], 4))
print("Left Arm :", np.round(d_820['left_arm_joint_offset_deg'], 4))
print("Head     :", np.round(d_820['head_joint_offset_deg'], 4))

# Let's check the npz dataset differences
npz_759 = np.load(os.path.join(res_dir, 'dataset_20260828_075907.npz'), allow_pickle=True)
npz_820 = np.load(os.path.join(res_dir, 'dataset_20260828_082005.npz'), allow_pickle=True)

print("\n=== Dataset Comparison ===")
print("759 keys:", list(npz_759.keys()))
print("820 keys:", list(npz_820.keys()))

if 'Tf_to_marker_right' in npz_759:
    print("759 Tf_right:", npz_759['Tf_to_marker_right'])
    print("759 Tf_left :", npz_759['Tf_to_marker_left'])

if 'Tf_to_marker_right' in npz_820:
    print("820 Tf_right:", npz_820['Tf_to_marker_right'])
    print("820 Tf_left :", npz_820['Tf_to_marker_left'])

# Let's compare joint angles captured in both datasets
q_759 = npz_759['q_meas']
q_820 = npz_820['q_meas']
print("Shape q_759:", q_759.shape, "q_820:", q_820.shape)
print("q_diff max (deg):", np.degrees(np.max(np.abs(q_759 - q_820))))

# Let's compare marker measurements
T_m_759 = npz_759['T_cam_to_marker_meas']
T_m_820 = npz_820['T_cam_to_marker_meas']
print("T_m_diff max trans (mm):", np.max(np.abs(T_m_759[:, :, :3, 3] - T_m_820[:, :, :3, 3])) * 1000.0)
