import numpy as np
import json
import os

res_dir = '/home/rainbow/camera_ws/result/result_step2'

npz_759 = np.load(os.path.join(res_dir, 'dataset_20260828_075907.npz'), allow_pickle=True)
npz_820 = np.load(os.path.join(res_dir, 'dataset_20260828_082005.npz'), allow_pickle=True)

print("759 marker shape:", npz_759['marker'].shape)
print("820 marker shape:", npz_820['marker'].shape)

print("759 q shape:", npz_759['q'].shape)
print("820 q shape:", npz_820['q'].shape)

# Check differences
q_diff = np.max(np.abs(npz_759['q'] - npz_820['q']))
print("q diff max (deg):", np.degrees(q_diff))

m_diff = np.max(np.abs(npz_759['marker'] - npz_820['marker']))
print("marker diff max:", m_diff)
