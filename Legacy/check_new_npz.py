import numpy as np
from core.calibration_core import load_npz_dataset
try:
    q_arm, q_head, T_meas = load_npz_dataset("result/dataset_20260507_192130.npz")
    print("q_arm shape:", q_arm.shape)
    if q_head is not None:
        print("q_head shape:", q_head.shape)
    print("T_meas shape:", T_meas.shape)
    
    if len(q_arm) > 0:
        # Measure pose variance
        poses = T_meas[:, 0] if T_meas.ndim == 4 else T_meas
        trans = poses[:, :3, 3]
        std_trans = np.std(trans, axis=0)
        print("Translation Std Dev (m):", np.round(std_trans, 3))
except Exception as e:
    print("Error:", e)
