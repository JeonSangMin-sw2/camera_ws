import numpy as np
npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260729_192805.npz"
data = np.load(npz_path)
for key in data.files:
    print(f"Key: {key}, Shape: {data[key].shape}")
