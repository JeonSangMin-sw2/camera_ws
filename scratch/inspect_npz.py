import numpy as np

data = np.load('/home/rainbow/camera_ws/result/result_step2/dataset_20260828_065350.npz', allow_pickle=True)
print("Keys in npz:", list(data.keys()))
for k in data.keys():
    print(f"  {k}: shape = {data[k].shape}")
