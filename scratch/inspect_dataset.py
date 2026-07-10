import numpy as np

path = "/home/rainbow/camera_data/result/result_/dataset_20260710_182629.npz"
data = np.load(path)
print("Keys in npz file:", list(data.keys()))

for k in data.keys():
    val = data[k]
    print(f"\nKey: {k}, Shape: {val.shape}, Type: {val.dtype}")
    if val.ndim == 1:
        print("Values:", val[:10])
    elif val.ndim == 2:
        print("First 2 rows:\n", val[:2])
    elif val.ndim == 3:
        print("First row:\n", val[0])
    elif val.ndim == 4:
        print("First sample first arm:\n", val[0, 0])
