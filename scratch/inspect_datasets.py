import numpy as np
import os

result_dir = "/home/rainbow/camera_ws/result"
for name in sorted(os.listdir(result_dir)):
    if name.endswith(".npz"):
        path = os.path.join(result_dir, name)
        data = np.load(path)
        print(f"\nDataset: {name}")
        for key in data.files:
            val = data[key]
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            if "arm" in key or "q" in key:
                print(f"    first element: {val[0] if len(val) > 0 else 'empty'}")
