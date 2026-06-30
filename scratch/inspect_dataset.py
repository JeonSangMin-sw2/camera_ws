import numpy as np

path = "/home/rainbow/camera_ws/result/dataset_20260616_113303.npz"
data = np.load(path)
for k in data.files:
    print(f"Key: {k}, Shape: {data[k].shape}")
    
# Let's print some q values
print("\nq_arm sample (first 2):")
print(data['q_arm'][:2])

# Let's print some marker measurements
print("\nmarker right (first sample):")
print(data['marker'][0, 0])
print("\nmarker left (first sample):")
print(data['marker'][0, 1])
