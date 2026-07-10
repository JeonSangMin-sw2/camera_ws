import numpy as np

path = "/home/rainbow/camera_data/result/result_/dataset_20260710_182629.npz"
data = np.load(path)

print("RIGHT ARM MARKER (marker[0, 0]):")
print(data['marker'][0, 0])

print("\nLEFT ARM MARKER (marker[0, 1]):")
print(data['marker'][0, 1])

print("\nRIGHT ARM COMMAND (q[0, :7]):")
print(np.round(np.rad2deg(data['q'][0, :7]), 2))

print("\nLEFT ARM COMMAND (q[0, 7:]):")
print(np.round(np.rad2deg(data['q'][0, 7:]), 2))
