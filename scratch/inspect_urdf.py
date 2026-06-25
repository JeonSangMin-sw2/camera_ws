import rby1_sdk.dynamics as rd
import os

urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
print("URDF path exists:", os.path.exists(urdf_path))

# Let's inspect base link by reading first 50 lines
with open(urdf_path, 'r') as f:
    for i in range(50):
        print(f.readline(), end='')
