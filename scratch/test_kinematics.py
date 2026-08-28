import os
import sys
import numpy as np
import rby1_sdk as rby
from scipy.spatial.transform import Rotation as R_scipy

# Let's inspect using rby1_sdk
robot = rby.create_robot("127.0.0.1", "rby1m") # or create a dummy or use urdf
# Let's see if we can load the dynamics or urdf
print("SDK version:", getattr(rby, '__version__', 'unknown'))
