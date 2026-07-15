import sys
sys.path.append("/home/rainbow/camera_ws")
from core.calibration.JointCalibrator import JointCalibrator

# let's just do a string replacement check
with open("/home/rainbow/camera_ws/core/calibration/JointCalibrator.py", "r") as f:
    code = f.read()

import re
match = re.search(r"nominal_joint_pos = initial_joint_pos\[cand_joint\].*?\n.*?q_cand\[cand_joint\] = nominal_joint_pos \+ np\.radians\(current_offset_deg\)", code, re.DOTALL)
if match:
    print("Found the buggy lines:")
    print(match.group(0))
