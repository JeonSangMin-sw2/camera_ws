from core.calibration_core import make_transform
import numpy as np

T1 = make_transform([0, 0, 0, 90, 0, 180])
T2 = make_transform([0, 0, 0, -90, 0, 180])

print("T1 (Roll=90, Yaw=180):\n", np.round(T1[:3, :3], 2))
print("T2 (Roll=-90, Yaw=180):\n", np.round(T2[:3, :3], 2))
