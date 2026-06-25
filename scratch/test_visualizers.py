import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core")))
sys.path.insert(2, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/calibration")))

from IntrinsicsCalibrator import IntrinsicsCalibrator
from MarkerCalibrator import MarkerCalibrator

# Test Intrinsics verification image
ic = IntrinsicsCalibrator()
ic.rms_error = 0.15
ic.cameraMatrix = np.eye(3)
ic.distCoeffs = np.zeros((5, 1))

dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw some patterns to look like a camera feed
dummy_img[100:380, 150:490] = 255

save_path_ic = os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/calibration/result_img/test_intrinsics_verification.png"))
ic.generate_verification_image(dummy_img, save_path_ic)
print("Intrinsics verification image generated:", os.path.exists(save_path_ic))

# Test Marker unified plot
mc = MarkerCalibrator(None, None)
theta = np.linspace(-np.pi/18, np.pi/18, 11)
res_6 = {
    'radius': 74.85,
    'rmse': 0.12,
    'pts_2d': np.column_stack((np.cos(theta)*74.8 + 0.1, np.sin(theta)*74.8 - 0.2)),
    'uc_opt': 0.1,
    'vc_opt': -0.2,
}
res_5 = {
    'radius': 280.15,
    'rmse': 0.18,
    'pts_2d': np.column_stack((np.cos(theta)*280.2 + 0.5, np.sin(theta)*280.2 + 0.4)),
    'uc_opt': 0.5,
    'vc_opt': 0.4,
}
res_4 = {
    'radius': 100.25,
    'rmse': 0.15,
    'pts_2d': np.column_stack((np.cos(theta)*100.2 + 0.3, np.sin(theta)*100.2 + 0.1)),
    'uc_opt': 0.3,
    'vc_opt': 0.1,
}
unified_res = {
    'x_e': 0.0, 'y_e': 75.0, 'z_e': -50.0,
    'roll_e': 0.1, 'pitch_e': -0.2, 'yaw_e': 180.0,
}

save_path_mc = os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/calibration/result_img/test_marker_unified.png"))
mc.generate_marker_plot(res_5, res_6, res_4, unified_res, "right", True, save_path_mc)
print("Marker unified plot generated:", os.path.exists(save_path_mc))
