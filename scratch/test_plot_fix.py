import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core")))
sys.path.insert(2, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/calibration")))

from core.calibration.JointCalibrator import JointCalibrator
from core.paths import CONFIG_PATHS

jc = JointCalibrator(None, None)

# Create mock sweep raw data
theta = np.linspace(-np.pi/4, np.pi/4, 20)
pts_a = np.column_stack((np.cos(theta)*100, np.sin(theta)*100, np.zeros_like(theta)))
pts_b = np.column_stack((np.cos(theta)*100 + 10, np.sin(theta)*100 + 5, np.zeros_like(theta)))

mock_sweep_res_1 = {
    '_plot_data': {
        'pts_a_cam': pts_a,
        'pts_b_cam': pts_b,
        'c_A': np.array([0.0, 0.0, 0.0]),
        'c_B': np.array([10.0, 5.0, 0.0]),
        'n_A': np.array([0.0, 0.0, 1.0]),
        'n_B': np.array([0.1, 0.0, 0.99]),
        'r_A': 100.0,
        'r_B': 100.0,
        'angle_between_normals': 5.73,
        'center_dist': 11.18,
    }
}

mock_sweep_res_2 = {
    '_plot_data': {
        'pts_a_cam': pts_a,
        'pts_b_cam': pts_b,
        'c_A': np.array([0.0, 0.0, 0.0]),
        'c_B': np.array([1.0, 0.5, 0.0]),
        'n_A': np.array([0.0, 0.0, 1.0]),
        'n_B': np.array([0.01, 0.0, 0.999]),
        'r_A': 100.0,
        'r_B': 100.0,
        'angle_between_normals': 0.57,
        'center_dist': 1.12,
    }
}

# Test 1: Direct _plot_data res
plot_path_1 = jc.save_calibration_comparison_plot("right", "wrist_pitch_v13", mock_sweep_res_1, mock_sweep_res_2, force_overwrite=True)
print("Test 1 (Direct res) plot path:", plot_path_1)
print("Test 1 file exists and size:", os.path.exists(plot_path_1), os.path.getsize(plot_path_1) if os.path.exists(plot_path_1) else 0)

# Test 2: Nested final_output dict (FullAuto style)
nested_pass1 = {
    'mode': 'wrist_pitch_v13',
    'first_res': mock_sweep_res_1,
    'final_res': mock_sweep_res_1,
}
nested_pass2 = {
    'mode': 'wrist_pitch_v13',
    'first_res': mock_sweep_res_1,
    'final_res': mock_sweep_res_2,
}
plot_path_2 = jc.save_calibration_comparison_plot("right", "wrist_pitch_v13", nested_pass1, nested_pass2, force_overwrite=True)
print("Test 2 (Nested final_output) plot path:", plot_path_2)
print("Test 2 file exists and size:", os.path.exists(plot_path_2), os.path.getsize(plot_path_2) if os.path.exists(plot_path_2) else 0)
