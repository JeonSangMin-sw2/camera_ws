import sys
import os
import numpy as np
import rby1_sdk as rby

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration.MarkerCalibrator import MarkerCalibrator

def load_latest_iteration_sweep(filepath):
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            lines.append(line.strip())
            
    # Find last occurrences of NEW ITERATION
    start_idx = 0
    for idx, line in enumerate(lines):
        if "=== NEW ITERATION ===" in line:
            start_idx = idx + 1
            
    angles = []
    poses = []
    q_fulls = []
    for line in lines[start_idx:]:
        if not line or line.startswith("#"):
            continue
        parts = [float(p.strip()) for p in line.split(",") if p.strip()]
        angles.append(parts[0])
        T_flat = parts[10:26]
        T = np.array(T_flat).reshape(4, 4)
        poses.append(T)
        
        q_full = np.zeros(30)
        if "axis_4" in filepath:
            q_full[4] = np.radians(parts[0])
        elif "axis_5" in filepath:
            q_full[5] = np.radians(parts[0])
        elif "axis_6" in filepath:
            q_full[6] = np.radians(parts[0])
        q_fulls.append(q_full)
        
    return np.array(angles), np.array(poses), q_fulls

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    calibrator = MarkerCalibrator(marker_st=None, robot=robot)
    calibrator.camera_config = {
        "Tf_to_marker_left": [0.0, 0.0545, -0.050, 90.12, 0.10, 0.00],
        "Tf_to_marker_right": [0.0, -0.0539, -0.046, 90.10, 0.10, 180.00]
    }
    
    base_dir = "/home/rainbow/camera_ws/result/result_txt"
    
    for side in ["right", "left"]:
        print(f"\n================ [{side.upper()} ARM] ================")
        angles_4, poses_4, q_full_4 = load_latest_iteration_sweep(os.path.join(base_dir, f"sweep_points_{side}_marker_axis_4.txt"))
        angles_5, poses_5, q_full_5 = load_latest_iteration_sweep(os.path.join(base_dir, f"sweep_points_{side}_marker_axis_5.txt"))
        angles_6, poses_6, q_full_6 = load_latest_iteration_sweep(os.path.join(base_dir, f"sweep_points_{side}_marker_axis_6.txt"))
        
        # Fit axes/radii
        res_4 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_4, angles_4, axis_prior=[0.0, 0.0, 1.0])
        res_5 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_5, angles_5, axis_prior=[0.0, 1.0, 0.0])
        res_6 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_6, angles_6, axis_prior=[1.0, 0.0, 0.0])
        
        marker_data_6 = {'captured_poses': poses_6, 'captured_q_full': q_full_6, 'radius': res_6['radius'], 'rmse': res_6['rmse']}
        marker_data_5 = {'captured_poses': poses_5, 'captured_q_full': q_full_5, 'radius': res_5['radius'], 'rmse': res_5['rmse']}
        marker_data_4 = {'captured_poses': poses_4, 'captured_q_full': q_full_4, 'radius': res_4['radius'], 'rmse': res_4['rmse']}
        
        print(f"Radii: R6={res_6['radius']:.4f}, R5={res_5['radius']:.4f}, R4={res_4['radius']:.4f}")
        
        res = calibrator.compute_unified_bracket_calibration(marker_data_5, marker_data_6, side, marker_data_4=marker_data_4)
        print("Calibrated Bracket:")
        print(f"  xe: {res['x_e']:.5f}, ye: {res['y_e']:.5f}, ze: {res['z_e']:.5f}")
        print(f"  L_5_ee: {res['L_5_ee']:.4f}")
        
finally:
    robot.disconnect()
