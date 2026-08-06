import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.MarkerCalibrator import MarkerCalibrator

def load_sweep_data(filepath):
    angles = []
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("#") or not line_str or "==" in line_str:
                continue
            parts = [float(p.strip()) for p in line_str.split(",")]
            angles.append(parts[0])
            T_flat = parts[10:26]
            T = np.array(T_flat).reshape(4, 4)
            poses.append(T)
    return np.array(angles), np.array(poses)

def analyze_circle_distortion(poses, angles, label):
    # Fit circle
    calibrator = MarkerCalibrator(marker_st=None, robot="mock_robot")
    res = calibrator.fit_circle_3d_and_6dof_misalignment(poses, angles, robust=True)
    
    # Get 3D points
    pts = np.array([T[:3, 3] * 1000.0 for T in poses]) # mm
    center = res['c_opt'] # mm
    normal = res['axis_opt']
    radius = res['radius']
    
    # Calculate distance from center for each point
    dists = np.linalg.norm(pts - center, axis=1)
    errors = dists - radius
    
    print(f"\nAnalysis for {label}:")
    print(f"  Radius: {radius:.3f} mm")
    print(f"  RMSE: {res['rmse']:.3f} mm")
    print(f"  Max Residual: {np.max(np.abs(errors)):.3f} mm")
    print(f"  Min Residual: {np.min(np.abs(errors)):.3f} mm")
    print(f"  Std Dev of Residuals: {np.std(errors):.3f} mm")
    
    # Check if residuals are correlated with coordinates (systematic distortion)
    # E.g. correlation between Z-coordinate (depth) and residual error
    depths = pts[:, 2]
    corr_depth = np.corrcoef(depths, errors)[0, 1]
    print(f"  Correlation (Depth vs Residual): {corr_depth:.3f}")
    
    # Correlation with X-coordinate (horizontal position in image)
    corr_x = np.corrcoef(pts[:, 0], errors)[0, 1]
    print(f"  Correlation (Horizontal vs Residual): {corr_x:.3f}")

def main():
    base_dir = "/home/rainbow/camera_ws/result/result_txt"
    
    # Load actual files
    angles_4, poses_4 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_4.txt"))
    angles_5, poses_5 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_5.txt"))
    angles_6, poses_6 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_6.txt"))
    
    analyze_circle_distortion(poses_4, angles_4, "Axis 4 (Wrist Yaw) Sweep")
    analyze_circle_distortion(poses_5, angles_5, "Axis 5 (Wrist Pitch) Sweep")
    analyze_circle_distortion(poses_6, angles_6, "Axis 6 (Wrist Roll) Sweep")

if __name__ == "__main__":
    main()
