import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')

def parse_sweep_file(filepath):
    poses = []
    angles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            angles.append(parts[0])
            T = np.array(parts[10:26]).reshape((4, 4))
            poses.append(T)
    return poses, np.array(angles)

# Load Right Arm sweep data
p4, a4 = parse_sweep_file('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_4.txt')
p5, a5 = parse_sweep_file('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_5.txt')
p6, a6 = parse_sweep_file('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_6.txt')

# 1. Fit circle planes from 3D positions in camera frame
def fit_plane_normal(poses):
    pts = np.array([T[:3, 3]*1000 for T in poses])
    c0 = np.mean(pts, axis=0)
    _, _, vt = np.linalg.svd(pts - c0)
    normal = vt[2]
    return normal / np.linalg.norm(normal), c0, pts

n4_cam, c4, pts4 = fit_plane_normal(p4)
n5_cam, c5, pts5 = fit_plane_normal(p5)
n6_cam, c6, pts6 = fit_plane_normal(p6)

print("=== 3D Position Plane Normals in Camera Frame ===")
print("n4_cam:", np.round(n4_cam, 4))
print("n5_cam:", np.round(n5_cam, 4))
print("n6_cam:", np.round(n6_cam, 4))

dot_45 = abs(np.dot(n4_cam, n5_cam))
dot_56 = abs(np.dot(n5_cam, n6_cam))
dot_46 = abs(np.dot(n4_cam, n6_cam))

ang_45 = np.degrees(np.arccos(np.clip(dot_45, -1, 1)))
ang_56 = np.degrees(np.arccos(np.clip(dot_56, -1, 1)))
ang_46 = np.degrees(np.arccos(np.clip(dot_46, -1, 1)))

print(f"Angle(N4, N5): {ang_45:.2f}°")
print(f"Angle(N5, N6): {ang_56:.2f}°")
print(f"Angle(N4, N6): {ang_46:.2f}°")
