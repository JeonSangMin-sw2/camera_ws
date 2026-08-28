import os
import sys
import numpy as np

sys.path.append('/home/rainbow/camera_ws')

def inspect_sweep(filepath, axis_name):
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
    
    pts = np.array([T[:3, 3]*1000 for T in poses])
    angles = np.array(angles)
    
    print(f"\n--- {axis_name} ---")
    print(f"Total points: {len(pts)}")
    print(f"Angle range : {angles.min():.2f}° to {angles.max():.2f}° (Span: {angles.max() - angles.min():.2f}°)")
    print(f"Start pos   : {np.round(pts[0], 2)} mm")
    print(f"End pos     : {np.round(pts[-1], 2)} mm")
    chord = np.linalg.norm(pts[-1] - pts[0])
    span_rad = np.radians(angles.max() - angles.min())
    r_geom = chord / (2 * np.sin(span_rad / 2))
    print(f"Chord length: {chord:.2f} mm")
    print(f"Geometric R : {r_geom:.2f} mm")
    
    # Check intermediate points
    mid_pt = pts[len(pts)//2]
    chord_mid = (pts[0] + pts[-1]) / 2.0
    sagitta = np.linalg.norm(mid_pt - chord_mid)
    print(f"Sagitta (H) : {sagitta:.2f} mm")
    r_sag = (chord**2) / (8 * sagitta) + sagitta / 2.0 if sagitta > 0.1 else 0.0
    print(f"Sagitta R   : {r_sag:.2f} mm")

inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_4.txt', 'Right Axis 4 (Yaw)')
inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_5.txt', 'Right Axis 5 (Pitch)')
inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_right_marker_axis_6.txt', 'Right Axis 6 (Roll)')

inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_left_marker_axis_4.txt', 'Left Axis 4 (Yaw)')
inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_left_marker_axis_5.txt', 'Left Axis 5 (Pitch)')
inspect_sweep('/home/rainbow/camera_ws/result/result_txt/sweep_points_left_marker_axis_6.txt', 'Left Axis 6 (Roll)')
