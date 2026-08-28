import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

txt_dir = '/home/rainbow/camera_ws/result/result_txt'

def parse_sweep_file(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            T_cam2marker = np.array(parts[10:26]).reshape((4, 4))
            poses.append(T_cam2marker)
    return poses

p4 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_4.txt'))
p5 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_5.txt'))
p6 = parse_sweep_file(os.path.join(txt_dir, 'sweep_points_right_marker_axis_6.txt'))

print(f"Right Arm Poses: P4={len(p4)}, P5={len(p5)}, P6={len(p6)}")

# Let's compute pairwise relative rotations between consecutive frames or from ref frame:
def analyze_axes(poses, label):
    mid_idx = len(poses) // 2
    R_ref = poses[mid_idx][:3, :3]
    axes = []
    angles = []
    for i, T in enumerate(poses):
        if i == mid_idx: continue
        R_rel = R_ref.T @ T[:3, :3]
        rotvec = R_scipy.from_matrix(R_rel).as_rotvec()
        ang = np.linalg.norm(rotvec)
        if ang > np.radians(2.0):
            axis = rotvec / ang
            axes.append(axis)
            angles.append(np.degrees(ang))
    axes = np.array(axes)
    print(f"\n{label}:")
    print(f"  Valid samples (>2 deg): {len(axes)}")
    print(f"  Mean axis: {np.round(np.mean(axes, axis=0), 4)}")
    print(f"  Std axis : {np.round(np.std(axes, axis=0), 4)}")
    return np.mean(axes, axis=0) / np.linalg.norm(np.mean(axes, axis=0))

ax4 = analyze_axes(p4, "Axis 4 (Yaw)")
ax5 = analyze_axes(p5, "Axis 5 (Pitch)")
ax6 = analyze_axes(p6, "Axis 6 (Roll)")

print("\n--- Pairwise Angles between Extracted Axes (deg) ---")
print(f"Angle(Ax4, Ax5): {np.degrees(np.arccos(np.clip(np.dot(ax4, ax5), -1, 1))):.2f}°")
print(f"Angle(Ax5, Ax6): {np.degrees(np.arccos(np.clip(np.dot(ax5, ax6), -1, 1))):.2f}°")
print(f"Angle(Ax4, Ax6): {np.degrees(np.arccos(np.clip(np.dot(ax4, ax6), -1, 1))):.2f}°")
