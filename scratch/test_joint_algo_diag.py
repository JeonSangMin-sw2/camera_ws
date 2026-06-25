"""
Joint Calibration Algorithm Diagnostic Test (v3)
=================================================
Confirms that for wrist_pitch_v13 (J5/WP calibration):
  - center-matching |c_J3_FK - c_J3_cam| is FLAT (zero gradient) → wrong criterion
  - radius-matching |r_B_FK(delta) - r_B_cam| has clear gradient → correct criterion

Reason: J3 and J5 both have Y-axis rotation. At ready pose, their axes are nearly
parallel → J5 perturbation moves marker in J3's sweep plane → c_J3 (on J3 axis) is
invariant to J5 offset. RADIUS of J3 sweep changes because marker-to-J3-axis distance
changes when J5 is perturbed.
"""

import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, "/home/rainbow/camera_ws")
sys.path.insert(0, "/home/rainbow/camera_ws/core/calibration")
from CalibratorBase import BaseCalibrator

BASE = "/home/rainbow/camera_ws/core/calibration"

def load_torso_poses(path):
    poses = []
    angles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.split(",")]
            angles.append(parts[0])
            T = np.array(parts[26:42]).reshape(4, 4)
            poses.append(T)
    return poses, angles

def fit(poses, angles, prior):
    return BaseCalibrator.fit_circle_3d_and_6dof_misalignment(
        poses, angles, axis_prior=np.array(prior)
    )

# Load data
poses_A5, angles_A5 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_A_axis_5.txt"))
poses_B3, angles_B3 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_B_axis_3.txt"))

r_A5 = fit(poses_A5, angles_A5, [0,1,0])
r_B3 = fit(poses_B3, angles_B3, [0,1,0])

c_A5 = r_A5['c_opt']; n_A5 = r_A5['axis_opt'] / np.linalg.norm(r_A5['axis_opt'])
c_B3 = r_B3['c_opt']; n_B3 = r_B3['axis_opt'] / np.linalg.norm(r_B3['axis_opt'])
radius_A5 = r_A5['radius']
radius_B3 = r_B3['radius']

pts_B3 = np.array([T[:3, 3] * 1000.0 for T in poses_B3])

print("=" * 65)
print(" [A] Axis Alignment Check: J3 vs J5 axes")
print("=" * 65)
cos_angle = abs(np.dot(n_A5, n_B3))
print(f"  n_J5 (torso frame): {np.round(n_A5, 3)}")
print(f"  n_J3 (torso frame): {np.round(n_B3, 3)}")
print(f"  |cos(J3, J5)| = {cos_angle:.4f}  ({'NEARLY PARALLEL → flat surface!' if cos_angle > 0.9 else 'not parallel'})")
print()

deltas_test = [-15, -10, -5, 0, 5, 10, 15]

print("=" * 65)
print(" [B] center-match surface: |c_J3_FK(delta) - c_J3_cam|")
print("     (Simulate J5 offset by rotating J3-sweep pts about J5 axis)")
print("=" * 65)
center_vals = []
for d in deltas_test:
    R = R_scipy.from_rotvec(np.radians(d) * n_A5).as_matrix()
    rotated = [R @ (p - c_A5) + c_A5 for p in pts_B3]
    c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(rotated, robust=False)
    dist = float(np.linalg.norm(c_fit - c_B3))
    center_vals.append(dist)
    print(f"    delta={d:+5.1f}° → |c_FK - c_cam| = {dist:.3f} mm")

var_c = max(center_vals) - min(center_vals)
print(f"\n  Variation: {var_c:.3f} mm  ← {'FLAT (bad)' if var_c < 2 else 'has gradient'}")

print()
print("=" * 65)
print(" [C] radius-match surface: |r_B_FK(delta) - r_B_cam|")
print("     (Compare FK-predicted J3-sweep RADIUS to camera-measured)")
print("=" * 65)
radius_vals = []
for d in deltas_test:
    R = R_scipy.from_rotvec(np.radians(d) * n_A5).as_matrix()
    rotated = [R @ (p - c_A5) + c_A5 for p in pts_B3]
    fit_res = BaseCalibrator.fit_circle_3d(rotated, robust=False)
    r_fit = float(fit_res[2])   # index 2 = radius
    diff = abs(r_fit - radius_B3)
    radius_vals.append(diff)
    print(f"    delta={d:+5.1f}° → |r_FK - r_cam| = {diff:.3f} mm  (r_FK={r_fit:.2f} mm)")

var_r = max(radius_vals) - min(radius_vals)
print(f"\n  Variation: {var_r:.3f} mm  ← {'FLAT (bad)' if var_r < 2 else 'has gradient (good!)'}")
min_idx = int(np.argmin(radius_vals))
print(f"  Minimum at delta = {deltas_test[min_idx]}°  (|diff| = {radius_vals[min_idx]:.3f} mm)")

print()
print("=" * 65)
print(" [D] Summary")
print("=" * 65)
print(f"""
  r_A  (J5 sweep radius = marker-to-J5-axis dist)  = {radius_A5:.2f} mm
  r_B  (J3 sweep radius = marker-to-J3-axis dist)  = {radius_B3:.2f} mm
  r_B - r_A  (≈ J3-J5 link length, perp component) = {radius_B3 - radius_A5:.2f} mm

  J3 axis (torso): {np.round(n_B3, 3)}
  J5 axis (torso): {np.round(n_A5, 3)}
  Axis alignment:  |cos| = {cos_angle:.4f}

  CONCLUSION:
    J3/J5 axes nearly parallel → c_J3 invariant to J5 offset → center-match FAILS.
    r_B (radius) DOES change with J5 offset → radius-match WORKS.

  CORRECT wrist_pitch_v13 criterion: |r_B_FK(delta) - r_B_camera| → 0
""")
