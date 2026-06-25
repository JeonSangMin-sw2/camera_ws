"""
Joint Calibration Algorithm Diagnostic Test (v2)
=================================================
Previous diagnosis found:
  - c_J3 is on J3 axis, ~310 mm from J5 axis (NOT 161.5 mm)
  - My fix (target = r_A = 161.5 mm) was WRONG
  - Original code (target = 0) was WRONG
  - perp_dist from J5 axis is nearly constant regardless of delta → bad criterion

Correct criterion:
  Compare FK-predicted c_J3(delta) to camera-measured c_J3 (from torso-frame data).
  At correct delta: c_J3_FK(delta) ≈ c_J3_camera → minimize 3D center distance.
"""

import os, sys
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, "/home/rainbow/camera_ws")
sys.path.insert(0, "/home/rainbow/camera_ws/core/calibration")
from CalibratorBase import BaseCalibrator

BASE = "/home/rainbow/camera_ws/core/calibration"

def load_torso_poses(path):
    """Returns list of T_t5_to_marker (4x4) and joint angles from the data file."""
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
poses_A6, angles_A6 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_A_axis_6.txt"))
poses_B5, angles_B5 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_B_axis_5.txt"))
poses_A5, angles_A5 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_A_axis_5.txt"))
poses_B3, angles_B3 = load_torso_poses(os.path.join(BASE, "sweep_points_right_joint_B_axis_3.txt"))

r_A6 = fit(poses_A6, angles_A6, [1,0,0])
r_B5_pitch = fit(poses_B5, angles_B5, [0,1,0])
r_A5 = fit(poses_A5, angles_A5, [0,1,0])
r_B3 = fit(poses_B3, angles_B3, [0,1,0])

c_A6 = r_A6['c_opt'];   n_A6 = r_A6['axis_opt']   / np.linalg.norm(r_A6['axis_opt'])
c_B5 = r_B5_pitch['c_opt']
c_A5 = r_A5['c_opt'];   n_A5 = r_A5['axis_opt']   / np.linalg.norm(r_A5['axis_opt'])
c_B3 = r_B3['c_opt'];   n_B3 = r_B3['axis_opt']   / np.linalg.norm(r_B3['axis_opt'])

def perp(c, c_ax, n_ax):
    v = c - c_ax; return float(np.linalg.norm(v - np.dot(v, n_ax)*n_ax))

# ---------------------------------------------------------------------------
# [A] 기하학 확인: c_J3는 J3 축 위 → J5 축과 무관한 거리를 가짐
# ---------------------------------------------------------------------------
print("=" * 65)
print(" [A] GEOMETRY CHECK: where is c_J3 relative to J5 axis?")
print("=" * 65)
print(f"  c_J3 (camera-measured, torso frame): {np.round(c_B3, 2)} mm")
print(f"  c_J5 (from J5-sweep A):              {np.round(c_A5, 2)} mm")
print(f"  c_J6 (from J6-sweep A):              {np.round(c_A6, 2)} mm")
print()
print(f"  perp_dist(c_J3, J5_axis) = {perp(c_B3, c_A5, n_A5):.2f} mm  ← ≈ J3-J5 link distance")
print(f"  perp_dist(c_J5, J6_axis) = {perp(c_B5, c_A6, n_A6):.2f} mm  ← should be ~0 (wrist center)")
print()
print(f"  r_A5 (J5 sweep radius = marker to J5 axis) = {r_A5['radius']:.2f} mm")
print(f"  r_B3 (J3 sweep radius = marker to J3 axis) = {r_B3['radius']:.2f} mm")
print(f"  difference r_B3 - r_A5                     = {r_B3['radius'] - r_A5['radius']:.2f} mm  ← ≈ J3-J5 link")
print()
print("  CONCLUSION: c_J3 is ~310 mm from J5 axis = J3-to-J5 link distance.")
print("  My previous claim 'c_J3 must be 161.5 mm from J5 axis' = WRONG.")
print("  My code fix (target = r_A = 161.5 mm) = WRONG.")

# ---------------------------------------------------------------------------
# [B] perp_dist surface flatness: confirm it doesn't change with delta
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print(" [B] SURFACE FLATNESS: perp_dist(c_J3_rotated, J5_axis) vs delta")
print("=" * 65)
print("  (Simulating FK perturb of J5 by rotating J3-sweep points about J5 axis)")

pts_B3 = np.array([T[:3, 3] * 1000.0 for T in poses_B3])
deltas_test = [-15, -10, -5, 0, 5, 10, 15]
perp_vals = []
for d in deltas_test:
    R = R_scipy.from_rotvec(np.radians(d) * n_A5).as_matrix()
    rotated = [R @ (p - c_A5) + c_A5 for p in pts_B3]
    c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(rotated, robust=False)
    pv = perp(c_fit, c_A5, n_A5)
    perp_vals.append(pv)
    print(f"    delta={d:+5.1f}° → perp_dist = {pv:.2f} mm  (c_fit={np.round(c_fit, 1)})")

variation = max(perp_vals) - min(perp_vals)
print(f"\n  Total variation = {variation:.2f} mm  ← {'FLAT (bad criterion)' if variation < 10 else 'has gradient'}")

# ---------------------------------------------------------------------------
# [C] CORRECT APPROACH: compare FK-predicted c_J3 to camera-measured c_J3
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print(" [C] CORRECT APPROACH: |c_J3_FK(delta) - c_J3_camera|")
print("=" * 65)
print(f"  c_J3_camera = {np.round(c_B3, 2)} mm (from torso-frame data)")
print()
print("  Simulating J5 offset by rotating J3-sweep points about J5 axis,")
print("  then comparing resulting circle center to c_J3_camera.")
print("  (This approximates FK perturb without actual robot FK)")
print()

dist_vals = []
for d in deltas_test:
    R = R_scipy.from_rotvec(np.radians(d) * n_A5).as_matrix()
    rotated = [R @ (p - c_A5) + c_A5 for p in pts_B3]
    c_fit, _, _, _, _, _, _ = BaseCalibrator.fit_circle_3d(rotated, robust=False)
    dist = float(np.linalg.norm(c_fit - c_B3))
    dist_vals.append(dist)
    print(f"    delta={d:+5.1f}° → |c_FK - c_cam| = {dist:.3f} mm")

variation_c = max(dist_vals) - min(dist_vals)
print(f"\n  Total variation = {variation_c:.2f} mm  ← {'FLAT (bad)' if variation_c < 1 else 'has gradient (good)'}")
min_idx = int(np.argmin(dist_vals))
print(f"  Minimum at delta = {deltas_test[min_idx]}°  (dist = {dist_vals[min_idx]:.3f} mm)")

# ---------------------------------------------------------------------------
# [D] Summary: what the CORRECT calibration criterion should be
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print(" [D] SUMMARY OF ERRORS AND CORRECT APPROACH")
print("=" * 65)
print("""
  WHAT WAS WRONG (wrist_roll_v13):
    ✗ Original code: perp_dist(c_J3_FK, J5_axis) → 0
        Reason wrong: c_J3 is on J3 axis, ~310 mm from J5 axis regardless of J5 offset.
                      Optimization surface is nearly flat → no meaningful minimum.

    ✗ My fix:        perp_dist(c_J3_FK, J5_axis) → r_A (161.5 mm)
        Reason wrong: 161.5 mm (marker-to-J5 distance) has no geometric relationship
                      to c_J3's distance from J5 axis. c_J3 is a different geometric
                      point (center of J3 sweep circle = point on J3 axis ≈ 310 mm away).

  WHAT SHOULD BE DONE:
    ✓ Correct criterion: |c_J3_FK(delta) - c_J3_camera| → 0
        Where:
          c_J3_camera = circle center from camera-measured torso-frame poses (dataset_B)
          c_J3_FK(delta) = circle center from FK-predicted marker positions with J5 perturbed by delta

        When delta = true_J5_offset: FK matches reality → c_J3_FK ≈ c_J3_camera → residual ≈ 0
""")
