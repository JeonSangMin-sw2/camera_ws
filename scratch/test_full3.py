import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

theta_6_enc = -10.0
true_offset = 2.3 # Now let's inject 2.3 for J7 and see what happens!
theta_6_phys = theta_6_enc + true_offset

R_bracket = R_scipy.from_euler('ZYX', [0.05, -0.1, -0.1], degrees=True).as_matrix()
R_ee_m_ideal = R_scipy.from_euler('ZYX', [180, 0, 90], degrees=True).as_matrix()
R_ee_m_gt = R_bracket @ R_ee_m_ideal

# 1. Sweep J5 (Joint 6)
# J5 nominal axis in EE is [0,1,0]. But wait! J5 rotates the EE!
# EE is attached to J5. So when J5 rotates, the EE rotates!
# Let's generate the poses for the sweep!
theta_5_sweep = np.linspace(-20, 20, 50)
positions = []
Rs = []
for th5 in theta_5_sweep:
    # Kinematics:
    # Cam to J6 = Identity (for simplicity)
    # J6 rotates by theta_6_phys around Z
    R_j6 = R_scipy.from_euler('z', theta_6_phys, degrees=True).as_matrix()
    
    # J5 rotates by th5 around Y
    # Wait, in the robot, does J5 rotate around Y?
    # Yes, n_nom_v12 for axis_5 is [0, 1, 0]
    R_j5 = R_scipy.from_euler('y', th5, degrees=True).as_matrix()
    
    # FK: R_cam_to_ee = R_j6 @ R_j5
    # Wait! In the UR kinematics, J5 comes BEFORE J6? Or AFTER J6?
    # axis_5 is J6, axis_6 is J7. So J5 (axis_5) comes BEFORE J6 (axis_6)!!
    # So R_cam_to_ee = R_j5 @ R_j6
    R_cam_to_ee = R_j5 @ R_j6
    
    T_cam_to_ee = np.eye(4)
    T_cam_to_ee[:3, :3] = R_cam_to_ee
    # The EE has some length L from J5.
    L = np.array([0.0, 0.0, 0.1])
    # Position of EE in Cam:
    pos_ee = R_j5 @ R_j6 @ np.array([0,0,0]) + R_j5 @ L
    
    T_cam_to_m = R_cam_to_ee @ R_ee_m_gt
    pos_m = pos_ee + R_cam_to_ee @ np.array([0.0005, 0.0, 0.002]) # bracket_pos
    
    positions.append(pos_m)
    Rs.append(T_cam_to_m)

positions = np.array(positions)
# Fit plane to find axis_opt
centroid = np.mean(positions, axis=0)
centered = positions - centroid
U, S, Vt = np.linalg.svd(centered)
n5_cam = Vt[-1]

# Make sure n5_cam points in positive Y (as J5 axis is Y)
if np.dot(n5_cam, [0, 1, 0]) < 0:
    n5_cam = -n5_cam

# R_ref_5 is the center of the sweep (th5 = 0)
R_ref_5 = Rs[len(Rs)//2]

n5_marker_actual = R_ref_5.T @ n5_cam

# JoinCalibrator logic
y_ee_m_ideal = R_ee_m_ideal.T @ np.array([0, 1.0, 0])
z_axis = R_ee_m_ideal.T @ np.array([0, 0, 1.0])
ref_y = y_ee_m_ideal
ref_x = np.cross(z_axis, ref_y)

n5_proj = n5_marker_actual - np.dot(n5_marker_actual, z_axis) * z_axis
n5_proj /= np.linalg.norm(n5_proj)

diff_angle = np.degrees(np.arctan2(np.dot(n5_proj, ref_x), np.dot(n5_proj, ref_y)))
print("diff_angle:", diff_angle)

