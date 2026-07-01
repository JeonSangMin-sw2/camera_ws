import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.CalibratorBase import BaseCalibrator

def load_data(path):
    angles = []
    poses = []
    cam_pts = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(x.strip()) for x in line.strip().split(",")]
            angles.append(parts[0])
            cam_pts.append(parts[1:4]) # in mm
            # T_cam2marker is in parts[10:26]
            T = np.array(parts[10:26]).reshape(4, 4)
            poses.append(T)
    return np.array(angles), np.array(poses), np.array(cam_pts)

def main():
    path_A = "/home/rainbow/camera_ws/sweep_points_right_joint_A_axis_4 (2).txt"
    path_B = "/home/rainbow/camera_ws/sweep_points_right_joint_B_axis_6 (2).txt"

    angles_A, poses_A, cam_pts_A = load_data(path_A)
    angles_B, poses_B, cam_pts_B = load_data(path_B)

    print(f"Loaded Sweep A (Axis 4): {len(angles_A)} points")
    print(f"Loaded Sweep B (Axis 6): {len(angles_B)} points")

    # Fit circles using standard 3D circle fit (unconstrained) to check distortion
    c_fit_A, R_fit_A, r_fit_A, rmse_fit_A, _, _, _ = BaseCalibrator.fit_circle_3d(cam_pts_A, robust=True)
    c_fit_B, R_fit_B, r_fit_B, rmse_fit_B, _, _, _ = BaseCalibrator.fit_circle_3d(cam_pts_B, robust=True)

    print("\n--- Circle Fit Analysis (Unconstrained) ---")
    print(f"Sweep A: Center={np.round(c_fit_A, 2)}, Radius={r_fit_A:.2f} mm, RMSE={rmse_fit_A:.4f} mm")
    print(f"Sweep B: Center={np.round(c_fit_B, 2)}, Radius={r_fit_B:.2f} mm, RMSE={rmse_fit_B:.4f} mm")

    # If RMSE is low, circle is not distorted.
    if rmse_fit_A < 2.0 and rmse_fit_B < 2.0:
        print(">> Circles are clean and NOT distorted! Fit RMSE is very low.")
    else:
        print(">> WARNING: High fitting error. Circles might be distorted or noisy.")

    # Let's test the nominal axis projections
    # Under v1.2 wrist_pitch:
    # cand_joint = 5 (Wrist Pitch J5)
    # sweep_joint_A = 4 (Wrist Yaw J4)
    # sweep_joint_B = 6 (Wrist Roll J6)
    #
    # In torso frame:
    # a_cand_t5 = [0, 1, 0] (y-axis)
    # a_A_t5 = [0, 0, 1] (z-axis)
    # a_B_t5 = [0, 0, 1] (z-axis)
    # Note: For mock robot, they are hardcoded as above.
    
    a_cand_t5 = np.array([0.0, 1.0, 0.0])
    a_A_t5 = np.array([0.0, 0.0, 1.0])
    a_B_t5 = np.array([0.0, 0.0, 1.0])

    # Rotation matrix of euler [-90, 0, -90] (torso to camera)
    R_fixed = R_scipy.from_euler('ZYX', [-90.0, 0.0, -90.0], degrees=True).as_matrix()

    # Compare transpose vs no transpose:
    for name, R_rob_to_cam in [("R_fixed (No Transpose)", R_fixed), ("R_fixed.T (With Transpose)", R_fixed.T)]:
        print(f"\n--- Analysis using: {name} ---")
        
        a_cand_cam = R_rob_to_cam @ a_cand_t5
        a_A_cam = R_rob_to_cam @ a_A_t5
        a_B_cam_nom = R_rob_to_cam @ a_B_t5

        a_cand_cam /= np.linalg.norm(a_cand_cam)
        a_A_cam /= np.linalg.norm(a_A_cam)
        a_B_cam_nom /= np.linalg.norm(a_B_cam_nom)

        res_A = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_A, angles_A, axis_prior=a_A_cam)
        res_B = BaseCalibrator.fit_circle_3d_and_6dof_misalignment(poses_B, angles_B, axis_prior=a_B_cam_nom)

        n_A = res_A['axis_opt']
        n_B = res_B['axis_opt']
        if np.dot(n_A, n_B) < 0:
            n_B = -n_B

        angle_between_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))
        print(f"  Angle between normals: {angle_between_normals:.4f} deg")

        # Project nominal and actual axes onto plane perpendicular to candidate axis
        a_A_proj = a_A_cam - np.dot(a_A_cam, a_cand_cam) * a_cand_cam
        a_B_proj = a_B_cam_nom - np.dot(a_B_cam_nom, a_cand_cam) * a_cand_cam
        a_A_proj /= np.linalg.norm(a_A_proj)
        a_B_proj /= np.linalg.norm(a_B_proj)
        nominal_angle = np.arctan2(np.dot(np.cross(a_A_proj, a_B_proj), a_cand_cam), np.dot(a_A_proj, a_B_proj))

        n_A = n_A if np.dot(n_A, a_A_cam) > 0 else -n_A
        n_B = n_B if np.dot(n_B, a_B_cam_nom) > 0 else -n_B
        
        n_A_proj = n_A - np.dot(n_A, a_cand_cam) * a_cand_cam
        n_B_proj = n_B - np.dot(n_B, a_cand_cam) * a_cand_cam
        n_A_proj /= np.linalg.norm(n_A_proj)
        n_B_proj /= np.linalg.norm(n_B_proj)
        actual_angle = np.arctan2(np.dot(np.cross(n_A_proj, n_B_proj), a_cand_cam), np.dot(n_A_proj, n_B_proj))

        diff_angle = actual_angle - nominal_angle
        diff_angle = (diff_angle + np.pi) % (2 * np.pi) - np.pi
        diff_angle_deg = np.degrees(diff_angle)

        print(f"  diff_angle (deg): {diff_angle_deg:.4f}")
        
        # Old constant sign:
        sign_old = -1.0 if diff_angle > 0.0 else 1.0
        print(f"  Offset Direction Sign (Old code): {sign_old}")
        print(f"  Recommended Correction (Old code): {sign_old * abs(diff_angle_deg) * 0.95:+.4f} deg")

if __name__ == "__main__":
    main()
