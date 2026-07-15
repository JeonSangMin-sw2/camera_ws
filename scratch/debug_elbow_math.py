import sys
import os
import numpy as np
sys.path.append("/home/rainbow/camera_ws")
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
import rby1_sdk.dynamics as rd

def load_sweep_data(path):
    angles = []
    poses = []
    cam_pts = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(x.strip()) for x in line.strip().split(",")]
            angles.append(parts[0])
            cam_pts.append(parts[1:4])
            T = np.array(parts[10:26]).reshape(4, 4)
            poses.append(T)
    return np.array(angles), poses, np.array(cam_pts)

def main():
    dir_path = "/home/rainbow/camera_ws/result/result_REAL/result_txt"
    a_path = os.path.join(dir_path, "sweep_points_right_joint_A_axis_2.txt")
    b_path = os.path.join(dir_path, "sweep_points_right_joint_B_axis_4.txt")
    
    angles_A, poses_A, _ = load_sweep_data(a_path)
    angles_B, poses_B, _ = load_sweep_data(b_path)
    
    print(f"Loaded A: {len(poses_A)}, B: {len(poses_B)}")
    
    # Run fit_circle_3d_and_6dof_misalignment
    # For this, we need axis_prior. Let's just use fit_circle_3d to get n_A and n_B directly.
    _, R_A, _, _, _, _, _ = BaseCalibrator.fit_circle_3d([p[:3,3]*1000 for p in poses_A], robust=False)
    _, R_B, _, _, _, _, _ = BaseCalibrator.fit_circle_3d([p[:3,3]*1000 for p in poses_B], robust=False)
    
    n_A = R_A[:, 2]
    n_B = R_B[:, 2]
    print(f"Raw n_A: {n_A}")
    print(f"Raw n_B: {n_B}")
    
    # We need a_cand_cam. To get a_cand_cam exactly, we need FK.
    # But wait, the sweep_points file has T_torso2marker_flat(16) etc? No, it has T_t5_to_marker.
    # It doesn't have the torso FK directly, but we can load the robot and compute FK.
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    names = dyn_robot.get_joint_names()
    right_arm_idx = [names.index(f"right_arm_{i}") for i in range(7)]
    
    # Initial joint pos?
    # We can guess initial joint pos is 0 except for offsets. But we don't know the exact ready pose used in result_REAL.
    # Wait, the nominal_angle only depends on the ready pose if the ready pose changes the relative orientation of the axes.
    # J3 (Elbow) revolves around Y.
    # J2 (Sweep A) revolves around Z.
    # J4 (Sweep B) revolves around Z.
    
    # In URDF, J2 and J4 axes are always parallel to each other IF J3 is at 0!
    # Because J3 rotates around Y, if J3 is non-zero, J4's Z-axis is rotated relative to J2's Z-axis by exactly the J3 angle!
    # So the true angle between J2 axis and J4 axis is exactly the J3 angle.
    
    # Let's check n_A and n_B angle!
    angle_normals = np.degrees(np.arccos(np.clip(np.dot(n_A, n_B), -1.0, 1.0)))
    print(f"Angle between n_A and n_B: {angle_normals:.4f} degrees")

if __name__ == "__main__":
    main()
