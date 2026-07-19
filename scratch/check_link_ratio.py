import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt

sys.path.append("/home/jsm/camera_ws")
import rby1_sdk.dynamics as rd

# Standalone 3D circle fitting
def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    basis1 = Vt[0, :]
    basis2 = Vt[1, :]
    
    points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
    
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    center_3d = centroid + xc * basis1 + yc * basis2
    return center_3d, r

def main():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        print(f"URDF not found at {urdf_path}")
        return
        
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    joint_names = dyn_robot.get_joint_names()
    
    # We want to analyze:
    # Axis 4, 5, 6 sweeps for Right and Left arms
    result_dir = "/home/jsm/camera_ws/result/result_txt"
    
    arms = ["right", "left"]
    axes = [4, 5, 6]
    
    print("=========================================================================")
    print(" KINEMATIC SWEEP RADIUS VERIFICATION (URDF VS MEASURED)")
    print("=========================================================================")
    
    for arm in arms:
        print(f"\n--- {arm.upper()} ARM ANALYSIS ---")
        for axis in axes:
            filename = f"sweep_points_{arm}_marker_axis_{axis}.txt"
            fpath = os.path.join(result_dir, filename)
            if not os.path.exists(fpath):
                print(f"File {filename} not found.")
                continue
                
            angles = []
            cam_pts = []
            torso_pts = []
            ee_pts = []
            T_ee2marker_list = []
            
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip().startswith("#") or not line.strip():
                        continue
                    parts = [float(x.strip()) for x in line.strip().split(",")]
                    angles.append(parts[0])
                    cam_pts.append(parts[1:4])
                    torso_pts.append(parts[4:7])
                    ee_pts.append(parts[7:10]) # EE_X, EE_Y, EE_Z in mm
                    
                    # 16 elements of T_ee2marker
                    T_flat = parts[26:42] if len(parts) > 42 else parts[-16:]
                    T_ee2marker_list.append(np.array(T_flat).reshape(4, 4))
            
            cam_pts = np.array(cam_pts)
            torso_pts = np.array(torso_pts)
            ee_pts = np.array(ee_pts) # in mm
            
            # 1. Measured radius from circle fit
            _, r_meas = fit_circle_3d(cam_pts) # in mm (should match torso fit)
            _, r_meas_torso = fit_circle_3d(torso_pts)
            
            # 2. Compute Nominal Radius from URDF
            # Joint axis names:
            # right_arm_3 (J3: elbow), right_arm_4 (J4: wrist_yaw), right_arm_5 (J5: wrist_pitch), right_arm_6 (J6: wrist_roll/yaw2)
            # The nominal sweep radius of joint J is the perpendicular distance from joint axis J to the marker center.
            # Since the sweep file contains the marker's 3D position in the EE frame (ee_pts),
            # we can compute the marker position relative to joint axis frame J.
            
            # Let's define the link chain from joint axis frame J to the EE frame.
            # Using rby1_sdk, we can build a temporary state to compute the transform.
            ee_link = f"ee_{arm}" # End-effector link for v1.2 / v1.3
            
            nominal_radii = []
            for ang, ee_p in zip(angles, ee_pts):
                # Set up joint angles for the arm
                q_full = np.zeros(dyn_robot.get_dof())
                
                # Active arm indices
                arm_idx = [joint_names.index(f"{arm}_arm_{i}") for i in range(7)]
                
                # We assume standard ready pose except for the swept joint
                # The ready pose during calibration typically has J3=elbow_pitch, J4=wrist_yaw, J5=wrist_pitch, J6=wrist_roll
                # For v1.2: J4 = wrist_yaw, J5 = wrist_pitch, J6 = wrist_roll (wrist_yaw2)
                # Let's populate the swept joint angle
                if axis == 4:
                    q_full[arm_idx[4]] = np.radians(ang)
                elif axis == 5:
                    q_full[arm_idx[5]] = np.radians(ang)
                elif axis == 6:
                    q_full[arm_idx[6]] = np.radians(ang)
                
                # Create state
                joint_link_name = f"link_{arm}_arm_{axis-1}" # J4 parent is link_arm_3, J5 parent is link_arm_4, etc.
                state = dyn_robot.make_state(
                    [joint_link_name, ee_link],
                    joint_names
                )
                state.set_q(q_full)
                dyn_robot.compute_forward_kinematics(state)
                T_joint_to_ee = dyn_robot.compute_transformation(state, 0, 1)
                
                # Marker position in joint frame
                p_marker_ee = np.array([ee_p[0], ee_p[1], ee_p[2], 1.0]) / 1000.0 # to meters
                p_marker_joint = T_joint_to_ee @ p_marker_ee
                p_marker_joint_xyz = p_marker_joint[:3] * 1000.0 # to mm
                
                # Rotation axis of the joint (Z-axis for J4/J6, Y-axis for J5)
                # For Rby1: J4 (wrist_yaw) rotates around Z, J5 (wrist_pitch) rotates around Y, J6 (wrist_roll) rotates around Z
                # Let's compute perpendicular distance to the rotation axis:
                if axis in [4, 6]: # Z-axis rotation
                    r_nom = np.sqrt(p_marker_joint_xyz[0]**2 + p_marker_joint_xyz[1]**2)
                else: # J5: Y-axis rotation
                    r_nom = np.sqrt(p_marker_joint_xyz[0]**2 + p_marker_joint_xyz[2]**2)
                nominal_radii.append(r_nom)
                
            r_nom_mean = np.mean(nominal_radii)
            discrepancy = r_meas - r_nom_mean
            
            print(f"Axis {axis} sweep ({filename}):")
            print(f"  - Measured Radius (Cam)   : {r_meas:.2f} mm")
            print(f"  - Measured Radius (Torso) : {r_meas_torso:.2f} mm")
            print(f"  - Nominal Radius (URDF)   : {r_nom_mean:.2f} mm")
            print(f"  - Discrepancy (Meas-Nom)  : {discrepancy:+.2f} mm")
            
if __name__ == "__main__":
    main()
