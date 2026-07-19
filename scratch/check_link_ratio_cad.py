import os
import sys
import numpy as np
import yaml

sys.path.append("/home/jsm/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import make_transform

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
    return center_3d, r

def main():
    # Load URDF v1.2
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        print("URDF v1.2 not found.")
        return
        
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    joint_names = dyn_robot.get_joint_names()
    
    # Load nominal camera/marker settings from setting.yaml
    with open("/home/jsm/camera_ws/config/setting.yaml", "r") as f:
        setting = yaml.safe_load(f)
    camera_cfg = setting.get("camera", {})
    
    # Nominal Tf_to_marker for v1.2
    ee_to_marker_nom_v12 = {
        "left": camera_cfg.get("Tf_to_marker_left_v12", [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]),
        "right": camera_cfg.get("Tf_to_marker_right_v12", [0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
    }
    
    # Load ready poses for v1.2
    with open("/home/jsm/camera_ws/config/ready_poses.yaml", "r") as f:
        ready_poses_cfg = yaml.safe_load(f)
    v12_ready = ready_poses_cfg.get("v1.2", {}).get("joint", {})
    
    result_dir = "/home/jsm/camera_ws/result/result_txt"
    
    print("=========================================================================")
    print(" KINEMATIC SWEEP RADIUS VERIFICATION USING CAD NOMINALS (v1.2)")
    print("=========================================================================")
    
    for arm in ["right", "left"]:
        print(f"\n--- {arm.upper()} ARM ---")
        
        # Nominal Tf_to_marker flat translation in ee_frame
        marker_nom = ee_to_marker_nom_v12[arm]
        T_ee_to_marker = make_transform(marker_nom)
        
        for axis in [4, 5, 6]:
            filename = f"sweep_points_{arm}_marker_axis_{axis}.txt"
            fpath = os.path.join(result_dir, filename)
            if not os.path.exists(fpath):
                continue
                
            angles = []
            cam_pts = []
            
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip().startswith("#") or not line.strip():
                        continue
                    parts = [float(x.strip()) for x in line.strip().split(",")]
                    angles.append(parts[0])
                    cam_pts.append(parts[1:4]) # Cam_X, Cam_Y, Cam_Z in mm
            
            cam_pts = np.array(cam_pts)
            
            # 1. Measured radius from circle fit
            # Fit circle in 3D to get the physical radius of the marker sweep in space
            centroid = np.mean(cam_pts, axis=0)
            centered = cam_pts - centroid
            U, S, Vt = np.linalg.svd(centered)
            basis1, basis2 = Vt[0, :], Vt[1, :]
            points_2d = np.column_stack((np.dot(centered, basis1), np.dot(centered, basis2)))
            A = np.column_stack((points_2d[:, 0], points_2d[:, 1], np.ones_like(points_2d[:, 0])))
            b = points_2d[:, 0]**2 + points_2d[:, 1]**2
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            r_meas = np.sqrt(c[2] + (c[0]/2)**2 + (c[1]/2)**2)
            
            # 2. Get the ready pose for this sweep
            # For Axis 4, the ready pose key is 'elbow'
            # For Axis 5, the ready pose key is 'wrist_pitch'
            # For Axis 6, the ready pose key is 'wrist_yaw2'
            if axis == 4:
                mode_key = "elbow"
            elif axis == 5:
                mode_key = "wrist_pitch"
            else:
                mode_key = "wrist_yaw2"
            
            ready_deg = v12_ready.get(mode_key, {}).get(f"{arm}_arm")
            if ready_deg is None:
                ready_deg = [-55.0, -45.0, 25.0, -127.0, 90.0, 0.0, 0.0] # Default fallback
            
            q_ready = np.radians(ready_deg)
            
            # Compute Nominal Radius
            nominal_radii = []
            for ang in angles:
                q_full = np.zeros(dyn_robot.get_dof())
                arm_idx = [joint_names.index(f"{arm}_arm_{i}") for i in range(7)]
                
                # Set ready pose
                for idx, val in zip(arm_idx, q_ready):
                    q_full[idx] = val
                    
                # Set active joint angle (adding relative sweep angle to ready pose baseline)
                q_full[arm_idx[axis]] = q_ready[axis] + np.radians(ang)
                
                # Link frames (using child link frame to align rotation axis with origin)
                joint_link_name = f"link_{arm}_arm_{axis}"
                ee_link = f"ee_{arm}"
                
                state = dyn_robot.make_state(
                    [joint_link_name, ee_link],
                    joint_names
                )
                state.set_q(q_full)
                dyn_robot.compute_forward_kinematics(state)
                T_joint_to_ee = dyn_robot.compute_transformation(state, 0, 1)
                
                # Marker position in J_frame
                p_marker_joint = T_joint_to_ee @ T_ee_to_marker
                p_marker_joint_xyz = p_marker_joint[:3, 3] * 1000.0 # to mm
                
                # Distance to axis of rotation
                if axis in [4, 6]: # Z axis
                    r_nom = np.sqrt(p_marker_joint_xyz[0]**2 + p_marker_joint_xyz[1]**2)
                else: # Y axis
                    r_nom = np.sqrt(p_marker_joint_xyz[0]**2 + p_marker_joint_xyz[2]**2)
                nominal_radii.append(r_nom)
                
            r_nom_mean = np.mean(nominal_radii)
            discrepancy = r_meas - r_nom_mean
            
            print(f"Axis {axis} sweep ({filename}):")
            print(f"  - Measured Radius (Cam)   : {r_meas:.2f} mm")
            print(f"  - Nominal Radius (CAD v12): {r_nom_mean:.2f} mm")
            print(f"  - Discrepancy (Meas-Nom)  : {discrepancy:+.2f} mm")

if __name__ == "__main__":
    main()
