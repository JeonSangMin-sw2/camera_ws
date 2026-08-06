import os
import sys
import numpy as np
import yaml

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import make_transform

def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
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
    return centroid + xc * basis1 + yc * basis2, r

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        print("URDF v1.2 not found.")
        return
        
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    joint_names = dyn_robot.get_joint_names()
    
    with open("/home/rainbow/camera_ws/config/setting.yaml", "r") as f:
        setting = yaml.safe_load(f)
    camera_cfg = setting.get("camera", {})
    
    ee_to_marker_nom_v12 = {
        "left": camera_cfg.get("Tf_to_marker_left_v12", [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]),
        "right": camera_cfg.get("Tf_to_marker_right_v12", [0.0, -0.054, -0.048, 90.0, 0.0, 180.0])
    }
    
    with open("/home/rainbow/camera_ws/config/ready_poses.yaml", "r") as f:
        ready_poses_cfg = yaml.safe_load(f)
    v12_ready = ready_poses_cfg.get("v1.2", {}).get("joint", {})
    
    result_dir = "/home/rainbow/camera_ws/result/result_step2"
    
    print("=========================================================================")
    print(" KINEMATIC SWEEP RADIUS VERIFICATION USING CAD NOMINALS (v1.2)")
    print("=========================================================================")
    
    # Let's find txt files under result_step2 or look at the current files
    # The txt files in the run are: sweep_points_right_marker_axis_4.txt etc.
    # Let's check where they are located. The user log says:
    # [DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
    # Usually they are saved in Cwd (workspace root) or result/result_txt. Let's see if they are in workspace root.
    
    for arm in ["right", "left"]:
        print(f"\n--- {arm.upper()} ARM ---")
        marker_nom = ee_to_marker_nom_v12[arm]
        T_ee_to_marker = make_transform(marker_nom)
        
        for axis in [4, 5, 6]:
            filename = f"sweep_points_{arm}_marker_axis_{axis}.txt"
            fpaths = [
                os.path.join("/home/rainbow/camera_ws", filename),
                os.path.join("/home/rainbow/camera_ws/result", filename),
                os.path.join("/home/rainbow/camera_ws/result/result_txt", filename),
                os.path.join("/home/rainbow/camera_ws/result/result_step2", filename)
            ]
            fpath = None
            for fp in fpaths:
                if os.path.exists(fp):
                    fpath = fp
                    break
            if not fpath:
                print(f"Axis {axis} sweep: {filename} not found.")
                continue
                
            angles = []
            cam_pts = []
            
            with open(fpath, "r") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str.startswith("#") or not line_str or "=" in line_str:
                        continue
                    try:
                        parts = [float(x.strip()) for x in line_str.split(",")]
                        angles.append(parts[0])
                        cam_pts.append(parts[1:4]) # Cam_X, Cam_Y, Cam_Z in mm
                    except ValueError:
                        continue
            
            cam_pts = np.array(cam_pts)
            center, r_meas = fit_circle_3d(cam_pts)
            
            if axis == 4:
                mode_key = "elbow"
            elif axis == 5:
                mode_key = "wrist_pitch"
            else:
                mode_key = "wrist_yaw2"
            
            ready_deg = v12_ready.get(mode_key, {}).get(f"{arm}_arm")
            if ready_deg is None:
                ready_deg = [-55.0, -45.0, 25.0, -127.0, 90.0, 0.0, 0.0]
            
            q_ready = np.radians(ready_deg)
            
            nominal_radii = []
            for ang in angles:
                q_full = np.zeros(dyn_robot.get_dof())
                arm_idx = [joint_names.index(f"{arm}_arm_{i}") for i in range(7)]
                
                for idx, val in zip(arm_idx, q_ready):
                    q_full[idx] = val
                    
                q_full[arm_idx[axis]] = q_ready[axis] + np.radians(ang)
                
                joint_link_name = f"link_{arm}_arm_{axis}"
                ee_link = f"ee_{arm}"
                
                state = dyn_robot.make_state(
                    [joint_link_name, ee_link],
                    joint_names
                )
                state.set_q(q_full)
                dyn_robot.compute_forward_kinematics(state)
                T_joint_to_ee = dyn_robot.compute_transformation(state, 0, 1)
                
                p_marker_joint = T_joint_to_ee @ T_ee_to_marker
                p_marker_joint_xyz = p_marker_joint[:3, 3] * 1000.0
                
                if axis in [4, 6]:
                    r_nom = np.sqrt(p_marker_joint_xyz[0]**2 + p_marker_joint_xyz[1]**2)
                else:
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
