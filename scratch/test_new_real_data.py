import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
from core.calibration.MarkerCalibrator import MarkerCalibrator

def load_sweep_data(filepath):
    angles = []
    poses = []
    torso_pts = []
    q_fulls = []
    with open(filepath, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("#") or not line_str or "==" in line_str:
                continue
            parts = [float(p.strip()) for p in line_str.split(",")]
            angles.append(parts[0])
            torso_pts.append(parts[4:7])
            # T_cam2marker_flat(16)
            T_flat = parts[10:26]
            T = np.array(T_flat).reshape(4, 4)
            poses.append(T)
            
            # Since q_full is 20-dim, let's build q_full from J4-J6 angles.
            # Look at sweep data columns: parts[0] is joint angle
            # For axis 4: Joint_4_Angle(deg)
            # For axis 5: Joint_5_Angle(deg)
            # For axis 6: Joint_6_Angle(deg)
            # Let's check what a typical q_full would contain.
            # Normally we have Joint 4 at index 4, Joint 5 at index 5, Joint 6 at index 6 in radians.
            # Let's populate index 4, 5, 6 depending on which axis we are sweeping,
            # or we can inspect if there's any active joint info.
            q_full = np.zeros(20)
            if "axis_4" in filepath:
                q_full[4] = np.radians(parts[0])
            elif "axis_5" in filepath:
                q_full[5] = np.radians(parts[0])
            elif "axis_6" in filepath:
                q_full[6] = np.radians(parts[0])
            q_fulls.append(q_full)
            
    return np.array(angles), np.array(poses), np.array(torso_pts), q_fulls

def main():
    base_dir = "/home/rainbow/camera_ws/result/result_txt"
    
    print("Loading raw files...")
    # Loading actual files
    angles_4, poses_4, torso_4, q_full_4 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_4.txt"))
    angles_5, poses_5, torso_5, q_full_5 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_5.txt"))
    angles_6, poses_6, torso_6, q_full_6 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_6.txt"))
    
    print(f"Loaded points count: Axis 4: {len(angles_4)}, Axis 5: {len(angles_5)}, Axis 6: {len(angles_6)}")
    
    # We will instantiate MarkerCalibrator
    import rby1_sdk.dynamics as rd
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    
    class MockRobot:
        def __init__(self, dyn_robot):
            self._dyn_robot = dyn_robot
        def get_dynamics(self):
            return self._dyn_robot
        class MockModel:
            @property
            def robot_joint_names(self):
                return [
                    'wheel_fr', 'wheel_fl', 'wheel_rr', 'wheel_rl',
                    'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
                    'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
                    'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
                    'head_0', 'head_1'
                ]
            @property
            def right_arm_idx(self):
                return [10, 11, 12, 13, 14, 15, 16]
            @property
            def left_arm_idx(self):
                return [17, 18, 19, 20, 21, 22, 23]
        def model(self):
            return self.MockModel()
        class MockState:
            def __init__(self):
                self.position = np.zeros(26)
        def get_state(self):
            return self.MockState()

    mock_robot_instance = MockRobot(dyn_robot)
    calibrator = MarkerCalibrator(marker_st=None, robot=mock_robot_instance)
    calibrator.get_robot_version = lambda: "1.3"
    # Use a camera_config with nominal values from setting.yaml (or default)
    # The setting.yaml tf_vec is [0.096, 0.00000, -0.005, 90.0, 0.0, -90.0]
    calibrator.camera_config = {
        "Tf_to_marker_right_v13": [0.096, 0.00000, -0.005, 90.0, 0.0, -90.0]
    }
    
    # Mock get_link_length to return a nominal length or let's see what is used in real calibrator
    # We saw in validate_calibration.py that L_5_ee = 233.5 or something.
    # What does the real get_link_length return?
    # Let's print the actual link length computed in the real system.
    # Let's define a mock compute_fk or use robot.
    # Let's first run fit_circle_3d_and_6dof_misalignment on each pose list
    res_4 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_4, angles_4, axis_prior=[0.0, 0.0, 1.0])
    res_5 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_5, angles_5, axis_prior=[0.0, 1.0, 0.0])
    res_6 = calibrator.fit_circle_3d_and_6dof_misalignment(poses_6, angles_6, axis_prior=[1.0, 0.0, 0.0])
    
    r4 = res_4['radius']
    r5 = res_5['radius']
    r6 = res_6['radius']
    print(f"Computed circle radii (in mm): R4={r4:.4f}, R5={r5:.4f}, R6={r6:.4f}")
    
    # Pack data
    marker_data_6 = {'captured_poses': poses_6, 'captured_q_full': q_full_6, 'radius': r6, 'rmse': res_6['rmse']}
    marker_data_5 = {'captured_poses': poses_5, 'captured_q_full': q_full_5, 'radius': r5, 'rmse': res_5['rmse']}
    marker_data_4 = {'captured_poses': poses_4, 'captured_q_full': q_full_4, 'radius': r4, 'rmse': res_4['rmse']}
    
    # We will test compute_unified_bracket_calibration_v1_3 with L_5_ee.
    # Let's mock robot kinematic functions to match the robot representation in the system.
    # But since robot="mock_robot", the robot kinematics branch won't be executed (or it will fall back).
    # Wait, let's see what happens if we call calibrator.compute_unified_bracket_calibration
    res = calibrator.compute_unified_bracket_calibration(marker_data_5, marker_data_6, "right", marker_data_4=marker_data_4)
    print("\n--- Calibration Results with default robot='mock_robot' (Gram-Schmidt fallback) ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
