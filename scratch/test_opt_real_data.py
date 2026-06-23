import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "calibration")))
from MarkerCalibrator import MarkerCalibrator

def load_sweep_data(filepath):
    angles = []
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [float(p.strip()) for p in line.strip().split(",")]
            angles.append(parts[0])
            T_flat = parts[10:26]
            T = np.array(T_flat).reshape(4, 4)
            poses.append(T)
    return np.array(angles), np.array(poses)

def fit_circle_3d(points):
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    ex = vh[0, :]
    ey = vh[1, :]
    pts_2d_in = np.dot(pts_centered, np.vstack((ex, ey)).T)
    A = np.c_[2 * pts_2d_in[:, 0], 2 * pts_2d_in[:, 1], np.ones(len(pts_2d_in))]
    b = pts_2d_in[:, 0]**2 + pts_2d_in[:, 1]**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc, vc = res[0], res[1]
    radius = np.sqrt(max(0.001, res[2] + uc**2 + vc**2))
    return radius

def main():
    base_dir = "/home/rainbow/camera_ws/core/calibration"
    
    angles_4, poses_4 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_4.txt"))
    angles_5, poses_5 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_5.txt"))
    angles_6, poses_6 = load_sweep_data(os.path.join(base_dir, "sweep_points_right_marker_axis_6.txt"))
    
    pts_4 = np.array([T[:3, 3] * 1000.0 for T in poses_4])
    pts_5 = np.array([T[:3, 3] * 1000.0 for T in poses_5])
    pts_6 = np.array([T[:3, 3] * 1000.0 for T in poses_6])
    
    r4 = fit_circle_3d(pts_4)
    r5 = fit_circle_3d(pts_5)
    r6 = fit_circle_3d(pts_6)
    
    q_fulls_4 = []
    for a in angles_4:
        q = np.zeros(20)
        q[4] = np.radians(a)
        q_fulls_4.append(q)
        
    q_fulls_5 = []
    for a in angles_5:
        q = np.zeros(20)
        q[5] = np.radians(a)
        q_fulls_5.append(q)
        
    q_fulls_6 = []
    for a in angles_6:
        q = np.zeros(20)
        q[6] = np.radians(a)
        q_fulls_6.append(q)
        
    # Setup calibrator and mock its properties
    calibrator = MarkerCalibrator(marker_st=None, robot=None)
    calibrator.get_robot_version = lambda: 1.3
    # Use actual L_5_ee solved by test_new_data
    calibrator.get_link_length = lambda arm_side: 253.5
    calibrator.camera_config = {
        "mount_to_cam": [0.0, 0.0, 0.0, -90.0, 0.0, -90.0],
        "Tf_to_marker_right_v13": [0.095, 0.0, -0.005, 90.0, 0.0, -90.0]
    }
    
    def mock_compute_fk(robot, dyn_model, q, ee_link, base_link="link_torso_5"):
        q4, q5, q6 = q[4], q[5], q[6]
        R4 = R_scipy.from_euler('Z', q4).as_matrix()
        R5 = R_scipy.from_euler('Y', q5).as_matrix()
        R6 = R_scipy.from_euler('X', q6).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R4 @ R5 @ R6
        T[:3, 3] = R4 @ R5 @ [0.0, 0.0, 0.2535] # lock to 253.5mm
        return T

    calibrator.compute_fk = mock_compute_fk
    
    class MockRobot:
        def get_dynamics(self):
            return None
    calibrator.robot = MockRobot()

    marker_data_4 = {'captured_poses': poses_4, 'captured_q_full': q_fulls_4, 'radius': r4}
    marker_data_5 = {'captured_poses': poses_5, 'captured_q_full': q_fulls_5, 'radius': r5}
    marker_data_6 = {'captured_poses': poses_6, 'captured_q_full': q_fulls_6, 'radius': r6}
    
    res = calibrator.compute_unified_bracket_calibration_v1_3(
        marker_data_5=marker_data_5,
        marker_data_6=marker_data_6,
        arm_side="right",
        marker_data_4=marker_data_4
    )
    
    print("\nCalibration Results on Real Right Arm Data:")
    print(f"  xe: {res['x_e']:.3f} mm")
    print(f"  ye: {res['y_e']:.3f} mm")
    print(f"  ze: {res['z_e']:.3f} mm")
    print(f"  roll:  {res['roll_e']:.3f} deg")
    print(f"  pitch: {res['pitch_e']:.3f} deg")
    print(f"  yaw:   {res['yaw_e']:.3f} deg")
    print(f"  opt_delta_5 (Joint 5 offset): {res['opt_delta_5']:.4f} deg")
    print(f"  opt_delta_6 (Joint 6 offset): {res['opt_delta_6']:.4f} deg")
    print(f"  ortho_err: {res['ortho_err']:.4f} deg")

if __name__ == "__main__":
    main()
