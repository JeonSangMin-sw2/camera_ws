import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import QPCalibrationOptimizer, se3_log

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    robot_config = rd.load_robot_from_urdf(urdf_path, "base")
    dyn_robot = rd.Robot(robot_config)
    
    active_joints = [
        'wheel_fr', 'wheel_fl', 'wheel_rr', 'wheel_rl',
        'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6',
        'head_0', 'head_1'
    ]
    
    class MockRobot:
        def __init__(self, dyn_robot, active_joints):
            self._dyn_robot = dyn_robot
            self._active_joints = active_joints
        def get_dynamics(self):
            return self._dyn_robot
        class MockModel:
            def __init__(self, active_joints):
                self.robot_joint_names = active_joints
            @property
            def right_arm_idx(self):
                return [10, 11, 12, 13, 14, 15, 16]
            @property
            def left_arm_idx(self):
                return [17, 18, 19, 20, 21, 22, 23]
            @property
            def head_idx(self):
                return [24, 25]
        def model(self):
            return self.MockModel(self._active_joints)
        class MockState:
            def __init__(self):
                self.position = np.zeros(26)
        def get_state(self):
            return self.MockState()
            
    mock_robot = MockRobot(dyn_robot, active_joints)
    
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260812_140518.npz"
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    q_head = data["q_head"]
    T_meas = data["marker"]
    
    active_arms = ["right", "left"]
    right_arm_idx = [10, 11, 12, 13, 14, 15, 16]
    left_arm_idx = [17, 18, 19, 20, 21, 22, 23]
    arm_idx = np.concatenate([right_arm_idx, left_arm_idx])
    head_idx = [24, 25]
    
    mount_to_cam_nom = [0.0495, -0.0115, 0.044, -90.0, 0.0, -90.0]
    ee_to_marker_nom = {
        "right": [0.0, -0.054, -0.048, 90.0, 0.0, 180.0],
        "left": [0.0, 0.054, -0.048, 90.0, 0.0, 0.0]
    }
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # Lock J0 to J6 offsets to the user's provided baseline values!
    baselines = {
        "right": [0.4886, 4.4061, 0.0013, 1.5709, -0.0048, 0.3999, -0.0040],
        "left": [1.6447, -4.1243, -0.0037, 1.9142, 0.0011, 2.6444, -0.0167]
    }
    
    D2R = np.radians(1.0)
    # The evaluation in prepare_q_full adds q_offset, but on the robot controller:
    # home_offset is subtracted from the encoder reading.
    # Wait, in the solver: T_model = T_nominal @ Exp(-q_offset) or similar?
    # No, in prepare_q_full: q_full = q_cmd + q_offset.
    # In RBY1 dynamics, positive joint rotation increases the joint angle.
    # So if the baseline joint offset is e.g. +4.4061 degrees, then q_offset is +4.4061 * D2R.
    # Let's set q_arm_offset_init directly to the baseline!
    q_arm_offset_init = np.concatenate([
        np.array(baselines["right"]) * D2R,
        np.array(baselines["left"]) * D2R
    ])
    
    # We lock arm optimization (optimize_arm = False) and only optimize head and camera!
    opt = QPCalibrationOptimizer(
        robot=mock_robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=mount_to_cam_nom,
        head_base_to_cam_nom=mount_to_cam_nom,
        ee_to_marker_nom=ee_to_marker_nom,
        head_idx=head_idx,
        camera_link="link_head_2",
        max_iter=500,
        optimize_arm=False, # Lock the arms to the baseline!
        optimize_head=True,
        optimize_camera=True,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1.0,
        active_arms=active_arms,
    )
    
    q_arm_offset, q_head_offset, xi_mount_cam, mount_to_cam_new, head_base_to_cam_new = opt.optimize(
        q_arm, q_head, T_meas,
        q_arm_offset_init=q_arm_offset_init
    )
    
    # Calculate residuals
    errs = []
    for q_arm_sample, q_head_sample, T_meas_pair in zip(q_arm, q_head, T_meas):
        for side_idx, arm_side in enumerate(["right", "left"]):
            T_meas_sample = T_meas_pair[side_idx]
            _, _, _, T_model = opt.evaluate_sample(
                q_arm_sample, q_head_sample, arm_side, q_arm_offset, q_head_offset, xi_mount_cam
            )
            T_err = np.linalg.inv(T_model) @ T_meas_sample
            xi = se3_log(T_err)
            errs.append(np.linalg.norm(xi[3:]) * 1000.0)
    mean_err = np.mean(errs)
    
    print("\n" + "="*60)
    print("RESULTS WITH ARM OFFSETS LOCKED TO USER'S BASELINE")
    print("="*60)
    print(f"Mean Marker Pos Error (Residual): {mean_err:.4f} mm")
    print(f"Head Pan (Yaw): {np.degrees(q_head_offset[0]):+.4f}°")
    print(f"Head Tilt (Pitch): {np.degrees(q_head_offset[1]):+.4f}°")
    print("-" * 60)
    print(f"mount_to_cam_new (x): {mount_to_cam_new[0]:.4f} m")
    print(f"mount_to_cam_new (y): {mount_to_cam_new[1]:.4f} m")
    print(f"mount_to_cam_new (z): {mount_to_cam_new[2]:.4f} m")
    print(f"mount_to_cam_new (R): {mount_to_cam_new[3]:.4f}°")
    print(f"mount_to_cam_new (P): {mount_to_cam_new[4]:.4f}°")
    print(f"mount_to_cam_new (Y): {mount_to_cam_new[5]:.4f}°")
    print("="*60)

if __name__ == "__main__":
    main()
