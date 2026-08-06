import numpy as np
import os
import sys

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk.dynamics as rd
from core.calibration_optimizer import QPCalibrationOptimizer, make_transform, se3_log

def main():
    urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.0.urdf"
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
                
        def model(self):
            return self.MockModel(self._active_joints)
            
        class MockState:
            def __init__(self):
                self.position = np.zeros(26)
                
        def get_state(self):
            return self.MockState()
            
    mock_robot = MockRobot(dyn_robot, active_joints)
    
    dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    T_meas_orig = data["marker"]
    
    active_arms = ["right", "left"]
    right_arm_idx = [10, 11, 12, 13, 14, 15, 16]
    left_arm_idx = [17, 18, 19, 20, 21, 22, 23]
    arm_idx = np.concatenate([right_arm_idx, left_arm_idx])
    
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    ee_to_marker_nom = {
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0],
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0]
    }
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # We will test scale factors from 1.0 to 1.10
    scale_factors = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10]
    for scale in scale_factors:
        # Scale the translation of T_meas
        T_meas_scaled = T_meas_orig.copy()
        T_meas_scaled[:, :, :3, 3] *= scale
        
        opt = QPCalibrationOptimizer(
            robot=mock_robot,
            arm_idx=arm_idx,
            ee_links=ee_links,
            mount_to_cam_nom=head_base_to_cam_nom,
            head_base_to_cam_nom=head_base_to_cam_nom,
            ee_to_marker_nom=ee_to_marker_nom,
            head_idx=None,
            camera_link="link_head_0",
            max_iter=500,
            lambda_cam_pos=0.0,
            lambda_cam_rot=0.0,
        )
        opt.optimize_arm = True
        opt.optimize_camera = True
        
        q_arm_offset, q_head_offset, xi_mount_cam, mount_to_cam_new, head_base_to_cam_new = opt.optimize(q_arm, None, T_meas_scaled)
        
        r_offsets = np.degrees(q_arm_offset[:7])
        l_offsets = np.degrees(q_arm_offset[7:])
        
        print(f"\nScale Factor: {scale:.2f}")
        print(f"  Right Joint Offsets (J0, J1, J3): {r_offsets[0]:.2f}, {r_offsets[1]:.2f}, {r_offsets[3]:.2f} deg")
        print(f"  Left Joint Offsets  (J0, J1, J3): {l_offsets[0]:.2f}, {l_offsets[1]:.2f}, {l_offsets[3]:.2f} deg")
        print(f"  Camera Extrinsics Absolute: {[round(x, 4) for x in mount_to_cam_new]}")

if __name__ == "__main__":
    main()
