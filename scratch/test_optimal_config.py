import sys
import os
import numpy as np
import rby1_sdk as rby

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer, make_transform

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
data = np.load(dataset_path)

q_arm_list = data["q_arm"]
T_meas_list = data["marker"]
q_head_list = np.zeros((q_arm_list.shape[0], 2)) # fixed camera

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    model = robot.model()
    arm_idx = np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]])
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # Bracket values from Step 1
    ee_to_marker = {
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0],
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0]
    }
    
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    
    # Step 1 joint offsets (anchors)
    joint_offsets = {
        "right": {"joint3": -2.7265, "joint5": -0.9283, "joint6": -0.2649},
        "left": {"joint3": -2.1863, "joint5": 0.0, "joint6": 0.0}
    }

    # We subclass to add moderate shoulder anchors (weight = 20.0)
    class OptimizedQPOptimizer(QPCalibrationOptimizer):
        def compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
            # Call parent
            res = super().compute_step(q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam)
            if res is None or res[0] is None:
                return res
            dx, total_err = res
            
            # Since compute_step solves the QP inside it, we need to modify H and g before solve_qp!
            # Let's inspect where solve_qp is called in compute_step.
            # It's better to modify H and g in-place if we override the internal variables or rewrite compute_step.
            return res

    # Actually, we can just modify calibration_optimizer.py to add a configurable shoulder anchor weight!
    # Let's see: if we set lambda_cam_pos to 0.0 and lambda_cam_rot to 0.0, what happens?
    
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=arm_idx,
        ee_links=ee_links,
        mount_to_cam_nom=[0.047, 0.009, 0.057, -90.0, 0.0, -90.0],
        head_base_to_cam_nom=head_base_to_cam_nom,
        ee_to_marker_nom=ee_to_marker,
        head_idx=[0, 1],
        eps=1e-7,
        lambda_cam_pos=0.0, # Let camera translation optimize freely!
        lambda_cam_rot=0.0, # Let camera rotation optimize freely!
        use_sag=False,
        optimize_head=False,
        optimize_camera=True,
        active_arms=["left", "right"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=joint_offsets
    )
    
    # We will run the three stages of optimization
    # Stage 1
    q_arm_offset, q_head_offset, xi_cam, _, _ = optimizer.optimize(
        q_arm_list, q_head_list, T_meas_list
    )
    
    # Stage 2 (Refinement)
    optimizer.optimize_camera = False
    q_arm_offset, q_head_offset, _, _, _ = optimizer.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_offset,
        q_head_offset_init=q_head_offset,
        xi_mount_cam_init=xi_cam
    )
    
    # Stage 3
    optimizer.optimize_camera = True
    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, head_base_to_cam_new = optimizer.optimize(
        q_arm_list, q_head_list, T_meas_list,
        q_arm_offset_init=q_arm_offset,
        q_head_offset_init=q_head_offset,
        xi_mount_cam_init=xi_cam
    )
    
    q_deg = np.degrees(q_arm_offset)
    print("\n--- RESULTS WITH FREE CAMERA OPTIMIZATION (lambda = 0.0) ---")
    print(f"Right Joint Offsets (deg): {q_deg[0:7]}")
    print(f"Left Joint Offsets (deg):  {q_deg[7:14]}")
    print(f"Mount to Cam New:          {mount_to_cam_new}")
    print(f"Head Base to Cam New:      {head_base_to_cam_new}")
    
finally:
    robot.disconnect()
