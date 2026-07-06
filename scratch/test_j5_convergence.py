import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append("/home/rainbow/camera_ws")
import rby1_sdk as rby
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.CalibratorBase import BaseCalibrator

def simulate_sweep(arm_side, gt_offsets, staged_offsets):
    robot = rby.create_robot("127.0.0.1:50051", "m")
    if not robot.connect():
        print("Failed to connect")
        return
        
    calib = JointCalibrator(marker_st=None, robot=robot)
    calib.robot_version = "1.3"
    calib.camera_config = {
        "Tf_to_marker_left": [0.095, 0.0, -0.008, 89.7, 1.5, -90.0],
        "mount_to_cam": [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]
    }
    calib.joint_offsets = staged_offsets
    
    # Let's get ready pose for wrist_pitch_v13
    is_v13 = True
    mode = "wrist_pitch_v13"
    q_ready_arm = calib.get_ready_pose("v1.3", "joint", "wrist_pitch", arm_side)
    
    T_ee_to_marker = calib.make_transform([0.095, 0.0, -0.008, 89.7, 1.5, -90.0])
    
    model = robot.model()
    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
    
    q_ready_base = np.zeros(len(robot.get_state().position))
    q_ready_base[arm_idx] = q_ready_arm
    
    # Generate mock sweep data
    jcfg = calib.JOINT_CONFIGS[mode]
    cand_joint = jcfg["cand_joint"]
    sweep_joint_A = jcfg["sweep_joint_A"]
    sweep_joint_B = jcfg["sweep_joint_B"]
    
    # Let's generate dataset A and dataset B
    # To do this, we need the SimulatedMarkerTransform logic
    # SimulatedMarkerTransform computes T_cam_to_marker for a commanded q
    # We will simulate SimulatedMarkerTransform with and without the AttributeError bug
    
    def get_sim_marker_pose(q, with_bug):
        q_actual = np.array(q)
        j6_gt = gt_offsets[arm_side]["joint6"]
        j5_gt = gt_offsets[arm_side]["joint5_v13"]
        j3_gt = gt_offsets[arm_side]["joint3"]
        
        j6_staged = staged_offsets[arm_side].get("wrist_roll", 0.0)
        j5_staged = staged_offsets[arm_side].get("wrist_pitch", 0.0)
        j3_staged = staged_offsets[arm_side].get("elbow", 0.0)
        
        # If with_bug is True, we simulate what the OLD code did assuming offsets was loaded (e.g. j5_staged is added)
        # Wait, if with_bug is False, but we still have AttributeError, then offsets is {} and j5_staged is 0.0.
        # Let's see what happens in both cases!
        if with_bug:
            # Staged offsets are added in simulator
            q_actual[arm_idx[6]] += np.radians(j6_gt + j6_staged)
            q_actual[arm_idx[5]] += np.radians(j5_gt + j5_staged)
            q_actual[arm_idx[3]] += np.radians(j3_gt + j3_staged)
        else:
            # Staged offsets are NOT added in simulator (AttributeError behavior)
            q_actual[arm_idx[6]] += np.radians(j6_gt)
            q_actual[arm_idx[5]] += np.radians(j5_gt)
            q_actual[arm_idx[3]] += np.radians(j3_gt)
            
        dyn_model = robot.get_dynamics()
        ee_name = f"ee_{arm_side}"
        T_t5_to_ee = BaseCalibrator.compute_fk(robot, dyn_model, q_actual, ee_name, "link_torso_5")
        T_t5_to_head = BaseCalibrator.compute_fk(robot, dyn_model, q, "link_head_2", "link_torso_5")
        
        # Camera transform
        T_mount_to_cam = calib.make_transform([0.047, 0.009, 0.057, -90.0, 0.0, -90.0])
        
        T_t5_to_cam = T_t5_to_head @ T_mount_to_cam
        T_torso_to_cam = np.linalg.inv(T_t5_to_cam)
        
        # Simulated ground-truth bracket offset
        T_bracket_offset = np.eye(4)
        T_bracket_offset[:3, 3] = gt_offsets[arm_side]["bracket_pos"]
        T_bracket_offset[:3, :3] = R_scipy.from_euler('ZYX', [0.0, 1.5, -0.3], degrees=True).as_matrix()
        
        T_cam_to_marker = T_torso_to_cam @ T_t5_to_ee @ T_bracket_offset @ T_ee_to_marker
        return T_cam_to_marker

    # Generate sweep datasets
    def generate_dataset(sweep_joint, sweep_range, j5_val, with_bug=True):
        dataset = []
        nominal_val = q_ready_arm[sweep_joint]
        for deg in np.linspace(-sweep_range, sweep_range, 5):
            q_cmd = np.array(q_ready_base)
            q_cmd[arm_idx[5]] += np.radians(j5_val)
            q_cmd[arm_idx[6]] += np.radians(staged_offsets[arm_side].get("wrist_roll", 0.0))
            q_cmd[arm_idx[3]] += np.radians(staged_offsets[arm_side].get("elbow", 0.0))
            
            q_cmd[arm_idx[sweep_joint]] += np.radians(deg)
            T_marker = get_sim_marker_pose(q_cmd, with_bug=with_bug)
            dataset.append((q_cmd, T_marker))
        return dataset

    # We want to test J5 calibration convergence for LEFT arm (gt_j5 = +3.60)
    # Staged offset starts at 0.0
    print(f"\n--- Simulating Left Arm J5 Sweep (with staged in simulator bug) ---")
    staged = 0.0
    for iter_idx in range(1, 5):
        staged_offsets[arm_side]["wrist_pitch"] = staged
        calib.joint_offsets = staged_offsets
        dataset_A = generate_dataset(sweep_joint_A, 20.0, staged, with_bug=True)
        dataset_B = generate_dataset(sweep_joint_B, 10.0, staged, with_bug=True)
        
        res = calib.compute_calibration_results(
            arm_side=arm_side,
            mode=mode,
            dataset_A=dataset_A,
            dataset_B=dataset_B,
            initial_joint_pos=q_ready_arm,
            current_offset_deg=staged,
            use_angle_based_fitting=True,
            save_debug=False
        )
        recommended = res["recommended_joint_offset"]
        print(f"Iteration {iter_idx}: staged={staged:.4f}°, step={recommended:.4f}°")
        staged += recommended * 0.9

    # Now let's test WITHOUT the bug (AttributeError, i.e. staged in simulator is 0.0)
    print(f"\n--- Simulating Left Arm J5 Sweep (WITHOUT staged in simulator, i.e. offsets is empty) ---")
    staged = 0.0
    for iter_idx in range(1, 5):
        staged_offsets[arm_side]["wrist_pitch"] = staged
        calib.joint_offsets = staged_offsets
        dataset_A = generate_dataset(sweep_joint_A, 20.0, staged, with_bug=False)
        dataset_B = generate_dataset(sweep_joint_B, 10.0, staged, with_bug=False)
        
        res = calib.compute_calibration_results(
            arm_side=arm_side,
            mode=mode,
            dataset_A=dataset_A,
            dataset_B=dataset_B,
            initial_joint_pos=q_ready_arm,
            current_offset_deg=staged,
            use_angle_based_fitting=True,
            save_debug=False
        )
        recommended = res["recommended_joint_offset"]
        print(f"Iteration {iter_idx}: staged={staged:.4f}°, step={recommended:.4f}°")
        staged += recommended * 0.9

if __name__ == "__main__":
    gt_offsets = {
        "left": {
            "joint6": 3.5,
            "joint5_v13": 3.6,
            "joint3": 0.7,
            "bracket_pos": [-0.002, 0.00, -0.003],
            "bracket_rpy": [-0.3, 1.5, 0.0]
        }
    }
    staged_offsets = {
        "left": {
            "wrist_roll": -3.2125,
            "wrist_pitch": 0.0,
            "elbow": 0.0
        }
    }
    simulate_sweep("left", gt_offsets, staged_offsets)
