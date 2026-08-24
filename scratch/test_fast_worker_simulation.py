import sys
import os
import time
import threading
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator
from main_ui import FullAutoWorker

class FastCalibratorMixin:
    # Speed up sleep in mock sweeps during testing
    pass

def run_simulation_validation(robot_ver="1.3", arm_side="right"):
    print(f"\n=======================================================")
    print(f"   FAST SIMULATION VALIDATION: v{robot_ver} ({arm_side.upper()} ARM)")
    print(f"=======================================================")
    
    jc = JointCalibrator(None, None)
    mc = MarkerCalibrator(None, None)
    jc.robot_version = robot_ver
    mc.robot_version = robot_ver
    
    # Check ground truth
    mock_gt = jc.MOCK_GT_OFFSETS[arm_side]
    gt_j6 = mock_gt["joint6"]
    gt_j5 = mock_gt["joint5_v13" if robot_ver == "1.3" else "joint5_v12"]
    gt_j3 = mock_gt["joint3"]
    gt_pos = [x * 1000.0 for x in mock_gt["bracket_pos"]]
    gt_rpy = mock_gt["bracket_rpy"]
    
    print(f"[GROUND TRUTH]")
    print(f"  * Joint Offsets: J6 = {gt_j6:+.2f}°, J5 = {gt_j5:+.2f}°, J3 = {gt_j3:+.2f}°")
    print(f"  * Bracket Pos  : X = {gt_pos[0]:+.2f}, Y = {gt_pos[1]:+.2f}, Z = {gt_pos[2]:+.2f} mm")
    print(f"  * Bracket RPY  : R = {gt_rpy[0]:+.2f}, P = {gt_rpy[1]:+.2f}, Y = {gt_rpy[2]:+.2f} deg")
    
    # 1. Test Marker Sweeps
    print(f"\n--- 1. Marker Sweeps (Axis 4, 6, 5) ---")
    res_4 = mc.perform_calibration_sweep(arm_side, 4, sweep_duration=0.5)
    res_6 = mc.perform_calibration_sweep(arm_side, 6, sweep_duration=0.5)
    res_5 = mc.perform_calibration_sweep(arm_side, 5, sweep_duration=0.5)
    print(f"  Axis 4 Radius: {res_4['radius']:.3f} mm, Axis 6 Radius: {res_6['radius']:.3f} mm, Axis 5 Radius: {res_5['radius']:.3f} mm")
    
    joint_offsets_store = {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0}
    
    if robot_ver == "1.3":
        # 2. Test J6 Wrist Roll
        print(f"\n--- 2. J6 Wrist Roll Calibration (Pass 1) ---")
        res_j6 = jc.perform_joint_calibration(arm_side, "wrist_roll_v13", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j6 = res_j6["recommended_joint_offset"]
        joint_offsets_store["joint6"] = opt_j6
        jc.joint_offsets[arm_side]["wrist_roll"] = opt_j6
        mc.joint_offsets[arm_side]["wrist_roll"] = opt_j6
        print(f"  Recommended J6 Offset: {opt_j6:.4f}° (GT: {gt_j6:+.2f}°, Error: {abs(opt_j6 - gt_j6):.4f}°)")
        
        # 3. Test J5 Wrist Pitch
        print(f"\n--- 3. J5 Wrist Pitch Calibration (Pass 1) ---")
        res_j5 = jc.perform_joint_calibration(arm_side, "wrist_pitch_v13", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j5 = res_j5["recommended_joint_offset"]
        joint_offsets_store["joint5"] = opt_j5
        jc.joint_offsets[arm_side]["wrist_pitch"] = opt_j5
        mc.joint_offsets[arm_side]["wrist_pitch"] = opt_j5
        print(f"  Recommended J5 Offset: {opt_j5:.4f}° (GT: {gt_j5:+.2f}°, Error: {abs(opt_j5 - gt_j5):.4f}°)")
        
        # 4. Test Bracket Calculation
        print(f"\n--- 4. Marker Bracket Calibration (3-Axis SVD) ---")
        bracket_res = mc.compute_unified_bracket_calibration(res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_deg=opt_j6, calib_pitch_deg=opt_j5)
        print(f"  Estimated Pos: X={bracket_res['x_e']:.2f} mm, Y={bracket_res['y_e']:.2f} mm, Z={bracket_res['z_e']:.2f} mm")
        print(f"  Estimated RPY: R={bracket_res['roll_e']:.2f}°, P={bracket_res['pitch_e']:.2f}°, Y={bracket_res['yaw_e']:.2f}°")
        
        # 5. Test J3 Elbow
        print(f"\n--- 5. J3 Elbow Calibration (Pass 1) ---")
        res_j3 = jc.perform_joint_calibration(arm_side, "elbow", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j3 = res_j3["recommended_joint_offset"]
        joint_offsets_store["joint3"] = opt_j3
        jc.joint_offsets[arm_side]["elbow"] = opt_j3
        mc.joint_offsets[arm_side]["elbow"] = opt_j3
        print(f"  Recommended J3 Offset: {opt_j3:.4f}° (GT: {gt_j3:+.2f}°, Error: {abs(opt_j3 - gt_j3):.4f}°)")
        
        # Pass 2 Verification & Skip
        print(f"\n--- 6. Pass 2 Convergence Skip Check ---")
        print(f"  J6 Converged in Pass 1: {res_j6.get('converged')} -> Skip Pass 2: {res_j6.get('converged') is True}")
        print(f"  J5 Converged in Pass 1: {res_j5.get('converged')} -> Skip Pass 2: {res_j5.get('converged') is True}")
        print(f"  J3 Converged in Pass 1: {res_j3.get('converged')} -> Skip Pass 2: {res_j3.get('converged') is True}")
        
    else:
        # v1.2 Flow
        # 1. J5
        print(f"\n--- 1. J5 Wrist Pitch Calibration (Pass 1) ---")
        res_j5 = jc.perform_joint_calibration(arm_side, "wrist_pitch", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j5 = res_j5["recommended_joint_offset"]
        joint_offsets_store["joint5"] = opt_j5
        jc.joint_offsets[arm_side]["wrist_pitch"] = opt_j5
        mc.joint_offsets[arm_side]["wrist_pitch"] = opt_j5
        print(f"  Recommended J5 Offset: {opt_j5:.4f}° (GT: {gt_j5:+.2f}°, Error: {abs(opt_j5 - gt_j5):.4f}°)")
        
        # 2. Bracket
        print(f"\n--- 2. Marker Bracket Calibration (v1.2 3-Axis SVD) ---")
        bracket_res = mc.compute_unified_bracket_calibration(res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_or_yaw_deg=0.0, calib_pitch_deg=opt_j5)
        print(f"  Estimated Pos: X={bracket_res['x_e']:.2f} mm, Y={bracket_res['y_e']:.2f} mm, Z={bracket_res['z_e']:.2f} mm")
        print(f"  Estimated RPY: R={bracket_res['roll_e']:.2f}°, P={bracket_res['pitch_e']:.2f}°, Y={bracket_res['yaw_e']:.2f}°")
        
        # 3. J6
        print(f"\n--- 3. J6 Wrist Yaw 2 Calibration (Pass 1) ---")
        res_j6 = jc.perform_joint_calibration(arm_side, "wrist_yaw2", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j6 = res_j6["recommended_joint_offset"]
        joint_offsets_store["joint6"] = opt_j6
        jc.joint_offsets[arm_side]["wrist_yaw2"] = opt_j6
        mc.joint_offsets[arm_side]["wrist_yaw2"] = opt_j6
        print(f"  Recommended J6 Offset: {opt_j6:.4f}° (GT: {gt_j6:+.2f}°, Error: {abs(opt_j6 - gt_j6):.4f}°)")
        
        # 4. J3
        print(f"\n--- 4. J3 Elbow Calibration (Pass 1) ---")
        res_j3 = jc.perform_joint_calibration(arm_side, "elbow", sweep_duration=1.0, current_offset_deg=0.0, pass_idx=1)
        opt_j3 = res_j3["recommended_joint_offset"]
        joint_offsets_store["joint3"] = opt_j3
        jc.joint_offsets[arm_side]["elbow"] = opt_j3
        mc.joint_offsets[arm_side]["elbow"] = opt_j3
        print(f"  Recommended J3 Offset: {opt_j3:.4f}° (GT: {gt_j3:+.2f}°, Error: {abs(opt_j3 - gt_j3):.4f}°)")
        
        # Pass 2 Verification & Skip
        print(f"\n--- 5. Pass 2 Convergence Skip Check ---")
        print(f"  J5 Converged in Pass 1: {res_j5.get('converged')} -> Skip Pass 2: {res_j5.get('converged') is True}")
        print(f"  J6 Converged in Pass 1: {res_j6.get('converged')} -> Skip Pass 2: {res_j6.get('converged') is True}")
        print(f"  J3 Converged in Pass 1: {res_j3.get('converged')} -> Skip Pass 2: {res_j3.get('converged') is True}")

    print(f"\n=======================================================")
    print(f"   SUMMARY OF ERRORS (v{robot_ver} {arm_side.upper()} ARM):")
    print(f"   J6 Offset Error : {abs(joint_offsets_store['joint6'] - gt_j6):.4f}°")
    print(f"   J5 Offset Error : {abs(joint_offsets_store['joint5'] - gt_j5):.4f}°")
    print(f"   J3 Offset Error : {abs(joint_offsets_store['joint3'] - gt_j3):.4f}°")
    print(f"=======================================================\n")

if __name__ == '__main__':
    run_simulation_validation(robot_ver="1.3", arm_side="right")
    run_simulation_validation(robot_ver="1.3", arm_side="left")
    run_simulation_validation(robot_ver="1.2", arm_side="right")
    run_simulation_validation(robot_ver="1.2", arm_side="left")
