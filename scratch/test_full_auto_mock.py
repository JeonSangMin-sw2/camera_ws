import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator

class MockRobotWrapper:
    def __init__(self, model_name="m"):
        self._real_robot = rby1_sdk.create_robot("127.0.0.1", model_name)
        self.position = np.zeros(32)
        
    def model(self):
        return self._real_robot.model()
        
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
        
    def get_state(self):
        class State:
            pass
        s = State()
        s.position = np.array(self.position)
        return s
        
    def is_connected(self):
        return True
        
    def cancel_control(self):
        pass

def run_test(robot_ver="1.3", arm_side="right"):
    print(f"\n=======================================================")
    print(f"   TESTING FULL AUTO CALIBRATION: v{robot_ver} ({arm_side.upper()} ARM)")
    print(f"=======================================================\n")
    
    mock_robot = MockRobotWrapper("m" if robot_ver == "1.3" else "a")
    jc = JointCalibrator(None, mock_robot)
    mc = MarkerCalibrator(None, mock_robot)
    jc.robot_version = robot_ver
    mc.robot_version = robot_ver
    
    # Initialize joint offsets store
    joint_offsets_store = {
        "right": {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0},
        "left":  {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0}
    }
    
    pass1_joint_results = {}
    is_v13 = (robot_ver == "1.3")
    
    for pass_idx in (1, 2):
        print(f"\n--- PASS {pass_idx} ---")
        if is_v13:
            # 1. Marker Sweeps
            print(f"1. Sweeping Axis 4, 6, 5...")
            res_4 = mc.perform_calibration_sweep(arm_side, 4, pass_idx=pass_idx)
            res_6 = mc.perform_calibration_sweep(arm_side, 6, pass_idx=pass_idx)
            res_5 = mc.perform_calibration_sweep(arm_side, 5, pass_idx=pass_idx)
            
            # 2. Calibrate J6 Wrist Roll
            pass1_res_roll = pass1_joint_results.get("wrist_roll")
            if pass_idx == 2 and pass1_res_roll and pass1_res_roll.get("converged", False):
                print(f"[PASS 2 SKIP] J6 (Wrist Roll) converged in Pass 1 ({pass1_res_roll['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_roll = pass1_res_roll["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint6"] = opt_roll
                jc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                mc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
            else:
                print(f"\n2. Calibrating J6 (Wrist Roll)...")
                res_roll = jc.perform_joint_calibration(
                    arm_side, "wrist_roll_v13",
                    current_offset_deg=joint_offsets_store[arm_side]["joint6"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_roll
                )
                if pass_idx == 1:
                    pass1_joint_results["wrist_roll"] = res_roll
                opt_roll = res_roll["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint6"] = opt_roll
                jc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                mc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                print(f"  -> Recommended J6 Offset: {opt_roll:.4f}° (Converged: {res_roll.get('converged')})")
            
            # 3. Calibrate J5 Wrist Pitch
            pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
            if pass_idx == 2 and pass1_res_pitch and pass1_res_pitch.get("converged", False):
                print(f"[PASS 2 SKIP] J5 (Wrist Pitch) converged in Pass 1 ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
            else:
                print(f"\n3. Calibrating J5 (Wrist Pitch)...")
                res_pitch = jc.perform_joint_calibration(
                    arm_side, "wrist_pitch_v13",
                    current_offset_deg=joint_offsets_store[arm_side]["joint5"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_pitch
                )
                if pass_idx == 1:
                    pass1_joint_results["wrist_pitch"] = res_pitch
                opt_pitch = res_pitch["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                print(f"  -> Recommended J5 Offset: {opt_pitch:.4f}° (Converged: {res_pitch.get('converged')})")
            
            # 4. Compute Bracket
            print(f"\n4. Computing Unified Bracket Calibration...")
            unified_res = mc.compute_unified_bracket_calibration(
                res_5, res_6, arm_side, marker_data_4=res_4,
                calib_roll_deg=opt_roll, calib_pitch_deg=opt_pitch
            )
            print(f"  -> Estimated Bracket Position (mm): X={unified_res['x_e']:.2f}, Y={unified_res['y_e']:.2f}, Z={unified_res['z_e']:.2f}")
            print(f"  -> Estimated Bracket RPY (deg): R={unified_res['roll_e']:.2f}, P={unified_res['pitch_e']:.2f}, Y={unified_res['yaw_e']:.2f}")
            
            # 5. Calibrate J3 Elbow
            pass1_res_elbow = pass1_joint_results.get("elbow")
            if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                print(f"[PASS 2 SKIP] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow
            else:
                print(f"\n5. Calibrating J3 (Elbow)...")
                res_elbow = jc.perform_joint_calibration(
                    arm_side, "elbow",
                    current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_elbow
                )
                if pass_idx == 1:
                    pass1_joint_results["elbow"] = res_elbow
                opt_elbow = res_elbow["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow
                print(f"  -> Recommended J3 Offset: {opt_elbow:.4f}° (Converged: {res_elbow.get('converged')})")
            
        else:
            # v1.2 Flow
            # 1. J5
            pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
            if pass_idx == 2 and pass1_res_pitch and pass1_res_pitch.get("converged", False):
                print(f"[PASS 2 SKIP] J5 (Wrist Pitch) converged in Pass 1 ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
            else:
                print(f"\n1. Calibrating J5 (Wrist Pitch)...")
                res_pitch = jc.perform_joint_calibration(
                    arm_side, "wrist_pitch",
                    current_offset_deg=joint_offsets_store[arm_side]["joint5"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_pitch
                )
                if pass_idx == 1:
                    pass1_joint_results["wrist_pitch"] = res_pitch
                opt_pitch = res_pitch["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                print(f"  -> Recommended J5 Offset: {opt_pitch:.4f}°")
            
            # 2. Marker Sweeps
            print(f"\n2. Sweeping Axis 4, 6, 5...")
            res_4 = mc.perform_calibration_sweep(arm_side, 4, pass_idx=pass_idx)
            res_6 = mc.perform_calibration_sweep(arm_side, 6, pass_idx=pass_idx)
            res_5 = mc.perform_calibration_sweep(arm_side, 5, pass_idx=pass_idx)
            
            # 3. Compute Bracket
            print(f"\n3. Computing Unified Bracket Calibration...")
            staged_yaw2 = joint_offsets_store[arm_side]["joint6"]
            unified_res = mc.compute_unified_bracket_calibration(
                res_5, res_6, arm_side, marker_data_4=res_4,
                calib_roll_or_yaw_deg=staged_yaw2, calib_pitch_deg=opt_pitch
            )
            print(f"  -> Estimated Bracket Position (mm): X={unified_res['x_e']:.2f}, Y={unified_res['y_e']:.2f}, Z={unified_res['z_e']:.2f}")
            print(f"  -> Estimated Bracket RPY (deg): R={unified_res['roll_e']:.2f}, P={unified_res['pitch_e']:.2f}, Y={unified_res['yaw_e']:.2f}")
            
            # 4. J6
            pass1_res_yaw2 = pass1_joint_results.get("wrist_yaw2")
            if pass_idx == 2 and pass1_res_yaw2 and pass1_res_yaw2.get("converged", False):
                print(f"[PASS 2 SKIP] J6 (Wrist Yaw 2) converged in Pass 1 ({pass1_res_yaw2['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_roll = pass1_res_yaw2["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint6"] = opt_roll
                jc.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                mc.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
            else:
                print(f"\n4. Calibrating J6 (Wrist Yaw 2)...")
                res_roll = jc.perform_joint_calibration(
                    arm_side, "wrist_yaw2",
                    current_offset_deg=joint_offsets_store[arm_side]["joint6"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_yaw2
                )
                if pass_idx == 1:
                    pass1_joint_results["wrist_yaw2"] = res_roll
                opt_roll = res_roll["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint6"] = opt_roll
                jc.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                mc.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                print(f"  -> Recommended J6 Offset: {opt_roll:.4f}°")
            
            # 5. J3
            pass1_res_elbow = pass1_joint_results.get("elbow")
            if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                print(f"[PASS 2 SKIP] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow
            else:
                print(f"\n5. Calibrating J3 (Elbow)...")
                res_elbow = jc.perform_joint_calibration(
                    arm_side, "elbow",
                    current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                    pass_idx=pass_idx,
                    pass1_res=pass1_res_elbow
                )
                if pass_idx == 1:
                    pass1_joint_results["elbow"] = res_elbow
                opt_elbow = res_elbow["recommended_joint_offset"]
                joint_offsets_store[arm_side]["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow
                print(f"  -> Recommended J3 Offset: {opt_elbow:.4f}°")

    print(f"\n=======================================================")
    print(f"   FINAL RESULTS: v{robot_ver} ({arm_side.upper()} ARM)")
    print(f"   Joint 6 Offset: {joint_offsets_store[arm_side]['joint6']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS[arm_side]['joint6']:.2f}°)")
    gt_j5 = BaseCalibrator.MOCK_GT_OFFSETS[arm_side]['joint5_v13' if is_v13 else 'joint5_v12']
    print(f"   Joint 5 Offset: {joint_offsets_store[arm_side]['joint5']:.4f}° (GT: {gt_j5:.2f}°)")
    print(f"   Joint 3 Offset: {joint_offsets_store[arm_side]['joint3']:.4f}° (GT: {BaseCalibrator.MOCK_GT_OFFSETS[arm_side]['joint3']:.2f}°)")
    print(f"=======================================================\n")

if __name__ == '__main__':
    run_test(robot_ver="1.3", arm_side="right")
    run_test(robot_ver="1.3", arm_side="left")
    run_test(robot_ver="1.2", arm_side="right")
    run_test(robot_ver="1.2", arm_side="left")
