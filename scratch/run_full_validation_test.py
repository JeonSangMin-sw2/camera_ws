import sys
import os
import numpy as np

# Ensure path includes camera_ws root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rby1_sdk
from core.calibration.CalibratorBase import BaseCalibrator
from core.calibration.JointCalibrator import JointCalibrator
from core.calibration.MarkerCalibrator import MarkerCalibrator

def generate_ideal_sweep_data(calibrator, arm_side, sweep_joint, start_deg, end_deg, cand_joint=None, current_offset_deg=0.0, num_points=60):
    """Generates synthetic sweep dataset using ground truth model without real-time sleep."""
    is_v13 = calibrator.is_v13()
    robot = calibrator.robot
    model = robot.model()
    dyn_model = robot.get_dynamics()
    arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
    
    # Ready pose
    mode_name = "wrist_pitch_v13" if (is_v13 and cand_joint == 5) else ("wrist_roll_v13" if (is_v13 and cand_joint == 6) else ("wrist_pitch" if cand_joint == 5 else ("wrist_yaw2" if cand_joint == 6 else "elbow")))
    ready_pose = calibrator.get_ready_pose("v1.3" if is_v13 else "v1.2", "joint" if cand_joint is not None else "marker", mode_name if cand_joint is not None else "marker", arm_side)
    
    q_base = np.zeros(32)
    for idx, val in zip(arm_idx, ready_pose):
        q_base[idx] = val
        
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), num_points)
    dataset = []
    
    for ang in angles:
        q_curr = q_base.copy()
        q_curr[arm_idx[sweep_joint]] += ang
        
        # In mock GT, simulated camera pose
        pose = calibrator.get_simulated_marker_pose(
            arm_side, sweep_joint=sweep_joint, 
            current_offset_deg=current_offset_deg, 
            cand_joint=cand_joint, 
            q_actual=q_curr
        )
        dataset.append((q_curr, pose))
        
    return dataset

def run_arm_validation(robot_ver="1.3", arm_side="right"):
    print(f"\n=======================================================")
    print(f"   FULL VALIDATION TEST: ROBOT v{robot_ver} ({arm_side.upper()} ARM)")
    print(f"=======================================================")
    
    is_v13 = (robot_ver == "1.3")
    robot = rby1_sdk.create_robot("127.0.0.1", "m" if is_v13 else "a")
    jc = JointCalibrator(None, robot)
    mc = MarkerCalibrator(None, robot)
    jc.robot_version = robot_ver
    mc.robot_version = robot_ver
    
    arm_idx = robot.model().left_arm_idx if arm_side == "left" else robot.model().right_arm_idx
    
    gt = BaseCalibrator.MOCK_GT_OFFSETS[arm_side]
    gt_j6 = gt["joint6"]
    gt_j5 = gt["joint5_v13" if is_v13 else "joint5_v12"]
    gt_j3 = gt["joint3"]
    gt_bracket_pos = [x * 1000.0 for x in gt["bracket_pos"]]
    gt_bracket_rpy = gt["bracket_rpy"]
    
    print(f"[GROUND TRUTH OFFSETS]")
    print(f"  Joint 6: {gt_j6:+.2f}° | Joint 5: {gt_j5:+.2f}° | Joint 3: {gt_j3:+.2f}°")
    print(f"  Bracket Pos (mm): X={gt_bracket_pos[0]:+.2f}, Y={gt_bracket_pos[1]:+.2f}, Z={gt_bracket_pos[2]:+.2f}")
    print(f"  Bracket RPY (deg): R={gt_bracket_rpy[0]:+.2f}, P={gt_bracket_rpy[1]:+.2f}, Y={gt_bracket_rpy[2]:+.2f}")
    
    joint_offsets_store = {"joint3": 0.0, "joint5": 0.0, "joint6": 0.0}
    pass1_results = {}
    
    for pass_idx in (1, 2):
        print(f"\n-------------------------------------------------------")
        print(f"   PASS {pass_idx} EXECUTION")
        print(f"-------------------------------------------------------")
        
        if is_v13:
            # === v1.3 SEQUENTIAL WORKFLOW ===
            # 1. Marker Sweeps (Axis 4, 6, 5)
            print(f"[1/5] Sweeping Marker Axes 4, 6, 5...")
            data_4 = generate_ideal_sweep_data(mc, arm_side, 4, -15.0, 15.0)
            data_6 = generate_ideal_sweep_data(mc, arm_side, 6, -15.0, 15.0)
            data_5 = generate_ideal_sweep_data(mc, arm_side, 5, 0.0, -30.0)
            
            res_4 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_4], [np.degrees(q[arm_idx[4]] - data_4[0][0][arm_idx[4]]) for q, _ in data_4], axis_prior=[0,0,1], robust=False)
            res_6 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_6], [np.degrees(q[arm_idx[6]] - data_6[0][0][arm_idx[6]]) for q, _ in data_6], axis_prior=[1,0,0], robust=False)
            res_5 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_5], [np.degrees(q[arm_idx[5]] - data_5[0][0][arm_idx[5]]) for q, _ in data_5], axis_prior=[0,1,0], robust=False)
            
            # 2. Calibrate J6 Wrist Roll (Axis 6 & 5 sweeps)
            if pass_idx == 2 and pass1_results.get("wrist_roll", {}).get("converged"):
                print(f"[PASS 2 SKIP] J6 (Wrist Roll) converged in Pass 1 ({pass1_results['wrist_roll']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_roll = pass1_results["wrist_roll"]["recommended_joint_offset"]
            else:
                print(f"[2/5] Calibrating J6 (Wrist Roll)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 6, -20.0, 20.0, cand_joint=6, current_offset_deg=joint_offsets_store["joint6"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 5, -15.0, 15.0, cand_joint=6, current_offset_deg=joint_offsets_store["joint6"])
                res_j6 = jc.compute_calibration_results(arm_side, "wrist_roll_v13", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint6"], cand_joint=6, sweep_joint_A=6, sweep_joint_B=5)
                opt_roll = res_j6["recommended_joint_offset"]
                res_j6["converged"] = abs(opt_roll - joint_offsets_store["joint6"]) < 0.05
                if pass_idx == 1:
                    pass1_results["wrist_roll"] = res_j6
                print(f"  -> Recommended J6 Offset: {opt_roll:.4f}° (Converged: {res_j6['converged']})")
                joint_offsets_store["joint6"] = opt_roll
                jc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                mc.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                
            # 3. Calibrate J5 Wrist Pitch (Axis 6 & 4 sweeps)
            if pass_idx == 2 and pass1_results.get("wrist_pitch", {}).get("converged"):
                print(f"[PASS 2 SKIP] J5 (Wrist Pitch) converged in Pass 1 ({pass1_results['wrist_pitch']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_pitch = pass1_results["wrist_pitch"]["recommended_joint_offset"]
            else:
                print(f"[3/5] Calibrating J5 (Wrist Pitch)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 6, -15.0, 15.0, cand_joint=5, current_offset_deg=joint_offsets_store["joint5"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 4, -15.0, 15.0, cand_joint=5, current_offset_deg=joint_offsets_store["joint5"])
                res_j5 = jc.compute_calibration_results(arm_side, "wrist_pitch_v13", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint5"], cand_joint=5, sweep_joint_A=6, sweep_joint_B=4)
                opt_pitch = res_j5["recommended_joint_offset"]
                res_j5["converged"] = abs(opt_pitch - joint_offsets_store["joint5"]) < 0.05
                if pass_idx == 1:
                    pass1_results["wrist_pitch"] = res_j5
                print(f"  -> Recommended J5 Offset: {opt_pitch:.4f}° (Converged: {res_j5['converged']})")
                joint_offsets_store["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                
            # 4. Marker Bracket Calibration (3-Axis SVD)
            print(f"[4/5] Computing 3-Axis SVD Bracket Calibration...")
            bracket_res = mc.compute_unified_bracket_calibration(res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_deg=opt_roll, calib_pitch_deg=opt_pitch)
            print(f"  -> Bracket Pos (mm): X={bracket_res['x_e']:.2f}, Y={bracket_res['y_e']:.2f}, Z={bracket_res['z_e']:.2f}")
            print(f"  -> Bracket RPY (deg): R={bracket_res['roll_e']:.2f}, P={bracket_res['pitch_e']:.2f}, Y={bracket_res['yaw_e']:.2f}")
            
            # 5. Calibrate J3 Elbow (Axis 2 & 4 sweeps)
            if pass_idx == 2 and pass1_results.get("elbow", {}).get("converged"):
                print(f"[PASS 2 SKIP] J3 (Elbow) converged in Pass 1 ({pass1_results['elbow']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_elbow = pass1_results["elbow"]["recommended_joint_offset"]
            else:
                print(f"[5/5] Calibrating J3 (Elbow)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 2, -15.0, 15.0, cand_joint=3, current_offset_deg=joint_offsets_store["joint3"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 4, -15.0, 15.0, cand_joint=3, current_offset_deg=joint_offsets_store["joint3"])
                res_j3 = jc.compute_calibration_results(arm_side, "elbow", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint3"], cand_joint=3, sweep_joint_A=2, sweep_joint_B=4)
                opt_elbow = res_j3["recommended_joint_offset"]
                res_j3["converged"] = abs(opt_elbow - joint_offsets_store["joint3"]) < 0.05
                if pass_idx == 1:
                    pass1_results["elbow"] = res_j3
                print(f"  -> Recommended J3 Offset: {opt_elbow:.4f}° (Converged: {res_j3['converged']})")
                joint_offsets_store["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow
                
        else:
            # === v1.2 SEQUENTIAL WORKFLOW ===
            # 1. J5 Wrist Pitch (Axis 4 & 6 sweeps)
            if pass_idx == 2 and pass1_results.get("wrist_pitch", {}).get("converged"):
                print(f"[PASS 2 SKIP] J5 (Wrist Pitch) converged in Pass 1 ({pass1_results['wrist_pitch']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_pitch = pass1_results["wrist_pitch"]["recommended_joint_offset"]
            else:
                print(f"[1/5] Calibrating J5 (Wrist Pitch)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 4, -15.0, 15.0, cand_joint=5, current_offset_deg=joint_offsets_store["joint5"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 6, -15.0, 15.0, cand_joint=5, current_offset_deg=joint_offsets_store["joint5"])
                res_j5 = jc.compute_calibration_results(arm_side, "wrist_pitch", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint5"], cand_joint=5, sweep_joint_A=4, sweep_joint_B=6)
                opt_pitch = res_j5["recommended_joint_offset"]
                res_j5["converged"] = abs(opt_pitch - joint_offsets_store["joint5"]) < 0.05
                if pass_idx == 1:
                    pass1_results["wrist_pitch"] = res_j5
                print(f"  -> Recommended J5 Offset: {opt_pitch:.4f}° (Converged: {res_j5['converged']})")
                joint_offsets_store["joint5"] = opt_pitch
                jc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                mc.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                
            # 2. Marker Sweeps (Axis 4, 6, 5)
            print(f"[2/5] Sweeping Marker Axes 4, 6, 5...")
            data_4 = generate_ideal_sweep_data(mc, arm_side, 4, -15.0, 15.0)
            data_6 = generate_ideal_sweep_data(mc, arm_side, 6, -15.0, 15.0)
            data_5 = generate_ideal_sweep_data(mc, arm_side, 5, 0.0, -30.0)
            
            res_4 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_4], [np.degrees(q[arm_idx[4]] - data_4[0][0][arm_idx[4]]) for q, _ in data_4], axis_prior=[0,0,1], robust=False)
            res_6 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_6], [np.degrees(q[arm_idx[6]] - data_6[0][0][arm_idx[6]]) for q, _ in data_6], axis_prior=[0,0,1], robust=False)
            res_5 = mc.fit_circle_3d_and_6dof_misalignment([p for _, p in data_5], [np.degrees(q[arm_idx[5]] - data_5[0][0][arm_idx[5]]) for q, _ in data_5], axis_prior=[0,1,0], robust=False)
            
            # 3. Marker Bracket Calibration (v1.2 3-Axis SVD)
            print(f"[3/5] Computing v1.2 3-Axis SVD Bracket Calibration...")
            staged_yaw2 = joint_offsets_store["joint6"]
            bracket_res = mc.compute_unified_bracket_calibration(res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_or_yaw_deg=staged_yaw2, calib_pitch_deg=opt_pitch)
            print(f"  -> Bracket Pos (mm): X={bracket_res['x_e']:.2f}, Y={bracket_res['y_e']:.2f}, Z={bracket_res['z_e']:.2f}")
            print(f"  -> Bracket RPY (deg): R={bracket_res['roll_e']:.2f}, P={bracket_res['pitch_e']:.2f}, Y={bracket_res['yaw_e']:.2f}")
            
            # 4. Calibrate J6 Wrist Yaw 2 (Axis 6 & 5 sweeps)
            if pass_idx == 2 and pass1_results.get("wrist_yaw2", {}).get("converged"):
                print(f"[PASS 2 SKIP] J6 (Wrist Yaw 2) converged in Pass 1 ({pass1_results['wrist_yaw2']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_yaw2 = pass1_results["wrist_yaw2"]["recommended_joint_offset"]
            else:
                print(f"[4/5] Calibrating J6 (Wrist Yaw 2)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 6, -20.0, 20.0, cand_joint=6, current_offset_deg=joint_offsets_store["joint6"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 5, -15.0, 15.0, cand_joint=6, current_offset_deg=joint_offsets_store["joint6"])
                res_j6 = jc.compute_calibration_results(arm_side, "wrist_yaw2", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint6"], cand_joint=6, sweep_joint_A=6, sweep_joint_B=5)
                opt_yaw2 = res_j6["recommended_joint_offset"]
                res_j6["converged"] = abs(opt_yaw2 - joint_offsets_store["joint6"]) < 0.05
                if pass_idx == 1:
                    pass1_results["wrist_yaw2"] = res_j6
                print(f"  -> Recommended J6 Offset: {opt_yaw2:.4f}° (Converged: {res_j6['converged']})")
                joint_offsets_store["joint6"] = opt_yaw2
                jc.joint_offsets[arm_side]["wrist_yaw2"] = opt_yaw2
                mc.joint_offsets[arm_side]["wrist_yaw2"] = opt_yaw2
                
            # 5. Calibrate J3 Elbow (Axis 2 & 4 sweeps)
            if pass_idx == 2 and pass1_results.get("elbow", {}).get("converged"):
                print(f"[PASS 2 SKIP] J3 (Elbow) converged in Pass 1 ({pass1_results['elbow']['recommended_joint_offset']:.4f}°). Skipping.")
                opt_elbow = pass1_results["elbow"]["recommended_joint_offset"]
            else:
                print(f"[5/5] Calibrating J3 (Elbow)...")
                data_A = generate_ideal_sweep_data(jc, arm_side, 2, -15.0, 15.0, cand_joint=3, current_offset_deg=joint_offsets_store["joint3"])
                data_B = generate_ideal_sweep_data(jc, arm_side, 4, -15.0, 15.0, cand_joint=3, current_offset_deg=joint_offsets_store["joint3"])
                res_j3 = jc.compute_calibration_results(arm_side, "elbow", data_A, data_B, [0]*7, current_offset_deg=joint_offsets_store["joint3"], cand_joint=3, sweep_joint_A=2, sweep_joint_B=4)
                opt_elbow = res_j3["recommended_joint_offset"]
                res_j3["converged"] = abs(opt_elbow - joint_offsets_store["joint3"]) < 0.05
                if pass_idx == 1:
                    pass1_results["elbow"] = res_j3
                print(f"  -> Recommended J3 Offset: {opt_elbow:.4f}° (Converged: {res_j3['converged']})")
                joint_offsets_store["joint3"] = opt_elbow
                jc.joint_offsets[arm_side]["elbow"] = opt_elbow
                mc.joint_offsets[arm_side]["elbow"] = opt_elbow

    err_j6 = abs(joint_offsets_store["joint6"] - gt_j6)
    err_j5 = abs(joint_offsets_store["joint5"] - gt_j5)
    err_j3 = abs(joint_offsets_store["joint3"] - gt_j3)
    
    print(f"\n=======================================================")
    print(f"   ACCURACY VERIFICATION SUMMARY: v{robot_ver} ({arm_side.upper()} ARM)")
    print(f"   J6 Offset: Calib = {joint_offsets_store['joint6']:+.4f}°, GT = {gt_j6:+.2f}° -> Error = {err_j6:.4f}°")
    print(f"   J5 Offset: Calib = {joint_offsets_store['joint5']:+.4f}°, GT = {gt_j5:+.2f}° -> Error = {err_j5:.4f}°")
    print(f"   J3 Offset: Calib = {joint_offsets_store['joint3']:+.4f}°, GT = {gt_j3:+.2f}° -> Error = {err_j3:.4f}°")
    print(f"=======================================================\n")
    
    assert err_j6 < 0.01, f"J6 error too large: {err_j6}"
    assert err_j5 < 0.01, f"J5 error too large: {err_j5}"
    assert err_j3 < 0.01, f"J3 error too large: {err_j3}"
    print(f"[SUCCESS] All joint offsets converged within < 0.01° error!")

if __name__ == '__main__':
    run_arm_validation(robot_ver="1.3", arm_side="right")
    run_arm_validation(robot_ver="1.3", arm_side="left")
    run_arm_validation(robot_ver="1.2", arm_side="right")
    run_arm_validation(robot_ver="1.2", arm_side="left")
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! BOTH v1.3 AND v1.2 FULLY VALIDATED! 🎉\n")
