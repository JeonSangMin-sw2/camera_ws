import os
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from core.paths import CONFIG_PATHS


def execute_full_auto_sequence(
    joint_calibrator,
    marker_calibrator,
    joint_offsets_store: dict,
    stop_event=None,
    log_callback=None,
    status_callback=None,
    bracket_finished_callback=None,
    joint_finished_callback=None,
    save_debug: bool = False
) -> None:
    """
    Executes the full automated multi-pass calibration sequence for dual arms.
    Supports both robot model v1.2 (sequential J5 -> Marker -> J6 -> J3) and 
    v1.3 (unified 3-axis spherical wrist -> Marker -> J3).

    :param joint_calibrator: JointCalibrator instance
    :param marker_calibrator: MarkerCalibrator instance
    :param joint_offsets_store: In-memory store for joint offsets dict
    :param stop_event: threading.Event to signal cancellation
    :param log_callback: Callable[[str], None] for logging messages
    :param status_callback: Callable[[bool], None] for status updates
    :param bracket_finished_callback: Callable[[dict], None] for bracket result data
    :param joint_finished_callback: Callable[[dict], None] for joint result data
    :param save_debug: Whether to save debug trajectory data
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    def emit_status(val: bool):
        if status_callback:
            status_callback(val)

    def emit_bracket(data: dict):
        if bracket_finished_callback:
            bracket_finished_callback(data)

    def emit_joint(data: dict):
        if joint_finished_callback:
            joint_finished_callback(data)

    def is_stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    log("Starting FULL AUTO sequential calibration...")
    if joint_calibrator.robot is None:
        raise RuntimeError("Robot is not connected.")

    version_num = marker_calibrator.get_robot_version()
    is_v13 = (version_num == "1.3")

    for arm_side in ["right", "left"]:
        pass1_joint_results = {"wrist_pitch": None, "elbow": None}
        # Backup of parameters before Pass 1 for early exit / change checking
        prev_j6 = joint_offsets_store[arm_side]["joint6"]
        prev_j5 = joint_offsets_store[arm_side]["joint5"]
        prev_j3 = joint_offsets_store[arm_side]["joint3"]

        # Nominal bracket baseline to calculate bracket parameter changes
        ver_key = "1.3" if is_v13 else "1.2"
        nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm_side]
        tf_vec_init = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
        if tf_vec_init is not None and len(tf_vec_init) == 6:
            prev_bracket_pos = np.array(tf_vec_init[:3]) * 1000.0
            prev_bracket_rot = np.array(tf_vec_init[3:])
        else:
            prev_bracket_pos = np.array(nominal_vec[:3]) * 1000.0
            prev_bracket_rot = np.array(nominal_vec[3:])

        res_4 = None
        res_5 = None
        res_6 = None

        for pass_idx in range(1, 4):
            log("\n" + "=" * 50)
            log(f"   STARTING PASS {pass_idx}/3 FOR {arm_side.upper()} ARM")
            log("=" * 50 + "\n")
            log(f"[INFO] Detected Robot Version: {version_num} (is_v1.3: {is_v13})")

            for calibrator in [joint_calibrator, marker_calibrator]:
                if arm_side not in calibrator.joint_offsets:
                    calibrator.joint_offsets[arm_side] = {}
                calibrator.joint_offsets[arm_side]["wrist_pitch"] = joint_offsets_store[arm_side]["joint5"]
                if is_v13:
                    calibrator.joint_offsets[arm_side]["wrist_roll"] = joint_offsets_store[arm_side]["joint6"]
                    calibrator.joint_offsets[arm_side]["wrist_yaw2"] = 0.0
                else:
                    calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                    calibrator.joint_offsets[arm_side]["wrist_yaw2"] = joint_offsets_store[arm_side]["joint6"]
                calibrator.joint_offsets[arm_side]["elbow"] = joint_offsets_store[arm_side]["joint3"]

            # --- Step 1: Sequential Calibration Execution ---
            if is_v13:
                # === v1.3 UNIFIED 3-AXIS SPHERICAL WRIST SEQUENCE ===
                # 1. 3-Axis Continuous Sweeps (Axis 4: Yaw, Axis 6: Roll, Axis 5: Pitch)
                joint_calibrator.current_calib_mode = None
                marker_calibrator.current_calib_mode = "marker"
                log(f"[FULL AUTO] Starting Unified 3-Axis Sweeps for {arm_side} arm (Pass {pass_idx})...")
                log(f"[FULL AUTO] Moving {arm_side} arm to marker ready pose...")
                if not marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=log):
                    raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                if is_stopped():
                    return

                state = joint_calibrator.robot.get_state()
                model = joint_calibrator.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...")
                res_4 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 4, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_4:
                    raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                res_4['axis_mode'] = 4
                res_4['axis'] = res_4['axis_opt']
                if is_stopped():
                    return

                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...")
                res_6 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 6, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_6:
                    raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                res_6['axis_mode'] = 6
                res_6['axis'] = res_6['axis_opt']
                if is_stopped():
                    return

                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...")
                res_5 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 5, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_5:
                    raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                res_5['axis_mode'] = 5
                res_5['axis'] = res_5['axis_opt']
                if is_stopped():
                    return

                # 2. Phase 1: Compute Wrist Joints (J5 & J6) from 3-Axis Normals
                log(f"\n[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for {arm_side} arm (Pass {pass_idx})...")
                curr_pitch = joint_offsets_store[arm_side]["joint5"]
                curr_roll = joint_offsets_store[arm_side]["joint6"]

                wrist_res = marker_calibrator.compute_wrist_joints_from_3axis_sweeps(
                    res_4, res_5, res_6, arm_side,
                    calib_pitch_deg=curr_pitch, calib_roll_deg=curr_roll
                )
                if not wrist_res:
                    raise RuntimeError(f"Wrist joint orthogonality calculation failed on {arm_side} arm")

                opt_pitch = wrist_res["d5_opt_deg"]
                opt_roll = wrist_res["d6_opt_deg"]
                ortho_err = wrist_res["ortho_err"]

                joint_offsets_store[arm_side]["joint5"] = opt_pitch
                joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch

                joint_offsets_store[arm_side]["joint6"] = opt_roll
                joint_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll
                marker_calibrator.joint_offsets[arm_side]["wrist_roll"] = opt_roll

                log(f"[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: {opt_pitch:+.4f}°")
                log(f"[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: {opt_roll:+.4f}°")
                log(f"[FULL AUTO] [Phase 1] Orthogonality Residual: {ortho_err:.3f}°")

                # Emit UI signals for Joint 6 & Joint 5 results
                joint_res_roll = {
                    'arm_side': arm_side, 'mode': 'wrist_roll_v13', 'pass_idx': pass_idx,
                    'recommended_joint_offset': opt_roll, 'converged': True,
                    'angle_err_deg': abs(opt_roll - curr_roll), 'rmse': res_6.get('rmse', 0.0)
                }
                joint_res_pitch = {
                    'arm_side': arm_side, 'mode': 'wrist_pitch_v13', 'pass_idx': pass_idx,
                    'recommended_joint_offset': opt_pitch, 'converged': True,
                    'angle_err_deg': abs(opt_pitch - curr_pitch), 'rmse': res_5.get('rmse', 0.0)
                }
                if pass_idx == 1:
                    pass1_joint_results["wrist_roll"] = joint_res_roll
                    pass1_joint_results["wrist_pitch"] = joint_res_pitch

                emit_joint(joint_res_roll)
                emit_joint(joint_res_pitch)

                # 3. Phase 2: Compute Pure Marker Bracket Transform (Tf_to_marker)
                log(f"[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for {arm_side} arm...")
                bracket_res = marker_calibrator.compute_marker_bracket_from_orthogonal_sweeps(
                    res_4, res_5, res_6, arm_side
                )
                bracket_res.update(wrist_res)

                # Generate and Save 3-Axis Verification Plot
                bracket_res['res_5'] = res_5
                bracket_res['res_6'] = res_6
                bracket_res['res_4'] = res_4
                bracket_res['arm_side'] = arm_side
                bracket_res['pass_idx'] = pass_idx

                plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                plot_saved = marker_calibrator.generate_marker_plot(res_5, res_6, res_4, bracket_res, arm_side, is_v13, plot_path)
                if plot_saved:
                    bracket_res['plot_path_combined'] = plot_path

                x_m, y_m, z_m = bracket_res['x_e'] / 1000.0, bracket_res['y_e'] / 1000.0, bracket_res['z_e'] / 1000.0
                new_vals = [x_m, y_m, z_m, bracket_res['roll_e'], bracket_res['pitch_e'], bracket_res['yaw_e']]
                key = f"Tf_to_marker_{arm_side}"
                marker_calibrator.camera_config[key] = new_vals
                joint_calibrator.camera_config[key] = new_vals

                emit_bracket(bracket_res)
                time.sleep(0.5)
                if is_stopped():
                    return

                # 4. Calibrate J3 Elbow
                pass1_res_elbow = pass1_joint_results.get("elbow")
                if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                    log(f"[FULL AUTO] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    emit_joint(pass1_res_elbow)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "elbow"
                    log("[FULL AUTO] Sweeping Elbow (Joint 3)...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_elbow = joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_elbow
                    )
                    if not joint_res_elbow:
                        raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["elbow"] = joint_res_elbow
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    joint_res_elbow['pass_idx'] = pass_idx

                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "elbow", pass1_res_elbow if pass1_res_elbow else joint_res_elbow, joint_res_elbow,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_elbow['plot_path_combined'] = plot_path

                    emit_joint(joint_res_elbow)
                    time.sleep(0.5)

            else:
                # === v1.2 CALIBRATION SEQUENCE ===
                # 1. Calibrate J5 (Wrist Pitch) FIRST
                pass1_res_pitch = pass1_joint_results.get("wrist_pitch")
                if pass_idx == 2 and pass1_res_pitch and pass1_res_pitch.get("converged", False):
                    log(f"[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 ({pass1_res_pitch['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_pitch = pass1_res_pitch["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch
                    emit_joint(pass1_res_pitch)
                else:
                    log(f"[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 {arm_side} arm...")
                    for calibrator in [joint_calibrator, marker_calibrator]:
                        calibrator.joint_offsets[arm_side]["wrist_pitch"] = joint_offsets_store[arm_side]["joint5"]
                        calibrator.joint_offsets[arm_side]["wrist_roll"] = 0.0
                        calibrator.joint_offsets[arm_side]["wrist_yaw2"] = joint_offsets_store[arm_side]["joint6"]
                        calibrator.joint_offsets[arm_side]["elbow"] = joint_offsets_store[arm_side]["joint3"]

                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "wrist_pitch", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for wrist_pitch on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_pitch = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_pitch",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint5"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_pitch
                    )
                    if not joint_res_pitch:
                        raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_pitch"] = joint_res_pitch
                    joint_res_pitch['arm_side'] = arm_side
                    joint_res_pitch['mode'] = "wrist_pitch"
                    joint_res_pitch['pass_idx'] = pass_idx

                    opt_pitch = joint_res_pitch["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    marker_calibrator.joint_offsets[arm_side]["wrist_pitch"] = opt_pitch
                    joint_offsets_store[arm_side]["joint5"] = opt_pitch

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_pitch", pass1_res_pitch if pass1_res_pitch else joint_res_pitch, joint_res_pitch,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_pitch['plot_path_combined'] = plot_path

                    emit_joint(joint_res_pitch)
                    time.sleep(0.5)
                if is_stopped():
                    return

                # 2. Marker Bracket Sweeps (Axis 4, 6, 5) with calibrated J5
                joint_calibrator.current_calib_mode = None
                marker_calibrator.current_calib_mode = "marker"
                log(f"[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 {arm_side} arm (Pass {pass_idx}/2)...")
                log(f"[FULL AUTO] Moving {arm_side} arm to ready pose...")
                if not marker_calibrator.perform_move_to_ready_pose(arm_side, log_callback=log):
                    raise RuntimeError(f"Failed to move to marker ready pose on {arm_side} arm")
                if is_stopped():
                    return

                state = joint_calibrator.robot.get_state()
                model = joint_calibrator.robot.model()
                arm_idx = model.left_arm_idx if arm_side == "left" else model.right_arm_idx
                first_starting_pose = list(state.position[arm_idx])
                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 4...")
                res_4 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 4, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_4:
                    raise RuntimeError(f"Axis 4 marker sweep failed on {arm_side} arm")
                res_4['axis_mode'] = 4
                res_4['axis'] = res_4['axis_opt']
                if is_stopped():
                    return

                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 6...")
                res_6 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 6, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_6:
                    raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
                res_6['axis_mode'] = 6
                res_6['axis'] = res_6['axis_opt']
                if is_stopped():
                    return

                if hasattr(marker_calibrator, 'user_taught_ready_poses') and isinstance(marker_calibrator.user_taught_ready_poses, dict):
                    arm_taught = marker_calibrator.user_taught_ready_poses.get(arm_side, {})
                    if isinstance(arm_taught, dict) and "marker" in arm_taught and arm_taught["marker"] is not None:
                        first_starting_pose = list(arm_taught["marker"])

                log(f"[FULL AUTO] Sweeping Axis 5...")
                res_5 = marker_calibrator.perform_calibration_sweep(
                    arm_side, 5, log_callback=log, status_callback=emit_status,
                    save_debug=save_debug, initial_joint_pos=first_starting_pose, pass_idx=pass_idx
                )
                if not res_5:
                    raise RuntimeError(f"Axis 5 marker sweep failed on {arm_side} arm")
                res_5['axis_mode'] = 5
                res_5['axis'] = res_5['axis_opt']
                if is_stopped():
                    return

                # 3. Compute Marker Bracket (1-time lock)
                log("\n[FULL AUTO] Computing unified marker bracket calibration for v1.2...")
                staged_pitch = joint_offsets_store[arm_side]["joint5"]
                staged_yaw2 = joint_offsets_store[arm_side]["joint6"]
                unified_res = marker_calibrator.compute_unified_bracket_calibration(
                    res_5, res_6, arm_side, marker_data_4=res_4, calib_roll_or_yaw_deg=staged_yaw2, calib_pitch_deg=staged_pitch
                )

                unified_res['res_5'] = res_5
                unified_res['res_6'] = res_6
                if res_4 is not None:
                    unified_res['res_4'] = res_4
                unified_res['arm_side'] = arm_side
                unified_res['pass_idx'] = pass_idx

                plot_path = os.path.join(CONFIG_PATHS["plot_dir"], f"circle_fit_{arm_side}_marker_unified.png")
                plot_saved = marker_calibrator.generate_marker_plot(res_5, res_6, res_4, unified_res, arm_side, is_v13, plot_path)
                if plot_saved:
                    unified_res['plot_path_combined'] = plot_path

                x_m, y_m, z_m = unified_res['x_e'] / 1000.0, unified_res['y_e'] / 1000.0, unified_res['z_e'] / 1000.0
                new_vals = [x_m, y_m, z_m, unified_res['roll_e'], unified_res['pitch_e'], unified_res['yaw_e']]
                key = f"Tf_to_marker_{arm_side}"
                marker_calibrator.camera_config[key] = new_vals
                joint_calibrator.camera_config[key] = new_vals

                emit_bracket(unified_res)
                time.sleep(0.5)
                if is_stopped():
                    return

                # 4. Calibrate J6 (Wrist Yaw 2) & Iteration 2 Verification
                pass1_res_yaw2 = pass1_joint_results.get("wrist_yaw2")
                if pass_idx == 2 and pass1_res_yaw2 and pass1_res_yaw2.get("converged", False):
                    log(f"[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 ({pass1_res_yaw2['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_roll = pass1_res_yaw2["recommended_joint_offset"]
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    emit_joint(pass1_res_yaw2)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "wrist_yaw2"
                    log(f"\n[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...")
                    joint_res_roll = joint_calibrator.perform_joint_calibration(
                        arm_side, "wrist_yaw2",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint6"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_yaw2
                    )
                    if not joint_res_roll:
                        raise RuntimeError(f"J6 (Wrist Yaw 2) calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["wrist_yaw2"] = joint_res_roll

                    opt_roll = joint_res_roll["recommended_joint_offset"]
                    log(f"[FULL AUTO] Staging J6 offset: {opt_roll:.4f}°")
                    joint_offsets_store[arm_side]["joint6"] = opt_roll
                    joint_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll
                    marker_calibrator.joint_offsets[arm_side]["wrist_yaw2"] = opt_roll

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "wrist_yaw2", pass1_res_yaw2 if pass1_res_yaw2 else joint_res_roll, joint_res_roll,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_roll['plot_path_combined'] = plot_path

                    joint_res_roll['arm_side'] = arm_side
                    joint_res_roll['mode'] = "wrist_yaw2"
                    joint_res_roll['pass_idx'] = pass_idx
                    emit_joint(joint_res_roll)
                    time.sleep(0.5)
                    if is_stopped():
                        return

                # 5. Calibrate J3 Elbow
                pass1_res_elbow = pass1_joint_results.get("elbow")
                if pass_idx == 2 and pass1_res_elbow and pass1_res_elbow.get("converged", False):
                    log(f"[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 ({pass1_res_elbow['recommended_joint_offset']:.4f}°). Skipping Pass 2 sweep.")
                    opt_elbow = pass1_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow
                    emit_joint(pass1_res_elbow)
                else:
                    marker_calibrator.current_calib_mode = None
                    joint_calibrator.current_calib_mode = "elbow"
                    log("[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...")
                    if not joint_calibrator.perform_move_to_ready_pose(arm_side, "elbow", log_callback=log):
                        raise RuntimeError(f"Failed to move to ready pose for elbow on {arm_side} arm")
                    if is_stopped():
                        return

                    joint_res_elbow = joint_calibrator.perform_joint_calibration(
                        arm_side, "elbow",
                        log_callback=log,
                        status_callback=emit_status,
                        current_offset_deg=joint_offsets_store[arm_side]["joint3"],
                        save_debug=save_debug,
                        pass_idx=pass_idx,
                        pass1_res=pass1_res_elbow
                    )
                    if not joint_res_elbow:
                        raise RuntimeError(f"Elbow joint calibration failed on {arm_side} arm")
                    if pass_idx == 1:
                        pass1_joint_results["elbow"] = joint_res_elbow
                    joint_res_elbow['arm_side'] = arm_side
                    joint_res_elbow['mode'] = "elbow"
                    joint_res_elbow['pass_idx'] = pass_idx

                    opt_elbow = joint_res_elbow["recommended_joint_offset"]
                    joint_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    marker_calibrator.joint_offsets[arm_side]["elbow"] = opt_elbow
                    joint_offsets_store[arm_side]["joint3"] = opt_elbow

                    plot_path = joint_calibrator.save_calibration_comparison_plot(
                        arm_side, "elbow", pass1_res_elbow if pass1_res_elbow else joint_res_elbow, joint_res_elbow,
                        log_callback=log, force_overwrite=True
                    )
                    if plot_path:
                        joint_res_elbow['plot_path_combined'] = plot_path

                    emit_joint(joint_res_elbow)
                    time.sleep(0.5)

            # Pass Evaluation & Convergence Check
            j6_change = abs(joint_offsets_store[arm_side]["joint6"] - prev_j6)
            j5_change = abs(joint_offsets_store[arm_side]["joint5"] - prev_j5)
            j3_change = abs(joint_offsets_store[arm_side]["joint3"] - prev_j3)

            tf_vec_now = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm_side}")
            if tf_vec_now is not None and len(tf_vec_now) == 6:
                now_bracket_pos = np.array(tf_vec_now[:3]) * 1000.0
                now_bracket_rot = np.array(tf_vec_now[3:])
            else:
                now_bracket_pos = prev_bracket_pos
                now_bracket_rot = prev_bracket_rot

            pos_change = np.linalg.norm(now_bracket_pos - prev_bracket_pos)

            # Compute rotation change in degrees
            R_prev = R_scipy.from_euler('ZYX', [prev_bracket_rot[2], prev_bracket_rot[1], prev_bracket_rot[0]], degrees=True)
            R_now = R_scipy.from_euler('ZYX', [now_bracket_rot[2], now_bracket_rot[1], now_bracket_rot[0]], degrees=True)
            rot_change = np.rad2deg(np.linalg.norm((R_now * R_prev.inv()).as_rotvec()))

            log(f"\n[PASS {pass_idx} EVALUATION] Staged parameter changes for {arm_side.upper()} Arm:")
            log(f"  * Joint 6 Change      : {j6_change:.4f}°")
            log(f"  * Joint 5 Change      : {j5_change:.4f}°")
            log(f"  * Joint 3 Change      : {j3_change:.4f}°")
            log(f"  * Bracket Pos Change  : {pos_change:.4f} mm")
            log(f"  * Bracket Rot Change  : {rot_change:.4f}°")

            # Convergence Criteria on Pass >= 2: joints < 0.10°, bracket pos < 0.5 mm, bracket rot < 0.15°
            if pass_idx >= 2:
                if j6_change < 0.10 and j5_change < 0.10 and j3_change < 0.10 and pos_change < 0.5 and rot_change < 0.15:
                    log(f"[PASS {pass_idx} EVALUATION] All parameters converged physically (Step changes < 0.10°).")
                    log(f"[PASS {pass_idx} EVALUATION] Calibration completed in Pass {pass_idx}!")
                    break
                elif pass_idx < 3:
                    log(f"[PASS {pass_idx} EVALUATION] Step changes exceed tolerance (J6: {j6_change:.3f}°, J5: {j5_change:.3f}°). Proceeding to Pass {pass_idx + 1} for verification refinement.")

            # Update prev values for next pass check
            prev_j6 = joint_offsets_store[arm_side]["joint6"]
            prev_j5 = joint_offsets_store[arm_side]["joint5"]
            prev_j3 = joint_offsets_store[arm_side]["joint3"]
            prev_bracket_pos = now_bracket_pos
            prev_bracket_rot = now_bracket_rot

        log(f"[INFO] {arm_side.upper()} arm sequential calibration completed successfully.")
        if is_stopped():
            return
        time.sleep(1.0)

    log("\n" + "=" * 50)
    log("   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!")
    log("=" * 50 + "\n")

    # Print Final Calibrated Results Report in the same style as simulated ground truth
    log("[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):")
    for arm in ["right", "left"]:
        j_store = joint_offsets_store.get(arm, {})
        j6_cal = j_store.get("joint6", 0.0)
        j5_cal = j_store.get("joint5", 0.0)
        j3_cal = j_store.get("joint3", 0.0)

        tf_vec = marker_calibrator.camera_config.get(f"Tf_to_marker_{arm}")
        if tf_vec is not None and len(tf_vec) == 6:
            x_cal = tf_vec[0] * 1000.0
            y_cal = tf_vec[1] * 1000.0
            z_cal = tf_vec[2] * 1000.0
            r_cal = tf_vec[3]
            p_cal = tf_vec[4]
            y_cal_deg = tf_vec[5]
        else:
            ver_key = "1.3" if is_v13 else "1.2"
            nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
            x_cal = nominal_vec[0] * 1000.0
            y_cal = nominal_vec[1] * 1000.0
            z_cal = nominal_vec[2] * 1000.0
            r_cal = nominal_vec[3]
            p_cal = nominal_vec[4]
            y_cal_deg = nominal_vec[5]

        ver_key = "1.3" if is_v13 else "1.2"
        nominal_vec = joint_calibrator.NOMINAL_BRACKET_TEMPLATES[ver_key][arm]
        x_nom = nominal_vec[0] * 1000.0
        y_nom = nominal_vec[1] * 1000.0
        z_nom = nominal_vec[2] * 1000.0
        r_nom = nominal_vec[3]
        p_nom = nominal_vec[4]
        y_nom_deg = nominal_vec[5]

        dx = x_cal - x_nom
        dy = y_cal - y_nom
        dz = z_cal - z_nom

        R_ideal = R_scipy.from_euler('ZYX', [y_nom_deg, p_nom, r_nom], degrees=True)
        R_actual = R_scipy.from_euler('ZYX', [y_cal_deg, p_cal, r_cal], degrees=True)
        R_offset = R_actual * R_ideal.inv()
        yaw_off, pitch_off, roll_off = R_offset.as_euler('ZYX', degrees=True)

        roll_off = (roll_off + 180) % 360 - 180
        pitch_off = (pitch_off + 180) % 360 - 180
        yaw_off = (yaw_off + 180) % 360 - 180

        log(f"  --- {arm.upper()} ARM ---")
        log(f"  * Bracket Pos: X: {dx:+.1f}, Y: {dy:+.1f}, Z: {dz:+.1f} mm")
        log(f"  * Bracket Rot: R: {roll_off:+.2f}, P: {pitch_off:+.2f}, Y: {yaw_off:+.2f} deg")
        log(f"  * Joint Offsets: Joint 6: {j6_cal:+.2f}°, Joint 5: {j5_cal:+.2f}°, Joint 3: {j3_cal:+.2f}°")
    log("==================================================\n")
