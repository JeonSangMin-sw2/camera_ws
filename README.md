[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial arm & head joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/3 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 right arm...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -0.2666°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0543° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.2666°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.2666° (click APPLY OFFSET to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) against nominal bracket reference...
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-1.71°, optimal_offset=-1.70°

[ITERATION 2/6] Sweeping physically with staged offset -1.6957°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[ERROR] Marker is not visible.
[INFO] Prompting user for manual teaching due to marker visibility error...
[INFO] Preserved user-taught ready pose for right arm (wrist_yaw2).
[INFO] Moving right arm to wrist_yaw2 Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Preserved user-taught ready pose detected for right arm (wrist_yaw2). Using taught posture.
[INFO] Ready Pose Reached.
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=0.67°, optimal_offset=-1.02°

[ITERATION 3/6] Sweeping physically with staged offset -1.0219°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=0.54°, optimal_offset=-0.48°

[ITERATION 4/6] Sweeping physically with staged offset -0.3855°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.67°, optimal_offset=-1.06°

[ITERATION 5/6] Sweeping physically with staged offset -0.6752°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=0.24°, optimal_offset=-0.43°

[ITERATION 6/6] Sweeping physically with staged offset -0.2783°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.67°, optimal_offset=-0.94°

[INFO] Joint wrist_yaw2 did not meet 0.06° convergence tolerance due to measurement noise floor.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.4788°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.4788° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 right arm (Pass 1/2)...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[ERROR] Full Auto sequential calibration failed: Bracket-only fit rejected: converged=True, rank=6/6, at_bounds=False, normalized_rms=7.8672. Calibrate J5/J6 before fitting the bracket; check fixed joint inputs and sweep quality.
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 2325, in run
    unified_res = self.marker_calibrator.compute_unified_bracket_calibration(
  File "/home/nvidia/camera_ws/core/calibration/MarkerCalibrator.py", line 514, in compute_unified_bracket_calibration
    return self.fit_encoder_bracket(marker_data_4, marker_data_5, marker_data_6, arm_side, calib_roll_or_yaw_deg if calib_roll_or_yaw_deg is not None else calib_roll_deg, calib_pitch_deg)
  File "/home/nvidia/camera_ws/core/calibration/MarkerCalibrator.py", line 887, in fit_encoder_bracket
    result = fit_bracket_sweeps(self.robot, side, [data4, data5, data6], nominal,
  File "/home/nvidia/camera_ws/core/calibration/bracket_fitting.py", line 64, in fit_bracket_sweeps
    raise RuntimeError(
RuntimeError: Bracket-only fit rejected: converged=True, rank=6/6, at_bounds=False, normalized_rms=7.8672. Calibrate J5/J6 before fitting the bracket; check fixed joint inputs and sweep quality.

[INFO] Full Auto sequential calibration ended.
[ERROR] Full Auto Calibration FAILED: Bracket-only fit rejected: converged=True, rank=6/6, at_bounds=False, normalized_rms=7.8672. Calibrate J5/J6 before fitting the bracket; check fixed joint inputs and sweep quality.
