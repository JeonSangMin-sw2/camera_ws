[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for right arm (Pass 1/2)...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[WARN] End angle (109.95°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.07°, raw_diff=9.62°, optimal_offset=-0.45°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.9429 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.8048 mm
  * Estimated Circle Center Distance       : 53.0870 mm
  * Calculated Offset Correction           : -0.449239 deg
==================================================
[FULL AUTO] Staging J6 offset: -0.4492°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.4492° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -5.0567°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0208° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -5.0567°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.0567° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.7152°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.6923°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.2608°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.7475°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -2.0400°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 6/6] Sweeping physically with staged offset -1.7507°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0640° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.7507°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.7507° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.3754°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 0.0355°
  * Bracket Pos Change  : 0.8025 mm
  * Bracket Rot Change  : 0.6542°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for right arm (Pass 2/2)...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[WARN] End angle (109.95°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.44°, raw_diff=10.55°, optimal_offset=0.11°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.8791 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.7838 mm
  * Estimated Circle Center Distance       : 53.0128 mm
  * Calculated Offset Correction           : 0.105304 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.1053°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.1053° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -5.0567°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0888° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -5.0567°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.0567° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.7507°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0750° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.7507°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.7507° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for left arm (Pass 1/2)...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[STOP] Stop requested by user.
[STOP] Sending cancel_control to robot!
[INFO] Control cancelled safely via temporary connection.
[ERROR] Marker Axis 6 sweep motion failed or was cancelled.
[ERROR] Full Auto sequential calibration failed: Axis 6 marker sweep failed on left arm
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 1795, in run
    if not res_6: raise RuntimeError(f"Axis 6 marker sweep failed on {arm_side} arm")
RuntimeError: Axis 6 marker sweep failed on left arm

[INFO] Full Auto sequential calibration ended.
