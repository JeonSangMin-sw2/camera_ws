[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for right arm...
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

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.29°, raw_diff=11.10°, optimal_offset=0.81°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.1115 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.5252 mm
  * Estimated Circle Center Distance       : 47.6266 mm
  * Calculated Offset Correction           : 0.812956 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.8130°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.8130° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset 10.1849°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset 31.1703° exceeds safe bounds [-30.0°, 30.0°]. Clamping.

[ITERATION 3/6] Sweeping physically with staged offset 30.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP
[ERROR] Circle fitting failed or error is too large. Aborting step adjustment.

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0000° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: 30.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 30.0000° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -2.4120°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -3.1958°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -3.8697°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset -5.2452° exceeds safe bounds [-5.0°, 0.0°]. Clamping.

[ITERATION 4/6] Sweeping physically with staged offset -5.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset -6.3158° exceeds safe bounds [-5.0°, 0.0°]. Clamping.

[ITERATION 5/6] Sweeping physically with staged offset -5.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset -6.6238° exceeds safe bounds [-5.0°, 0.0°]. Clamping.

[ITERATION 6/6] Sweeping physically with staged offset -5.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
  [SAFETY WARNING] Staged offset -6.4613° exceeds safe bounds [-5.0°, 0.0°]. Clamping.
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -3.0000° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 1.1076°
  * Joint 5 Change      : 39.7201°
  * Joint 3 Change      : 0.5880°
  * Bracket Pos Change  : 0.4254 mm
  * Bracket Rot Change  : 0.6026°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[INFO] Pass 2: Reusing marker sweep datasets from Pass 1.

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.29°, raw_diff=11.10°, optimal_offset=0.81°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.1115 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.5252 mm
  * Estimated Circle Center Distance       : 47.6266 mm
  * Calculated Offset Correction           : 0.812956 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.8130°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.8130° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 30.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP
[ERROR] Marker is not visible.
[ERROR] Iteration 1 sweep failed. Aborting calibration.
[ERROR] Full Auto sequential calibration failed: Wrist pitch joint calibration failed on right arm
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 1893, in run
    raise RuntimeError(f"Wrist pitch joint calibration failed on {arm_side} arm")
RuntimeError: Wrist pitch joint calibration failed on right arm

[INFO] Full Auto sequential calibration ended.
