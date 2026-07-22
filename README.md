[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
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
[WARN] End angle (115.01°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.00°, raw_diff=2.85°, optimal_offset=-7.15°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 87.9737 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.2636 mm
  * Estimated Circle Center Distance       : 48.0584 mm
  * Calculated Offset Correction           : -7.147302 deg
==================================================
[FULL AUTO] Staging J6 offset: -7.1473°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -7.1473° (click APPLY OFFSET to save).

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

[ITERATION 2/6] Sweeping physically with staged offset -8.0043°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -7.6276°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0817° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -7.6276°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -7.6276° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.1211°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0478° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.1211°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.1211° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 7.1473°
  * Joint 5 Change      : 7.6276°
  * Joint 3 Change      : 1.1211°
  * Bracket Pos Change  : 1.4701 mm
  * Bracket Rot Change  : 0.7908°
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

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-17.15°, raw_diff=11.51°, optimal_offset=-5.64°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.7541 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.4402 mm
  * Estimated Circle Center Distance       : 49.7666 mm
  * Calculated Offset Correction           : -5.641026 deg
==================================================
[FULL AUTO] Staging J6 offset: -5.6410°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -5.6410° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -7.6276°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -7.4497°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0080° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -7.4497°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -7.4497° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.1211°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.2302°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0745° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.2302°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.2302° (click APPLY OFFSET to save).
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

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[WARN] End angle (115.01°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=10.00°, raw_diff=1.64°, optimal_offset=11.65°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.8046 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.2198 mm
  * Estimated Circle Center Distance       : 54.7930 mm
  * Calculated Offset Correction           : 11.645351 deg
==================================================
[FULL AUTO] Staging J6 offset: 11.6454°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 11.6454° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -4.5457°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -4.8490°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0171° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -4.8490°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -4.8490° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.5602°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.6686°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0156° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.6686°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.6686° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 11.6454°
  * Joint 5 Change      : 4.8490°
  * Joint 3 Change      : 1.6686°
  * Bracket Pos Change  : 1.9809 mm
  * Bracket Rot Change  : 0.2064°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for left arm (Pass 2/2)...
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

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[WARN] End angle (110.15°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=21.65°, raw_diff=-11.09°, optimal_offset=10.56°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.7346 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.3606 mm
  * Estimated Circle Center Distance       : 54.0116 mm
  * Calculated Offset Correction           : 10.556777 deg
==================================================
[FULL AUTO] Staging J6 offset: 10.5568°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 10.5568° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -4.8490°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0472° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -4.8490°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -4.8490° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.6686°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0126° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.6686°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.6686° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.0, Z: +0.1 mm
  * Bracket Rot: R: +0.51, P: -0.07, Y: +0.00 deg
  * Joint Offsets: Joint 6: -5.64°, Joint 5: -7.45°, Joint 3: -1.23°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: -0.0 mm
  * Bracket Rot: R: -0.63, P: -0.04, Y: +0.00 deg
  * Joint Offsets: Joint 6: +10.56°, Joint 5: -4.85°, Joint 3: -1.67°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 10.5568°
    * Joint 5 (Wrist Pitch): -4.8490°
    * Joint 3 (Elbow)      : -1.6686°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): -5.6410°
    * Joint 5 (Wrist Pitch): -7.4497°
    * Joint 3 (Elbow)      : -1.2302°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
Auto base head pose (deg): [-0.008 -0.012]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260722_173120.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05401411566458229), np.float64(-0.047865622924843906), np.float64(89.48562915168192), np.float64(0.06992239584055633), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05406963853234812), np.float64(-0.048029930751893345), np.float64(89.37125177778128), np.float64(-0.03989458687583355), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.2302416979621213, 'joint5': -7.4496923496490615, 'joint6': -5.641026190394765}, 'left': {'joint3': -1.6685655234061672, 'joint5': -4.848966338794833, 'joint6': 10.556777124959687}}

[INFO] === 2-PASS QP Refinement: Optimizing Arm + Head + Camera Trans (Camera Rot locked) ===
[INFO] Running 2nd Pass QP Refinement for sub-millimeter precision...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.2343deg, sigma_pos=0.51mm
Right arm joint offset (deg): [ 5.39159713e+00 -1.79557316e-09  6.87069925e+00  1.13024170e+00
 -8.70242751e+00  7.44869235e+00  5.74102619e+00]
Left arm joint offset (deg): [ 5.80624417e-02  1.93584836e-09  4.31052070e+00  1.56856552e+00
 -9.87936367e+00  4.84796634e+00 -1.06567771e+01]
Head joint offset (deg): [0.36806385 2.67241879]
mount_to_cam xi: [-0.00103123  0.00079738 -0.01219418  0.00062896 -0.00188488  0.00051195]
mount_to_cam_new: [0.04751267752032829, 0.008382339221449372, 0.05888840217916803, -90.05880925186901, -0.6986985975672688, -90.04532838635468]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260722_173120.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt
Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260722_173120.json
Arm: both
Right move offset (deg): [-5.391597130094448, 1.7955731566213686e-09, -6.87069924734892, -1.1302416990050714, 8.702427513637337, -7.448692350941503, -5.741026189453699]
Left move offset (deg): [-0.05806244165727275, -1.9358483634362086e-09, -4.310520699262243, -1.5685655240006124, 9.879363674782045, -4.847966339060546, 10.656777125931605]
Head move offset (deg): [-0.36806384611033843, -2.6724187921566602]
Preview move complete. Inspect the robot pose before applying.
[Step2] Init Pose requested.
[Step2] Clear Samples requested.
[INFO] Control cancelled safely via temporary connection.
Shared samples cleared.
Init pose failed: Failed to move to Step 1: Joint Ready Pose.
[Step2] Init Pose requested.
Auto base head pose (deg): [-0.368 -2.672]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.02m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motion done: Joint 0 Offset: -2.5deg
Auto motion done: Joint 0 Offset: -5.0deg
Auto motion done: Joint 0 Offset: 2.5deg
Auto motion done: Joint 0 Offset: 5.0deg
Auto motion done: Joint 1 Offset: -2.5deg
Auto motion done: Joint 1 Offset: -5.0deg
Auto motion done: Joint 1 Offset: 2.5deg
Auto motion done: Joint 1 Offset: 5.0deg
Auto motion done: Joint 4 Offset: -2.5deg
Auto motion done: Joint 4 Offset: -5.0deg
Auto motion done: Joint 4 Offset: 2.5deg
Auto motion done: Joint 4 Offset: 5.0deg
Auto motion done: Restore Baseline Pose
Auto motion done: RPY: (-2.50,0.00,0.00)
Auto motion done: RPY: (-5.00,0.00,0.00)
Auto motion done: RPY: (2.50,0.00,0.00)
Auto motion done: RPY: (5.00,0.00,0.00)
Auto motion done: RPY: (0.00,-2.50,0.00)
Auto motion done: RPY: (0.00,-5.00,0.00)
Auto motion done: RPY: (0.00,2.50,0.00)
Auto motion done: RPY: (0.00,5.00,0.00)
Auto motion done: RPY: (0.00,0.00,-2.50)
Auto motion done: RPY: (0.00,0.00,-5.00)
Auto motion done: RPY: (0.00,0.00,2.50)
Auto motion done: RPY: (0.00,0.00,5.00)
Auto motion done: Pos: (0.000,-0.015,0.000)
Auto motion done: Pos: (0.000,-0.030,0.000)
Auto motion done: Pos: (0.000,0.015,0.000)
Auto motion done: Pos: (0.000,0.030,0.000)
Auto motion done: Pos: (0.000,0.000,-0.015)
Auto motion done: Pos: (0.000,0.000,-0.030)
Auto motion done: Pos: (0.000,0.000,0.015)
Auto motion done: Pos: (0.000,0.000,0.030)
Auto motion done: Head Pan: -5.0deg
Auto motion done: Head Pan: 5.0deg
Auto motion done: Head Tilt: -5.0deg
Auto motion done: Head Tilt: 5.0deg
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260722_175935.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05401411566458229), np.float64(-0.047865622924843906), np.float64(89.48562915168192), np.float64(0.06992239584055633), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05406963853234812), np.float64(-0.048029930751893345), np.float64(89.37125177778128), np.float64(-0.03989458687583355), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.2302416979621213, 'joint5': -7.4496923496490615, 'joint6': -5.641026190394765}, 'left': {'joint3': -1.6685655234061672, 'joint5': -4.848966338794833, 'joint6': 10.556777124959687}}

[INFO] === 2-PASS QP Refinement: Optimizing Arm + Head + Camera Trans (Camera Rot locked) ===
[INFO] Running 2nd Pass QP Refinement for sub-millimeter precision...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.2324deg, sigma_pos=0.5371mm
Right arm joint offset (deg): [ 5.42984906e+00 -3.62172807e-09  6.88232088e+00  1.13024170e+00
 -8.70646701e+00  7.44869235e+00  5.74102619e+00]
Left arm joint offset (deg): [ 1.30905523e-01  2.81001207e-09  4.29341122e+00  1.56856552e+00
 -9.86582039e+00  4.84796634e+00 -1.06567771e+01]
Head joint offset (deg): [0.32920821 2.71681158]
mount_to_cam xi: [-0.00157689  0.00011954 -0.01302905  0.0004641  -0.00186913  0.00071001]
mount_to_cam_new: [0.0477114546639923, 0.008548040622976621, 0.0588715428949618, -90.09030947463123, -0.7465146802803471, -90.00626061868272]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260722_175935.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt
Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260722_175935.json
Arm: both
Right move offset (deg): [-5.429849060785946, 3.6217280706954545e-09, -6.882320880777026, -1.1302417005765475, 8.706467006697887, -7.448692353636416, -5.741026189289446]
Left move offset (deg): [-0.13090552328119995, -2.810012066468559e-09, -4.2934112224351395, -1.5685655240178429, 9.865820391590015, -4.847966338381244, 10.656777125480321]
Head move offset (deg): [-0.32920821066115047, -2.7168115835998825]
Preview move complete. Inspect the robot pose before applying.
