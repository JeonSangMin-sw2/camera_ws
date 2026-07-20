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
[WARN] End angle (115.01°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-10.00°, raw_diff=10.26°, optimal_offset=0.26°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.0896 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.3687 mm
  * Estimated Circle Center Distance       : 50.6703 mm
  * Calculated Offset Correction           : 0.259026 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.2590°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.2590° (click APPLY OFFSET to save).

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

[ITERATION 2/6] Sweeping physically with staged offset -5.2257°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0555° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -5.2257°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.2257° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.5887°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0980° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.5887°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.5887° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.2590°
  * Joint 5 Change      : 5.2257°
  * Joint 3 Change      : 1.5887°
  * Bracket Pos Change  : 1.2948 mm
  * Bracket Rot Change  : 0.2331°
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
[WARN] End angle (109.78°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-9.74°, raw_diff=9.99°, optimal_offset=0.25°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.1482 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.2919 mm
  * Estimated Circle Center Distance       : 50.9210 mm
  * Calculated Offset Correction           : 0.250703 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.2507°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.2507° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -5.2257°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0083° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -5.2257°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.2257° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.5887°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0930° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.5887°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.5887° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 ready pose=10.00°, raw_diff=-10.34°, optimal_offset=-0.33°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.3658 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.5568 mm
  * Estimated Circle Center Distance       : 51.3283 mm
  * Calculated Offset Correction           : -0.333175 deg
==================================================
[FULL AUTO] Staging J6 offset: -0.3332°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.3332° (click APPLY OFFSET to save).

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

[ITERATION 2/6] Sweeping physically with staged offset -3.2754°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -3.4294°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0073° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -3.4294°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -3.4294° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8617°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0748° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.8617°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.8617° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.3332°
  * Joint 5 Change      : 3.4294°
  * Joint 3 Change      : 1.8617°
  * Bracket Pos Change  : 2.7234 mm
  * Bracket Rot Change  : 0.3303°
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
[WARN] End angle (111.58°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=9.67°, raw_diff=-10.04°, optimal_offset=-0.37°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.5675 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.5831 mm
  * Estimated Circle Center Distance       : 52.0729 mm
  * Calculated Offset Correction           : -0.365452 deg
==================================================
[FULL AUTO] Staging J6 offset: -0.3655°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.3655° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -3.4294°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0377° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -3.4294°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -3.4294° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.8617°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0168° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.8617°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.8617° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.1, Z: -0.1 mm
  * Bracket Rot: R: -0.11, P: -0.06, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.25°, Joint 5: -5.23°, Joint 3: -1.59°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.0, Z: +0.1 mm
  * Bracket Rot: R: +0.67, P: -0.09, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.37°, Joint 5: -3.43°, Joint 3: -1.86°
==================================================

[INFO] Full Auto sequential calibration ended.
[Step2] Init Pose requested.
Auto base head pose (deg): [ 0.011 -1.714]
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.3655°
    * Joint 5 (Wrist Pitch): -3.4294°
    * Joint 3 (Elbow)      : -1.8617°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.2507°
    * Joint 5 (Wrist Pitch): -5.2257°
    * Joint 3 (Elbow)      : -1.5887°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[APPLY] Full auto results (Joints & Brackets) applied successfully.
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
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260720_211940.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054114251696390565), np.float64(-0.0480880339762535), np.float64(90.11143967245832), np.float64(0.06018016371613438), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05396925025857507), np.float64(-0.047894129354994756), np.float64(90.6704338330897), np.float64(-0.09169432764893766), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.5886600365493508, 'joint5': -5.225672360247806, 'joint6': 0.2507030826207082}, 'left': {'joint3': -1.8616834551006551, 'joint5': -3.4293688972329157, 'joint6': -0.3654524465018145}}

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.1539deg, sigma_pos=0.6354mm
Right arm joint offset (deg): [ 4.70060396  3.7403919  -1.48324104  1.68866008  1.36485552  5.22667237
 -0.3507029 ]
Left arm joint offset (deg): [ 2.29336089 -0.65446608 -1.1236207   1.96168344  0.41708958  3.42836891
  0.46545248]
Head joint offset (deg): [-0.06222865  1.15780645]
mount_to_cam xi: [0. 0. 0. 0. 0. 0.]
mount_to_cam_new: [0.047, 0.009, 0.057, -90.0, -0.0, -90.0]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt
Optimization finished successfully.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
Arm: both
Right move offset (deg): [-4.700603955775592, -3.7403919027653942, 1.4832410379698977, -1.6886600770828133, -1.364855515013915, -5.226672374140944, 0.35070289662965304]
Left move offset (deg): [-2.2933608911139074, 0.6544660766584132, 1.1236206957538712, -1.9616834413654962, -0.41708957674478847, -3.4283689118228513, -0.46545248344914886]
Head move offset (deg): [0.06222864594660276, -1.1578064540376385]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
Arm: both
Right move offset (deg): [-4.700603955775592, -3.7403919027653942, 1.4832410379698977, -1.6886600770828133, -1.364855515013915, -5.226672374140944, 0.35070289662965304]
Left move offset (deg): [-2.2933608911139074, 0.6544660766584132, 1.1236206957538712, -1.9616834413654962, -0.41708957674478847, -3.4283689118228513, -0.46545248344914886]
Head move offset (deg): [0.06222864594660276, -1.1578064540376385]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
Arm: both
Right move offset (deg): [-4.700603955775592, -3.7403919027653942, 1.4832410379698977, -1.6886600770828133, -1.364855515013915, -5.226672374140944, 0.35070289662965304]
Left move offset (deg): [-2.2933608911139074, 0.6544660766584132, 1.1236206957538712, -1.9616834413654962, -0.41708957674478847, -3.4283689118228513, -0.46545248344914886]
Head move offset (deg): [0.06222864594660276, -1.1578064540376385]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
Arm: both
Right move offset (deg): [-4.700603955775592, -3.7403919027653942, 1.4832410379698977, -1.6886600770828133, -1.364855515013915, -5.226672374140944, 0.35070289662965304]
Left move offset (deg): [-2.2933608911139074, 0.6544660766584132, 1.1236206957538712, -1.9616834413654962, -0.41708957674478847, -3.4283689118228513, -0.46545248344914886]
Head move offset (deg): [0.06222864594660276, -1.1578064540376385]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_211940.json
Arm: both
Right move offset (deg): [-4.700603955775592, -3.7403919027653942, 1.4832410379698977, -1.6886600770828133, -1.364855515013915, -5.226672374140944, 0.35070289662965304]
Left move offset (deg): [-2.2933608911139074, 0.6544660766584132, 1.1236206957538712, -1.9616834413654962, -0.41708957674478847, -3.4283689118228513, -0.46545248344914886]
Head move offset (deg): [0.06222864594660276, -1.1578064540376385]
Preview move complete. Inspect the robot pose before applying.
[MANUAL OVERRIDE] Staged RIGHT Arm Joint 6 offset manually to 0.0000°. (Not saved to disk yet. Click APPLY OFFSET to save)
[MANUAL OVERRIDE] Staged LEFT Arm Joint 6 offset manually to 0.0000°. (Not saved to disk yet. Click APPLY OFFSET to save)
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0000°
    * Joint 5 (Wrist Pitch): -3.4294°
    * Joint 3 (Elbow)      : -1.8617°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0000°
    * Joint 5 (Wrist Pitch): -5.2257°
    * Joint 3 (Elbow)      : -1.5887°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[Step2] Calculate requested.
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054114251696390565), np.float64(-0.0480880339762535), np.float64(90.11143967245832), np.float64(0.06018016371613438), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05396925025857507), np.float64(-0.047894129354994756), np.float64(90.6704338330897), np.float64(-0.09169432764893766), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.5886600365493508, 'joint5': -5.225672360247806, 'joint6': 0.0}, 'left': {'joint3': -1.8616834551006551, 'joint5': -3.4293688972329157, 'joint6': 0.0}}

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.1645deg, sigma_pos=0.6761mm
Right arm joint offset (deg): [ 4.6905457   3.68741077 -1.28686831  1.68866009  1.07900574  5.22667261
 -0.10000016]
Left arm joint offset (deg): [ 2.28983833 -0.43423967 -1.3711908   1.96168344  0.86605788  3.43036881
  0.0999998 ]
Head joint offset (deg): [-0.01493474  1.24735993]
mount_to_cam xi: [0. 0. 0. 0. 0. 0.]
mount_to_cam_new: [0.047, 0.009, 0.057, -90.0, -0.0, -90.0]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260720_212854.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt
Optimization finished successfully.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260720_212854.json
Arm: both
Right move offset (deg): [-4.690545696349513, -3.6874107686091246, 1.2868683080509706, -1.688660090845423, -1.0790057444396126, -5.2266726094719225, 0.10000015696046402]
Left move offset (deg): [-2.2898383268350044, 0.4342396738977629, 1.3711908009359461, -1.9616834370156284, -0.8660578765708452, -3.430368810611715, -0.09999979542972832]
Head move offset (deg): [0.014934736786303335, -1.247359930350939]
Preview move complete. Inspect the robot pose before applying.
