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
[INFO] wrist_yaw2: J7 ready pose=-10.00°, raw_diff=10.39°, optimal_offset=0.39°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.0001 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.3515 mm
  * Estimated Circle Center Distance       : 50.5492 mm
  * Calculated Offset Correction           : 0.388573 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.3886°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.3886° (click APPLY OFFSET to save).

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

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0798° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.5410°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.6866°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.8473°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -1.7001°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0563° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.7001°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.7001° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.3886°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 1.7001°
  * Bracket Pos Change  : 0.0661 mm
  * Bracket Rot Change  : 1.1366°
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
[WARN] End angle (115.00°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=-9.60°, raw_diff=11.00°, optimal_offset=1.40°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 88.9673 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.3067 mm
  * Estimated Circle Center Distance       : 50.4840 mm
  * Calculated Offset Correction           : 1.398536 deg
==================================================
[FULL AUTO] Staging J6 offset: 1.3985°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 1.3985° (click APPLY OFFSET to save).

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

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0691° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.7001°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0350° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.7001°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -1.7001° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 ready pose=10.00°, raw_diff=-9.29°, optimal_offset=0.72°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.8650 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.4572 mm
  * Estimated Circle Center Distance       : 52.9365 mm
  * Calculated Offset Correction           : 0.715984 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.7160°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.7160° (click APPLY OFFSET to save).

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

[ITERATION 2/6] Sweeping physically with staged offset -0.1261°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0408° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -0.1261°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.1261° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8668°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.9887°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0654° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.9887°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9887° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.7160°
  * Joint 5 Change      : 0.1261°
  * Joint 3 Change      : 1.9887°
  * Bracket Pos Change  : 0.7364 mm
  * Bracket Rot Change  : 0.6024°
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
[WARN] End angle (114.88°) exceeds max limit (110.00°). Clamping to 109.50°.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) first...
[INFO] wrist_yaw2: J7 ready pose=10.72°, raw_diff=-10.12°, optimal_offset=0.59°

==================================================
   SWEEP ANALYSIS & RESULTS (WRIST_YAW2)
==================================================
  * Camera Circle Normals Angle (Reference): 89.8109 deg
  * Circle Size Error (abs: r_A - r_B)     : 120.4571 mm
  * Estimated Circle Center Distance       : 52.7128 mm
  * Calculated Offset Correction           : 0.591086 deg
==================================================
[FULL AUTO] Staging J6 offset: 0.5911°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: 0.5911° (click APPLY OFFSET to save).

[FULL AUTO] Computing unified marker bracket calibration (J6 locked)...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Wrist Pitch (Joint 5)...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -0.1261°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0099° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -0.1261°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -0.1261° (click APPLY OFFSET to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -1.9887°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0484° < 0.1° (reached resolution limit)
  * Recommended Absolute Offset: -1.9887°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9887° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.1, Z: -0.1 mm
  * Bracket Rot: R: -0.67, P: -0.07, Y: +0.00 deg
  * Joint Offsets: Joint 6: +1.40°, Joint 5: +0.00°, Joint 3: -1.70°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: -0.1 mm
  * Bracket Rot: R: +1.19, P: -0.18, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.59°, Joint 5: -0.13°, Joint 3: -1.99°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.5911°
    * Joint 5 (Wrist Pitch): -0.1261°
    * Joint 3 (Elbow)      : -1.9887°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 1.3985°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -1.7001°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
Auto base head pose (deg): [-0.008 -0.027]
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
Auto motion done: Joint 2 Offset: -2.5deg
Auto motion done: Joint 2 Offset: -5.0deg
Auto motion done: Joint 2 Offset: 2.5deg
Auto motion done: Joint 2 Offset: 5.0deg
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
Auto motion done: Joint 2 Offset: -2.5deg
Auto motion done: Joint 2 Offset: -5.0deg
Auto motion done: Joint 2 Offset: 2.5deg
Auto motion done: Joint 2 Offset: 5.0deg
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
Auto motion done: Joint 2 Offset: -2.5deg
Auto motion done: Joint 2 Offset: -5.0deg
Auto motion done: Joint 2 Offset: 2.5deg
Auto motion done: Joint 2 Offset: 5.0deg
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
Auto motion done: Joint 2 Offset: -2.5deg
Auto motion done: Joint 2 Offset: -5.0deg
Auto motion done: Joint 2 Offset: 2.5deg
Auto motion done: Joint 2 Offset: 5.0deg
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
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260723_161931.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054083509694316016), np.float64(-0.048059024613677385), np.float64(90.66618428316619), np.float64(0.07010301962344569), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.054053044943639675), np.float64(-0.048105651955588274), np.float64(91.19322997235768), np.float64(-0.1753976948597319), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.7000851555556307, 'joint5': 0.0, 'joint6': 1.3985357574310857}, 'left': {'joint3': -1.9886799693687878, 'joint5': -0.1261388788565863, 'joint6': 0.591085657932739}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-9)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.437deg, sigma_pos=0.1519mm
Right arm joint offset (deg): [ 1.95908047e+00  4.12031826e+00 -1.56328117e-01  1.63383824e+00
  1.36981429e+00  1.00000097e-03 -1.29853577e+00]
Left arm joint offset (deg): [-0.6151945  -4.30507415 -0.22126391  2.08867997  0.30437312  0.12713888
 -0.49108565]
Head joint offset (deg): [0.04505941 0.06670141]
mount_to_cam xi: [-0.00046195  0.00072205 -0.00581324 -0.00085919  0.00012675  0.00066967]
mount_to_cam_new: [0.04569119618671281, 0.00929579179875218, 0.058119135956764933, -90.13586683724394, -1.0131619447466307, -90.1406601391088]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260723_161931.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +1.9591° | Baseline =  +2.2767° | Diff = 0.3176°
   J1: Calc =  +4.1203° | Baseline =  +4.1535° | Diff = 0.0332°
   J2: Calc =  -0.1563° | Baseline =  +0.0002° | Diff = 0.1565°
   J3: Calc =  +1.6338° | Baseline =  +1.6503° | Diff = 0.0165°
   J4: Calc =  +1.3698° | Baseline =  -0.0020° | Diff = 1.3718°
   J5: Calc =  +0.0010° | Baseline =  +0.1178° | Diff = 0.1168°
   J6: Calc =  -1.2985° | Baseline =  +0.0075° | Diff = 1.3060°
 [LEFT ARM]
   J0: Calc =  -0.6152° | Baseline =  -0.6041° | Diff = 0.0111°
   J1: Calc =  -4.3051° | Baseline =  -4.3301° | Diff = 0.0251°
   J2: Calc =  -0.2213° | Baseline =  -0.0161° | Diff = 0.2052°
   J3: Calc =  +2.0887° | Baseline =  +2.0672° | Diff = 0.0215°
   J4: Calc =  +0.3044° | Baseline =  +0.0002° | Diff = 0.3042°
   J5: Calc =  +0.1271° | Baseline =  +0.0483° | Diff = 0.0788°
   J6: Calc =  -0.4911° | Baseline =  +0.0000° | Diff = 0.4911°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260723_161931.json
Arm: both
Right move offset (deg): [-1.9590804745139154, -4.120318260037997, 0.15632811724434442, -1.633838235380067, -1.3698142875630437, -0.0010000009719166349, 1.298535767693034]
Left move offset (deg): [0.6151944968240968, 4.305074154389731, 0.22126391459515454, -2.088679968258366, -0.30437311741771717, -0.1271388775150728, 0.49108565328682713]
Head move offset (deg): [-0.04505941021290129, -0.06670141134311158]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260723_161931.json
Arm: both
Right move offset (deg): [-1.9590804745139154, -4.120318260037997, 0.15632811724434442, -1.633838235380067, -1.3698142875630437, -0.0010000009719166349, 1.298535767693034]
Left move offset (deg): [0.6151944968240968, 4.305074154389731, 0.22126391459515454, -2.088679968258366, -0.30437311741771717, -0.1271388775150728, 0.49108565328682713]
Head move offset (deg): [-0.04505941021290129, -0.06670141134311158]
Preview move complete. Inspect the robot pose before applying.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054083509694316016), np.float64(-0.048059024613677385), np.float64(90.66618428316619), np.float64(0.07010301962344569), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.054053044943639675), np.float64(-0.048105651955588274), np.float64(91.19322997235768), np.float64(-0.1753976948597319), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -1.7000851555556307, 'joint5': 0.0, 'joint6': 1.3985357574310857}, 'left': {'joint3': -1.9886799693687878, 'joint5': -0.1261388788565863, 'joint6': 0.591085657932739}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-9)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.437deg, sigma_pos=0.1519mm
Right arm joint offset (deg): [ 1.95908048e+00  4.12031826e+00 -1.56328117e-01  1.63383824e+00
  1.36981429e+00  1.00000097e-03 -1.29853577e+00]
Left arm joint offset (deg): [-0.6151945  -4.30507415 -0.22126392  2.08867997  0.30437312  0.12713888
 -0.49108565]
Head joint offset (deg): [0.04505941 0.06670141]
mount_to_cam xi: [-0.00046195  0.00072205 -0.00581324 -0.00085919  0.00012675  0.00066967]
mount_to_cam_new: [0.045691196190856896, 0.00929579179950749, 0.05811913595418195, -90.13586683974809, -1.0131619445428754, -90.14066013831017]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260723_174159.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +1.9591° | Baseline =  +2.2767° | Diff = 0.3176°
   J1: Calc =  +4.1203° | Baseline =  +4.1535° | Diff = 0.0332°
   J2: Calc =  -0.1563° | Baseline =  +0.0002° | Diff = 0.1565°
   J3: Calc =  +1.6338° | Baseline =  +1.6503° | Diff = 0.0165°
   J4: Calc =  +1.3698° | Baseline =  -0.0020° | Diff = 1.3718°
   J5: Calc =  +0.0010° | Baseline =  +0.1178° | Diff = 0.1168°
   J6: Calc =  -1.2985° | Baseline =  +0.0075° | Diff = 1.3060°
 [LEFT ARM]
   J0: Calc =  -0.6152° | Baseline =  -0.6041° | Diff = 0.0111°
   J1: Calc =  -4.3051° | Baseline =  -4.3301° | Diff = 0.0251°
   J2: Calc =  -0.2213° | Baseline =  -0.0161° | Diff = 0.2052°
   J3: Calc =  +2.0887° | Baseline =  +2.0672° | Diff = 0.0215°
   J4: Calc =  +0.3044° | Baseline =  +0.0002° | Diff = 0.3042°
   J5: Calc =  +0.1271° | Baseline =  +0.0483° | Diff = 0.0788°
   J6: Calc =  -0.4911° | Baseline =  +0.0000° | Diff = 0.4911°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260723_174159.json
Arm: both
Right move offset (deg): [-1.9590804752013498, -4.1203182601934305, 0.1563281167799882, -1.6338382353608725, -1.3698142874650865, -0.0010000009718495295, 1.2985357676925176]
Left move offset (deg): [0.6151944962799046, 4.30507415452258, 0.2212639151428895, -2.0886799682584245, -0.304373117761332, -0.12713887751515446, 0.4910856532869996]
Head move offset (deg): [-0.04505940942867997, -0.06670141009782884]
Preview move complete. Inspect the robot pose before applying.
