[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
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

[ITERATION 2/6] Sweeping physically with staged offset -7.7270°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -5.8474°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP
[ERROR] Marker is not visible.
[INFO] Prompting user for manual teaching due to marker visibility error...
[INFO] Preserved user-taught ready pose for right arm (wrist_pitch).
[INFO] Moving right arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Preserved user-taught ready pose detected for right arm (wrist_pitch). Using taught posture.
[INFO] Ready Pose Reached.

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0057° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -5.8474°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.8474° (click APPLY OFFSET to save).
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
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.00°, optimal_offset=0.01°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0150° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.1694°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0453° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1694°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1694° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0000°
  * Joint 5 Change      : 5.8474°
  * Joint 3 Change      : 2.1694°
  * Bracket Pos Change  : 3.2515 mm
  * Bracket Rot Change  : 1.1450°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-5.8474°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -5.8474° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 right arm (Pass 2/2)...
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
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.29°, optimal_offset=0.24°

[ITERATION 2/6] Sweeping physically with staged offset 0.2388°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.04°, optimal_offset=0.22°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0232° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.2388°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: 0.2388°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.2388° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1694°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1694° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] Calibrating J5 (Wrist Pitch) first on v1.2 left arm...
[INFO] Moving left arm to wrist_pitch Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -3.2828°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -3.9758°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -4.1493°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0355° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -4.1493°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -4.1493° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 left arm (Pass 1/2)...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.11°, optimal_offset=-0.08°

[ITERATION 2/6] Sweeping physically with staged offset -0.0817°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=0.02°, optimal_offset=-0.06°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0197° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0817°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.0817°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.0817° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.7769°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.9674°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0448° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.9674°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9674° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.0817°
  * Joint 5 Change      : 4.1493°
  * Joint 3 Change      : 1.9674°
  * Bracket Pos Change  : 2.0323 mm
  * Bracket Rot Change  : 0.8728°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-4.1493°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: -4.1493° (click APPLY OFFSET to save).
[FULL AUTO 2/3] Performing Marker Bracket Sweeps for v1.2 left arm (Pass 2/2)...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing unified marker bracket calibration for v1.2...
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).

[FULL AUTO] Calibrating J6 (Wrist Yaw 2) under locked bracket...
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset -0.0817°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.10°, optimal_offset=-0.16°

[ITERATION 2/6] Sweeping physically with staged offset -0.1572°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.00°, raw_diff=-0.01°, optimal_offset=-0.17°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0082° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1572°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1572°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1572° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-1.9674°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9674° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.6, Z: -1.0 mm
  * Bracket Rot: R: -1.50, P: +0.50, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.24°, Joint 5: -5.85°, Joint 3: -2.17°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.8, Z: -0.5 mm
  * Bracket Rot: R: -1.12, P: -0.14, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.16°, Joint 5: -4.15°, Joint 3: -1.97°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.1572°
    * Joint 5 (Wrist Pitch): -4.1493°
    * Joint 3 (Elbow)      : -1.9674°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.2388°
    * Joint 5 (Wrist Pitch): -5.8474°
    * Joint 3 (Elbow)      : -2.1694°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[ERROR] Failed to save bracket values: 'Marker_Detection' object has no attribute 'make_transform'
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
Auto base head pose (deg): [-0.02  -0.017]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: -2.5deg
Marker not detected.
[WARNING] Marker not detected at the initial ready pose. Prompting posture adjustment...
[INFO] Right arm marker not visible. Showing teaching dialog...
[INFO] Preserved user-taught ready pose for right arm (elbow).
[INFO] Re-verifying marker visibility at the new posture...
[Sample 1] Captured marker: R_pos=[105.4  24.8 238.1]mm, L_pos=[-47.4  21.2 232.3]mm
[INFO] Posture adjustment successful. Re-building motion plan from current pose...
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 2] Captured marker: R_pos=[ 97.   25.1 224.2]mm, L_pos=[-38.3  21.3 219. ]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 3] Captured marker: R_pos=[109.8  24.5 245.2]mm, L_pos=[-52.1  21.  238.8]mm
Auto motion done: Joint 0 Offset: 5.0deg
[Sample 4] Captured marker: R_pos=[114.3  24.1 252.3]mm, L_pos=[-56.9  20.8 245.4]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[104.   23.  243.8]mm, L_pos=[-47.1  23.1 226.2]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[102.5  21.4 249.4]mm, L_pos=[-46.7  25.  220.1]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[106.4  26.4 232.4]mm, L_pos=[-47.4  19.3 238.2]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[107.4  28.  226.8]mm, L_pos=[-47.2  17.4 244.3]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[110.3  23.5 235.5]mm, L_pos=[-51.5  19.8 228.1]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[115.6  22.4 233.2]mm, L_pos=[-55.8  18.2 224.3]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[100.8  26.1 241.1]mm, L_pos=[-43.8  22.7 236.6]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 96.8  27.5 244.3]mm, L_pos=[-40.6  24.1 241.3]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[107.5  19.3 236.1]mm, L_pos=[-46.4  26.6 235.9]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[109.8  13.9 234.3]mm, L_pos=[-45.6  31.8 239.7]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[103.5  30.3 240.4]mm, L_pos=[-48.5  15.8 228.9]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[101.7  35.8 243. ]mm, L_pos=[-49.9  10.4 225.6]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[103.3  38.9 231.1]mm, L_pos=[-49.4   6.5 237.3]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[112.1  17.3 223.2]mm, L_pos=[-45.4  28.7 252. ]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[ 99.3  32.6 254.7]mm, L_pos=[-49.5  14.3 213.8]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[106.4  10.4 245.1]mm, L_pos=[-44.7  35.7 227.4]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 98.5  31.9 232.1]mm, L_pos=[-40.4  19.1 254.1]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[112.5  19.9 243.5]mm, L_pos=[-55.2  20.9 213.1]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[105.3  24.9 238.1]mm, L_pos=[-47.4  21.3 232.3]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[107.4  24.4 236.6]mm, L_pos=[-45.6  21.3 234.1]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[109.5  24.2 235. ]mm, L_pos=[-43.8  21.3 236.1]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[103.4  25.1 239.8]mm, L_pos=[-49.3  21.1 230.5]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[101.3  25.4 241.5]mm, L_pos=[-51.3  20.9 228.8]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[104.2  26.7 236.9]mm, L_pos=[-46.6  23.3 231.5]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[103.2  28.7 235.5]mm, L_pos=[-45.9  25.6 230.7]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[106.5  22.9 239.4]mm, L_pos=[-48.4  18.9 233.1]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[107.8  21.  240.8]mm, L_pos=[-49.3  16.8 233.7]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[105.3  22.8 239.1]mm, L_pos=[-47.6  23.4 232.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[105.3  20.7 240.1]mm, L_pos=[-47.7  25.6 232.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[105.3  26.8 237.4]mm, L_pos=[-47.2  19.1 232.6]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[105.2  29.  236.5]mm, L_pos=[-47.   16.9 233.1]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[107.6  25.2 243.1]mm, L_pos=[-44.7  20.7 227.8]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[109.6  25.7 248.5]mm, L_pos=[-41.9  20.3 224.4]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[102.7  24.5 234.2]mm, L_pos=[-50.   21.7 237.7]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[ 99.9  24.2 231. ]mm, L_pos=[-52.1  22.4 243.7]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[105.6  26.4 235.9]mm, L_pos=[-47.7  22.9 230.6]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[105.7  28.1 234.5]mm, L_pos=[-48.   24.7 230.1]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[105.   23.2 241.2]mm, L_pos=[-47.   19.5 234.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[104.6  21.6 244.9]mm, L_pos=[-46.4  17.9 238.6]mm
Auto motion done: Head Pan: -5.0deg
[Sample 44] Captured marker: R_pos=[ 80.2  25.3 242.5]mm, L_pos=[-70.9  21.1 224. ]mm
Auto motion done: Head Pan: 5.0deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Head Tilt: -5.0deg
[Sample 45] Captured marker: R_pos=[105.8  49.7 240. ]mm, L_pos=[-47.   45.4 233.5]mm
Auto motion done: Head Tilt: 5.0deg
[Sample 46] Captured marker: R_pos=[105.1   0.3 234.5]mm, L_pos=[-47.6  -2.7 228.8]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260812_021103.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05461719315828338), np.float64(-0.048955506616553075), np.float64(91.49873591948017), np.float64(-0.4961133918359718), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05484329365927013), np.float64(-0.04847158374350183), np.float64(88.87816034374765), np.float64(-0.13781497535560217), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.1694282245885526, 'joint5': -5.847448648740237, 'joint6': 0.23881650443518826}, 'left': {'joint3': -1.967363821236688, 'joint5': -4.149266066127004, 'joint6': -0.15717757244013758}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=2deg, sigma_pos=0.7098mm
Right arm joint offset (deg): [ 8.0030018   3.45735264 -2.27724905  2.11942781 -1.44701382  5.79744762
 -0.18881453]
Left arm joint offset (deg): [-1.60044996 -6.65483512 -3.08089996  2.01736427 -1.83928337  4.19926716
  0.20717925]
Head joint offset (deg): [ 4.71150557 -0.21351634]
mount_to_cam xi: [0. 0. 0. 0. 0. 0.]
mount_to_cam_new: [0.0495, -0.0115, 0.044, -90.0, -0.0, -90.0]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260812_021103.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +8.0030° | Baseline =  +2.4163° | Diff = 5.5867°
   J1: Calc =  +3.4574° | Baseline =  +4.3889° | Diff = 0.9315°
   J2: Calc =  -2.2772° | Baseline =  +0.0004° | Diff = 2.2777°
   J3: Calc =  +2.1194° | Baseline =  +1.5768° | Diff = 0.5426°
   J4: Calc =  -1.4470° | Baseline =  -0.0367° | Diff = 1.4103°
   J5: Calc =  +5.7974° | Baseline =  +5.9579° | Diff = 0.1604°
   J6: Calc =  -0.1888° | Baseline =  +0.0136° | Diff = 0.2024°
 [LEFT ARM]
   J0: Calc =  -1.6004° | Baseline =  +0.2426° | Diff = 1.8430°
   J1: Calc =  -6.6548° | Baseline =  -4.0987° | Diff = 2.5562°
   J2: Calc =  -3.0809° | Baseline =  +0.0009° | Diff = 3.0818°
   J3: Calc =  +2.0174° | Baseline =  +1.9514° | Diff = 0.0659°
   J4: Calc =  -1.8393° | Baseline =  +0.0497° | Diff = 1.8889°
   J5: Calc =  +4.1993° | Baseline =  +4.1961° | Diff = 0.0031°
   J6: Calc =  +0.2072° | Baseline =  -0.0119° | Diff = 0.2190°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260812_021103.json
Arm: both
Right move offset (deg): [-8.003001797966181, -3.4573526350923474, 2.2772490452831247, -2.119427809525323, 1.4470138226410711, -5.797447618316442, 0.18881453368533074]
Left move offset (deg): [1.6004499646746941, 6.654835122238026, 3.0808999632238, -2.017364272969541, 1.8392833669607183, -4.199267155759993, -0.2071792459044839]
Head move offset (deg): [-4.7115055689904075, 0.21351634418266036]
Preview move complete. Inspect the robot pose before applying.
[Check Position] Step 1: Skipping Ready Pose move (already initialized in check session)
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-2.4163395343440595, -4.388874922648515, -0.00043510210396039605, -1.5768100247524752, 0.036694335937499996, -5.957885742187499, -0.013623046875]
Left move offset (deg): [-0.24256942295792083, 4.098661819306931, -0.0008702042079207921, -1.9514329362623763, -0.049658203125, -4.1961181640625, 0.011865234375]
Head move offset (deg): [0.0182373046875, -0.00021972656249999998]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'BASELINE' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-2.4163395343440595, -4.388874922648515, -0.00043510210396039605, -1.5768100247524752, 0.036694335937499996, -5.957885742187499, -0.013623046875]
Left move offset (deg): [-0.24256942295792083, 4.098661819306931, -0.0008702042079207921, -1.9514329362623763, -0.049658203125, -4.1961181640625, 0.011865234375]
Head move offset (deg): [0.0182373046875, -0.00021972656249999998]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'BASELINE' Zero Pose. Now resetting and applying home offset...
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.1694°, J5=-5.8474°, J6=0.2388°] L[J3=-1.9674°, J5=-4.1493°, J6=-0.1572°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (.*) on...
[INFO] Turning servos (.*) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.1694°, J5=-5.8474°, J6=0.2388°] L[J3=-1.9674°, J5=-4.1493°, J6=-0.1572°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Current pose home offset apply complete.
[SUCCESS] Saved offsets permanently to setting.yaml!
[INFO] Zeroed out applied arm offsets in baseline json: /home/nvidia/camera_ws/config/home_reset_baseline.json
Calibration Wizard Finished.
[Step2] Check Calibration State requested.
[Check State] Step 1: Moving to Joint Ready Pose...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
