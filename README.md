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

[ITERATION 2/6] Sweeping physically with staged offset -0.3147°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -0.4360°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0162° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.4360°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.4360° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.25°, optimal_offset=0.26°

[ITERATION 2/6] Sweeping physically with staged offset 0.2564°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.20°, optimal_offset=0.07°

[ITERATION 3/6] Sweeping physically with staged offset 0.0675°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.16°, optimal_offset=-0.08°

[ITERATION 4/6] Sweeping physically with staged offset -0.0786°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.11°, optimal_offset=0.04°

[ITERATION 5/6] Sweeping physically with staged offset 0.0339°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.18°, optimal_offset=-0.14°

[ITERATION 6/6] Sweeping physically with staged offset -0.0882°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.01°, optimal_offset=-0.09°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0298° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0882°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.0882°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0882° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.2534°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0549° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2534°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2534° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0882°
  * Joint 5 Change      : 0.4360°
  * Joint 3 Change      : 2.2534°
  * Bracket Pos Change  : 0.0874 mm
  * Bracket Rot Change  : 0.5290°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.4360°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.4360° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset -0.0882°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.12°, optimal_offset=-0.20°

[ITERATION 2/6] Sweeping physically with staged offset -0.1968°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.26°, optimal_offset=0.07°

[ITERATION 3/6] Sweeping physically with staged offset 0.0561°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.08°, optimal_offset=-0.01°

[ITERATION 4/6] Sweeping physically with staged offset -0.0083°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.38°, optimal_offset=-0.38°

[ITERATION 5/6] Sweeping physically with staged offset -0.2406°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=0.23°, optimal_offset=-0.00°

[ITERATION 6/6] Sweeping physically with staged offset -0.0017°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.02°, optimal_offset=-0.02°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0066° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0017°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.0017°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: -0.0017° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.2534°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2534° (click APPLY OFFSET to save).
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

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0101° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.16°, optimal_offset=-0.15°

[ITERATION 2/6] Sweeping physically with staged offset -0.1459°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.16°, optimal_offset=-0.29°

[ITERATION 3/6] Sweeping physically with staged offset -0.2916°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.30°, optimal_offset=0.03°

[ITERATION 4/6] Sweeping physically with staged offset 0.0215°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.30°, optimal_offset=-0.27°

[ITERATION 5/6] Sweeping physically with staged offset -0.1705°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.00°, optimal_offset=-0.16°

[ITERATION 6/6] Sweeping physically with staged offset -0.1008°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=-0.08°, optimal_offset=-0.16°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0175° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1008°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1008°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1008° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.6575°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.1576°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0338° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1576°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1576° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.1008°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.1576°
  * Bracket Pos Change  : 0.9882 mm
  * Bracket Rot Change  : 0.3063°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (0.0000°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch. Staged: 0.0000° (click APPLY OFFSET to save).
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

[ITERATION 1/6] Sweeping physically with staged offset -0.1008°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.03°, optimal_offset=-0.05°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0510° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1008°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1008°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1008° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1576°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1576° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: -0.0, Z: +45.7 mm
  * Bracket Rot: R: -1.83, P: -0.08, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.00°, Joint 5: -0.44°, Joint 3: -2.25°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.1, Z: +45.3 mm
  * Bracket Rot: R: +2.48, P: -0.29, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.10°, Joint 5: +0.00°, Joint 3: -2.16°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.1008°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.1576°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.0017°
    * Joint 5 (Wrist Pitch): -0.4360°
    * Joint 3 (Elbow)      : -2.2534°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: Joint 0 Offset: 0.0deg
Marker not detected.
[WARNING] Marker not detected at the initial ready pose. Prompting posture adjustment...
[INFO] Right arm marker not visible. Showing teaching dialog...
[INFO] Preserved user-taught ready pose for right arm (elbow).
[INFO] Re-verifying marker visibility at the new posture...
[Sample 1] Captured marker: R_pos=[ 87.2 -21.3 191.1]mm, L_pos=[-49.8 -20.3 193.5]mm
[INFO] Posture adjustment successful. Re-building motion plan from current pose...
Auto motion done: Joint 0 Offset: -2.5deg
[Sample 2] Captured marker: R_pos=[ 82.6 -34.  184.3]mm, L_pos=[-45.2 -33.  186.3]mm
Auto motion done: Joint 0 Offset: -5.0deg
[Sample 3] Captured marker: R_pos=[ 78.  -46.4 177.1]mm, L_pos=[-40.6 -45.6 178.9]mm
Auto motion done: Joint 0 Offset: 2.5deg
[Sample 4] Captured marker: R_pos=[ 92.   -8.3 197.2]mm, L_pos=[-54.5  -7.1 199.8]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 5] Captured marker: R_pos=[ 95.7 -22.  194.5]mm, L_pos=[-41.  -19.9 188.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 6] Captured marker: R_pos=[104.2 -23.  197.5]mm, L_pos=[-32.4 -19.7 184.1]mm
Auto motion done: Joint 1 Offset: 2.5deg
[Sample 7] Captured marker: R_pos=[ 78.8 -20.8 187.2]mm, L_pos=[-58.7 -20.8 197.4]mm
Auto motion done: Joint 1 Offset: 5.0deg
[Sample 8] Captured marker: R_pos=[ 70.7 -20.5 183.3]mm, L_pos=[-67.7 -21.7 201.3]mm
Auto motion done: Joint 2 Offset: -2.5deg
[Sample 9] Captured marker: R_pos=[ 94.5 -34.2 189.6]mm, L_pos=[-56.8 -33.8 192. ]mm
Auto motion done: Joint 2 Offset: -5.0deg
[Sample 10] Captured marker: R_pos=[102.3 -46.8 188. ]mm, L_pos=[-64.2 -47.1 190.5]mm
Auto motion done: Joint 2 Offset: 2.5deg
[Sample 11] Captured marker: R_pos=[ 80.4  -8.2 191.9]mm, L_pos=[-43.2  -6.4 194.2]mm
Auto motion done: Joint 2 Offset: 5.0deg
[Sample 12] Captured marker: R_pos=[ 73.9   5.2 192.5]mm, L_pos=[-37.1   7.7 194.7]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 13] Captured marker: R_pos=[ 88.8 -25.8 186.7]mm, L_pos=[-48.7 -16.  197.9]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 14] Captured marker: R_pos=[ 90.6 -30.5 182.5]mm, L_pos=[-47.8 -12.  202.7]mm
Auto motion done: Joint 4 Offset: 2.5deg
[Sample 15] Captured marker: R_pos=[ 85.9 -16.8 195.6]mm, L_pos=[-51.1 -24.5 189. ]mm
Auto motion done: Joint 4 Offset: 5.0deg
[Sample 16] Captured marker: R_pos=[ 84.7 -12.4 200.2]mm, L_pos=[-52.6 -28.9 184.8]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 17] Captured marker: R_pos=[ 67.1 -11.5 192. ]mm, L_pos=[-69.5 -30.2 192.4]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 18] Captured marker: R_pos=[ 75.1 -29.7 175.3]mm, L_pos=[-66.8 -13.6 211.1]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 19] Captured marker: R_pos=[102.9 -14.4 207.3]mm, L_pos=[-36.2 -28.4 176. ]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 20] Captured marker: R_pos=[106.5 -32.1 188.5]mm, L_pos=[-29.5 -11.4 193. ]mm
Auto motion done: Joint 1+2 (+5.0,+5.0)deg
[Sample 21] Captured marker: R_pos=[ 56.1   5.4 183.1]mm, L_pos=[-56.4   6.7 204.3]mm
Auto motion done: Joint 1+2 (-5.0,-5.0)deg
[Sample 22] Captured marker: R_pos=[117.8 -49.2 192.9]mm, L_pos=[-48.2 -46.  182.8]mm
Auto motion done: Restore Baseline Pose
[Sample 23] Captured marker: R_pos=[ 87.2 -21.3 191.1]mm, L_pos=[-49.8 -20.3 193.5]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 24] Captured marker: R_pos=[ 89.2 -21.1 190.7]mm, L_pos=[-47.8 -20.4 194. ]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 25] Captured marker: R_pos=[ 91.2 -20.9 190.6]mm, L_pos=[-45.7 -20.5 194.7]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 26] Captured marker: R_pos=[ 85.2 -21.5 191.4]mm, L_pos=[-51.8 -20.1 192.8]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 27] Captured marker: R_pos=[ 83.2 -21.8 191.8]mm, L_pos=[-53.9 -20.  192.6]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 28] Captured marker: R_pos=[ 86.8 -21.2 190.7]mm, L_pos=[-49.5 -20.1 193.2]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 29] Captured marker: R_pos=[ 86.3 -21.1 190.4]mm, L_pos=[-49.1 -19.9 193.1]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 30] Captured marker: R_pos=[ 87.7 -21.5 191.4]mm, L_pos=[-50.  -20.4 193.4]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 31] Captured marker: R_pos=[ 88.1 -21.4 191.5]mm, L_pos=[-50.3 -20.6 193.7]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 32] Captured marker: R_pos=[ 87.3 -23.7 190.8]mm, L_pos=[-49.7 -18.  193.9]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 33] Captured marker: R_pos=[ 87.3 -26.  190.7]mm, L_pos=[-49.6 -15.7 194.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 34] Captured marker: R_pos=[ 87.2 -19.1 191.4]mm, L_pos=[-49.8 -22.5 192.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 35] Captured marker: R_pos=[ 87.2 -16.7 191.8]mm, L_pos=[-49.9 -24.9 192.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 36] Captured marker: R_pos=[103.1 -21.1 191.6]mm, L_pos=[-33.9 -20.1 192.4]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 37] Captured marker: R_pos=[118.9 -21.  191.9]mm, L_pos=[-18.1 -20.  191.7]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 38] Captured marker: R_pos=[ 71.4 -21.4 190.6]mm, L_pos=[-65.6 -20.3 194.2]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 39] Captured marker: R_pos=[ 55.6 -21.4 189.9]mm, L_pos=[-81.5 -20.2 195.1]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 40] Captured marker: R_pos=[ 87.4  -6.5 190.3]mm, L_pos=[-49.9  -5.5 193.5]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 41] Captured marker: R_pos=[ 87.5   8.2 189.6]mm, L_pos=[-50.    9.3 193.2]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 42] Captured marker: R_pos=[ 86.9 -36.  191.8]mm, L_pos=[-49.5 -35.  193.6]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 43] Captured marker: R_pos=[ 86.7 -50.7 192.5]mm, L_pos=[-49.3 -49.7 193.8]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260805_194834.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.054049938192777675), np.float64(-0.002255458732120841), np.float64(91.83495557770591), np.float64(0.07578674789861581), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.05408121486793118), np.float64(-0.002655701998355449), np.float64(92.47965867087319), np.float64(-0.29350386746667584), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2534378397835346, 'joint5': -0.43596855459654615, 'joint6': -0.0016884614483332785}, 'left': {'joint3': -2.157574727066855, 'joint5': 0.0, 'joint6': -0.10082613713101526}}

[INFO] === 3-STAGE QP SEQUENTIAL OPTIMIZATION WORKFLOW ===
[STAGE 1/3] Global Rough Initialization (eps=1e-6)...
[STAGE 2/3] Joint Priority Refinement (Camera Extrinsics Locked, Arm + Head Free, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1.0
measurement_noise = sigma_rot=0.2073deg, sigma_pos=0.4011mm
Right arm joint offset (deg): [ 1.30024735  4.80923822 -1.42057528  2.30343796  1.83472668  0.48596875
 -0.04831198]
Left arm joint offset (deg): [-1.3050674  -4.29467649  0.7282694   2.20757461 -1.13467104  0.04999982
  0.1508257 ]
head_base-to-camera xi: [ 4.34005776e-02 -6.23174189e-03  8.20373065e-03  1.36254099e-04
  1.70690853e-04  2.41236207e-05]
head_base_to_cam_new: [0.10228195714594253, 0.010793242575831538, 0.04134102437631112, -86.03432494027327, 0.07251297509555492, -89.9110509408219]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260805_194834.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  +1.3002° | Baseline =  +0.0000° | Diff = 1.3002°
   J1: Calc =  +4.8092° | Baseline =  +0.0000° | Diff = 4.8092°
   J2: Calc =  -1.4206° | Baseline =  +0.0000° | Diff = 1.4206°
   J3: Calc =  +2.3034° | Baseline =  +0.0000° | Diff = 2.3034°
   J4: Calc =  +1.8347° | Baseline =  +0.0000° | Diff = 1.8347°
   J5: Calc =  +0.4860° | Baseline =  +0.0000° | Diff = 0.4860°
   J6: Calc =  -0.0483° | Baseline =  +0.0000° | Diff = 0.0483°
 [LEFT ARM]
   J0: Calc =  -1.3051° | Baseline =  +0.0000° | Diff = 1.3051°
   J1: Calc =  -4.2947° | Baseline =  +0.0000° | Diff = 4.2947°
   J2: Calc =  +0.7283° | Baseline =  +0.0000° | Diff = 0.7283°
   J3: Calc =  +2.2076° | Baseline =  +0.0000° | Diff = 2.2076°
   J4: Calc =  -1.1347° | Baseline =  +0.0000° | Diff = 1.1347°
   J5: Calc =  +0.0500° | Baseline =  +0.0000° | Diff = 0.0500°
   J6: Calc =  +0.1508° | Baseline =  +0.0000° | Diff = 0.1508°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_194834.json
Arm: both
Right move offset (deg): [-1.3002473477779088, -4.809238223404453, 1.4205752843509059, -2.303437964129378, -1.8347266775633917, -0.48596875040474635, 0.04831197989826306]
Left move offset (deg): [1.3050674043382788, 4.294676487380523, -0.7282694013018681, -2.207574609175794, 1.1346710416376022, -0.04999982376681485, -0.15082569670478796]
Preview move complete. Inspect the robot pose before applying.
[Step2] Apply Home Offset requested.

===== HOME OFFSET PREVIEW: Baseline Zero =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
Left move offset (deg): [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
Preview move complete. Inspect the robot pose before applying.

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_194834.json
Arm: both
Right move offset (deg): [-1.3002473477779088, -4.809238223404453, 1.4205752843509059, -2.303437964129378, -1.8347266775633917, -0.48596875040474635, 0.04831197989826306]
Left move offset (deg): [1.3050674043382788, 4.294676487380523, -0.7282694013018681, -2.207574609175794, 1.1346710416376022, -0.04999982376681485, -0.15082569670478796]
Preview move complete. Inspect the robot pose before applying.
[INFO] Moving robot to 'OPTIMIZED' Zero Pose before applying home offset...

===== HOME OFFSET PREVIEW: Optimized Zero =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260805_194834.json
Arm: both
Right move offset (deg): [-1.3002473477779088, -4.809238223404453, 1.4205752843509059, -2.303437964129378, -1.8347266775633917, -0.48596875040474635, 0.04831197989826306]
Left move offset (deg): [1.3050674043382788, 4.294676487380523, -0.7282694013018681, -2.207574609175794, 1.1346710416376022, -0.04999982376681485, -0.15082569670478796]
Preview move complete. Inspect the robot pose before applying.
[INFO] Arrived at 'OPTIMIZED' Zero Pose. Now resetting and applying home offset...
[APPLY] Saving optimized mount_to_cam to setting.yaml: [0.10228195714594253, 0.010793242575831538, 0.04134102437631112, -86.03432494027327, 0.07251297509555492, -89.9110509408219]
[APPLY] Saving optimized head_base_to_cam to setting.yaml: [0.10228195714594253, 0.010793242575831538, 0.04134102437631112, -86.03432494027327, 0.07251297509555492, -89.9110509408219]
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.2534°, J5=-0.4360°, J6=-0.0017°] L[J3=-2.1576°, J5=0.0000°, J6=-0.1008°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (^(?!head_joint_).*$) on...
[INFO] Turning servos (^(?!head_joint_).*$) on...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.0'
[INFO] Loaded joint offsets from setting.yaml: R[J3=-2.2534°, J5=-0.4360°, J6=-0.0017°] L[J3=-2.1576°, J5=0.0000°, J6=-0.1008°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Current pose home offset apply complete.
[SUCCESS] Saved offsets permanently to setting.yaml!
[INFO] Zeroed out applied arm offsets in baseline json: /home/nvidia/camera_ws/config/home_reset_baseline.json
