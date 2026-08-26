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

[ITERATION 2/6] Sweeping physically with staged offset -0.0614°...
   STARTING WRIST_PITCH CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0299° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.0614°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_wrist_pitch_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.0614° (click APPLY OFFSET to save).
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.01°, optimal_offset=0.03°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0274° < 0.06° (reached resolution limit)
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

[ITERATION 2/6] Sweeping physically with staged offset -2.1517°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0360° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1517°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1517° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 0.0000°
  * Joint 5 Change      : 0.0614°
  * Joint 3 Change      : 2.1517°
  * Bracket Pos Change  : 0.0955 mm
  * Bracket Rot Change  : 0.6755°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
[FULL AUTO 1/3] J5 (Wrist Pitch) converged in Pass 1 (-0.0614°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch. Staged: -0.0614° (click APPLY OFFSET to save).
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
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (0.0000°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_yaw2. Staged: 0.0000° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1517°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1517° (click APPLY OFFSET to save).
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
  * Step Correction: -0.0046° < 0.06° (reached resolution limit)
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
[INFO] wrist_yaw2: J7 nominal ready pose=0.01°, raw_diff=-0.20°, optimal_offset=-0.14°

[ITERATION 2/6] Sweeping physically with staged offset -0.1437°...
   STARTING WRIST_YAW2 CONTINUOUS OFFSET CALIBRATION SWEEP
[INFO] wrist_yaw2: J7 nominal ready pose=0.02°, raw_diff=0.04°, optimal_offset=-0.10°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0486° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -0.1437°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[FULL AUTO] Staging J6 offset: -0.1437°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_wrist_yaw2_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1437° (click APPLY OFFSET to save).
[FULL AUTO 3/3] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.9836°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.1003°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0277° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1003°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1003° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.1437°
  * Joint 5 Change      : 0.0000°
  * Joint 3 Change      : 2.1003°
  * Bracket Pos Change  : 0.0536 mm
  * Bracket Rot Change  : 0.2075°
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
[FULL AUTO 2/3] J6 (Wrist Yaw 2) converged in Pass 1 (-0.1437°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT wrist_yaw2. Staged: -0.1437° (click APPLY OFFSET to save).
[FULL AUTO 3/3] J3 (Elbow) converged in Pass 1 (-2.1003°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -2.1003° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.2, Z: -0.7 mm
  * Bracket Rot: R: +0.59, P: -0.13, Y: +0.00 deg
  * Joint Offsets: Joint 6: +0.00°, Joint 5: -0.06°, Joint 3: -2.15°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.3, Z: -0.2 mm
  * Bracket Rot: R: -0.84, P: -0.06, Y: +0.00 deg
  * Joint Offsets: Joint 6: -0.14°, Joint 5: +0.00°, Joint 3: -2.10°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Yaw 2): -0.1437°
    * Joint 5 (Wrist Pitch): 0.0000°
    * Joint 3 (Elbow)      : -2.1003°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Yaw 2): 0.0000°
    * Joint 5 (Wrist Pitch): -0.0614°
    * Joint 3 (Elbow)      : -2.1517°
[APPLY] Permanently saved all staged offsets across both arms to setting.yaml successfully!
==================================================

[SUCCESS] Saved Tf_to_marker values for both arms to setting.yaml
[INFO] Dynamically updated marker detector Tf_to_marker transforms in memory.
[APPLY] Full auto results (Joints & Brackets) applied successfully.
[Step2] Init Pose requested.
[Step2] Verifying marker visibility at the initial ready pose...
[INFO] Re-verifying marker visibility at the new posture...
[WARNING] Auto-centering head encountered a minor issue: list indices must be integers or slices, not tuple
[SUCCESS] Marker visibility verified successfully at the ready pose.
Auto base head pose (deg): [0. 0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (+0.0deg) + Head Tilt (-5.0deg)
[Sample 1] Captured marker: R_pos=[ 89.8 -16.2 189.9]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-5.0deg)
[Sample 2] Captured marker: R_pos=[ 82.4 -17.4 176.2]mm, L_pos=[-64.8 -18.3 175.6]mm
Auto motion done: J0 (-5.0deg) + Head Tilt (-10.0deg)
[Sample 3] Captured marker: R_pos=[ 82.5   2.2 181.9]mm, L_pos=[-64.8   1.3 181.3]mm
Auto motion done: J0 (-10.0deg) + Head Tilt (-10.0deg)
[Sample 4] Captured marker: R_pos=[ 75.6 -19.9 162.7]mm, L_pos=[-58.1 -20.7 162.1]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+0.0deg)
[Sample 5] Captured marker: R_pos=[ 97.8   5.7 208.7]mm, L_pos=[-80.1   4.8 208.3]mm
Auto motion done: J0 (+5.0deg) + Head Tilt (+5.0deg)
[Sample 6] Captured marker: R_pos=[ 97.8 -16.3 203.4]mm, L_pos=[-80.2 -17.1 202.9]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+5.0deg)
[Sample 7] Captured marker: R_pos=[106.4   5.5 222.3]mm, L_pos=[-88.6   4.6 221.8]mm
Auto motion done: J0 (+10.0deg) + Head Tilt (+10.0deg)
[Sample 8] Captured marker: R_pos=[106.3 -17.7 216.7]mm, L_pos=[-88.6 -18.4 216.3]mm
Auto motion done: Restore Baseline Pose
[Sample 9] Captured marker: R_pos=[ 89.8 -16.2 189.9]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 10] Captured marker: R_pos=[ 97.2 -17.9 192.4]mm, L_pos=[-65.  -15.4 186.5]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 11] Captured marker: R_pos=[104.6 -19.8 194.6]mm, L_pos=[-57.9 -13.9 183.4]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 12] Captured marker: R_pos=[ 82.6 -14.6 187.1]mm, L_pos=[-79.4 -18.8 191.8]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 13] Captured marker: R_pos=[ 75.4 -13.1 184. ]mm, L_pos=[-86.8 -20.7 194. ]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 14] Captured marker: R_pos=[ 92.4 -23.7 185.5]mm, L_pos=[-74.7 -24.5 184.9]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 15] Captured marker: R_pos=[ 95.1 -31.1 181. ]mm, L_pos=[-77.4 -31.8 180.3]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 16] Captured marker: R_pos=[ 87.4  -8.6 194.1]mm, L_pos=[-69.8  -9.4 193.5]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 17] Captured marker: R_pos=[ 85.   -0.8 198.1]mm, L_pos=[-67.5  -1.7 197.6]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 18] Captured marker: R_pos=[ 91.3 -21.  186. ]mm, L_pos=[-70.9 -12.2 193.3]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 19] Captured marker: R_pos=[ 92.9 -26.  182.4]mm, L_pos=[-69.8  -7.6 197.6]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 20] Captured marker: R_pos=[ 88.5 -11.5 193.9]mm, L_pos=[-73.6 -21.8 185.5]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 21] Captured marker: R_pos=[ 87.5  -6.8 198.2]mm, L_pos=[-75.3 -26.8 181.8]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[ 72.1  -3.6 191.7]mm, L_pos=[-89.  -30.3 186. ]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[ 79.4 -23.  177.1]mm, L_pos=[-85.3 -11.4 202.6]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 24] Captured marker: R_pos=[103.1 -10.7 203.3]mm, L_pos=[-61.8 -23.8 176.7]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 25] Captured marker: R_pos=[106.8 -29.5 186.5]mm, L_pos=[-54.5  -4.3 191.1]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 26] Captured marker: R_pos=[ 69.6   2.3 191.5]mm, L_pos=[-83.2  -5.5 203.1]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
[Sample 27] Captured marker: R_pos=[108.7 -34.6 184.9]mm, L_pos=[-64.2 -28.7 175.4]mm
Auto motion done: Restore Baseline Pose
[Sample 28] Captured marker: R_pos=[ 89.8 -16.2 189.9]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 29] Captured marker: R_pos=[ 95.7 -16.6 197.4]mm, L_pos=[-78.1 -17.5 196.7]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 30] Captured marker: R_pos=[101.9 -17.  204.7]mm, L_pos=[-84.2 -17.9 203.9]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[ 91.4 -11.8 205.1]mm, L_pos=[-71.1  -0.8 213.7]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[ 88.5  -2.3 209.4]mm, L_pos=[-66.2  21.6 226. ]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[ 89.8 -16.2 189.9]mm, L_pos=[-72.2 -17.  189.3]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[ 91.8 -36.2 181. ]mm, L_pos=[-70.3 -37.3 183.8]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[ 94.  -35.8 179.1]mm, L_pos=[-68.1 -37.7 185.9]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[ 87.6 -36.9 184.6]mm, L_pos=[-74.5 -36.8 180. ]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[ 85.8 -37.2 186.4]mm, L_pos=[-76.4 -36.5 178.3]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[ 89.4 -34.4 182.6]mm, L_pos=[-71.9 -35.  181.8]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[ 89.2 -32.6 182.6]mm, L_pos=[-71.7 -33.3 181.8]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[ 90.2 -38.6 182.7]mm, L_pos=[-72.6 -39.2 182.1]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[ 90.7 -40.4 182.9]mm, L_pos=[-72.9 -40.9 182.1]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[ 89.7 -38.8 182.5]mm, L_pos=[-72.2 -34.8 182.2]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[ 89.7 -40.7 182.3]mm, L_pos=[-72.1 -32.9 182.4]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[ 89.7 -34.2 182.9]mm, L_pos=[-72.2 -39.5 181.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[ 89.8 -32.3 183.3]mm, L_pos=[-72.3 -41.4 181.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 46] Captured marker: R_pos=[ 92.8 -37.1 187.8]mm, L_pos=[-69.1 -36.8 177.9]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 47] Captured marker: R_pos=[ 94.7 -37.7 193.6]mm, L_pos=[-66.5 -36.6 174.6]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 48] Captured marker: R_pos=[ 86.4 -36.1 178.5]mm, L_pos=[-75.1 -37.6 186.9]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 49] Captured marker: R_pos=[ 83.8 -36.  175.2]mm, L_pos=[-77.  -38.3 192.9]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 50] Captured marker: R_pos=[ 90.  -33.3 179.7]mm, L_pos=[-72.4 -33.7 178.9]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 51] Captured marker: R_pos=[ 90.1 -30.8 177.5]mm, L_pos=[-72.5 -31.3 176.8]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 52] Captured marker: R_pos=[ 89.5 -39.9 186.5]mm, L_pos=[-72.  -40.7 185.7]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 53] Captured marker: R_pos=[ 89.2 -42.5 190.9]mm, L_pos=[-71.8 -43.4 190.4]mm
Auto motion done: Head Pan: -3.50deg
[Sample 54] Captured marker: R_pos=[ 75.2 -36.8 187.1]mm, L_pos=[-86.3 -36.6 176.6]mm
Auto motion done: Head Pan: -1.75deg
[Sample 55] Captured marker: R_pos=[ 82.5 -36.7 185. ]mm, L_pos=[-79.3 -37.  179.3]mm
Auto motion done: Head Pan: +1.75deg
[Sample 56] Captured marker: R_pos=[ 96.9 -36.2 180. ]mm, L_pos=[-65.  -37.4 184.3]mm
Auto motion done: Head Pan: +3.50deg
[Sample 57] Captured marker: R_pos=[103.9 -35.9 177.3]mm, L_pos=[-57.8 -37.5 186.4]mm
Auto motion done: Head Tilt: -3.50deg
[Sample 58] Captured marker: R_pos=[ 89.8 -22.4 187.9]mm, L_pos=[-72.2 -23.1 187.2]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 59] Captured marker: R_pos=[ 89.8 -29.4 185.4]mm, L_pos=[-72.2 -30.1 184.7]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 60] Captured marker: R_pos=[ 89.7 -43.4 179.7]mm, L_pos=[-72.2 -44.  179. ]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 61] Captured marker: R_pos=[ 89.7 -50.2 176.5]mm, L_pos=[-72.2 -50.8 175.8]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260826_105435.npz
Auto motions sequence completed.
[Step2] Calculate requested.
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.0), np.float64(-0.05377892303553522), np.float64(-0.048749124793614054), np.float64(89.4088037890554), np.float64(0.13377721197411283), 180.0]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.0), np.float64(0.054310187731176324), np.float64(-0.04824730014769126), np.float64(89.15833817313023), np.float64(-0.0614602089272207), 0.0]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.1516934165067894, 'joint5': -0.06137254693253481, 'joint6': 0.0}, 'left': {'joint3': -2.1003027088121917, 'joint5': 0.0, 'joint6': -0.1436684682678275}}

[INFO] === 3-STAGE JOINT-CAMERA DECOUPLED CALIBRATION WORKFLOW ===
[STAGE 1/3] Coarse Multi-DOF Initial Alignment (eps=1e-5)...
[STAGE 2/3] Joint Priority Decoupling (Camera Extrinsics Locked, eps=1e-6)...
[STAGE 3/3] Final Joint-Camera Fine Integration (All Free, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=0.1135deg, sigma_pos=0.6115mm
Right arm joint offset (deg): [-1.12894231  1.13000681  1.47923503  2.15169342 -2.51604072  0.06137255
 -0.        ]
Left arm joint offset (deg): [-1.25767835 -1.16670692 -1.4094092   2.10030271  2.3734273  -0.
  0.14366847]
Head joint offset (deg): [0.23403633 6.5324764 ]
mount_to_cam xi: [-0.00037791 -0.00058127  0.00029292  0.00214685 -0.00975641 -0.00632908]
mount_to_cam_new: [0.038153767514752523, 0.007643118759807803, 0.06746691866163082, -90.07171141094068, -0.09318140620690271, -89.91671320989097]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260826_105435.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -1.1289° | Baseline =  +0.1573° | Diff = 1.2862°
   J1: Calc =  +1.1300° | Baseline =  +2.4059° | Diff = 1.2759°
   J2: Calc =  +1.4792° | Baseline =  +0.0007° | Diff = 1.4786°
   J3: Calc =  +2.1517° | Baseline =  +2.2051° | Diff = 0.0534°
   J4: Calc =  -2.5160° | Baseline =  +0.0020° | Diff = 2.5180°
   J5: Calc =  +0.0614° | Baseline =  +0.0090° | Diff = 0.0524°
   J6: Calc =  -0.0000° | Baseline =  +0.0101° | Diff = 0.0101°
 [LEFT ARM]
   J0: Calc =  -1.2577° | Baseline =  +0.1042° | Diff = 1.3619°
   J1: Calc =  -1.1667° | Baseline =  -1.7528° | Diff = 0.5861°
   J2: Calc =  -1.4094° | Baseline =  -0.0007° | Diff = 1.4088°
   J3: Calc =  +2.1003° | Baseline =  +2.1662° | Diff = 0.0659°
   J4: Calc =  +2.3734° | Baseline =  -0.0002° | Diff = 2.3736°
   J5: Calc =  -0.0000° | Baseline =  -0.0015° | Diff = 0.0015°
   J6: Calc =  +0.1437° | Baseline =  +0.0000° | Diff = 0.1437°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.
