[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
[FULL AUTO] Initial joint offsets reset to 0.0 before starting calibration.
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for right arm (Pass 1/2)...
[FULL AUTO] Moving right arm to marker ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Computing Simultaneous 3-Axis Solution for right arm...
[FULL AUTO] Staged Joint 5 (Pitch) Offset: +1.4171°
[FULL AUTO] Staged Joint 6 (Roll)  Offset: -3.9987°
[FULL AUTO] Orthogonality Residual: 2.574°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -3.9987° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 1.4171° (click APPLY OFFSET to save).
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 1.4171°, Joint 6: -3.9987°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -2.1271°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.2040°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0339° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.2040°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2040° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 3.9987°
  * Joint 5 Change      : 1.4171°
  * Joint 3 Change      : 2.2040°
  * Bracket Pos Change  : 30.3566 mm
  * Bracket Rot Change  : 3.4846°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for right arm (Pass 2/2)...
[FULL AUTO] Moving right arm to marker ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[FULL AUTO] Computing Simultaneous 3-Axis Solution for right arm...
[FULL AUTO] Staged Joint 5 (Pitch) Offset: +1.8688°
[FULL AUTO] Staged Joint 6 (Roll)  Offset: -1.3008°
[FULL AUTO] Orthogonality Residual: 1.677°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -1.3008° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 1.8688° (click APPLY OFFSET to save).
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 1.8688°, Joint 6: -1.3008°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-2.2040°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.2040° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING PASS 1/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for left arm (Pass 1/2)...
[FULL AUTO] Moving left arm to marker ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing Simultaneous 3-Axis Solution for left arm...
[FULL AUTO] Staged Joint 5 (Pitch) Offset: +0.0594°
[FULL AUTO] Staged Joint 6 (Roll)  Offset: +1.7223°
[FULL AUTO] Orthogonality Residual: 2.541°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 1.7223° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: 0.0594° (click APPLY OFFSET to save).
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: 0.0594°, Joint 6: 1.7223°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.7633°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -1.9787°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.9161°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0465° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.9161°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9161° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 1.7223°
  * Joint 5 Change      : 0.0594°
  * Joint 3 Change      : 1.9161°
  * Bracket Pos Change  : 31.1542 mm
  * Bracket Rot Change  : 2.6268°
[PASS 1 EVALUATION] Some changes exceed thresholds. Proceeding to Pass 2 for refinement.

==================================================
   STARTING PASS 2/2 FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO] Starting Unified 3-Axis Sweeps for left arm (Pass 2/2)...
[FULL AUTO] Moving left arm to marker ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4 (Wrist Yaw)...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_left_marker_axis_4.txt
[FULL AUTO] Sweeping Axis 6 (Wrist Roll)...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_left_marker_axis_6.txt
[FULL AUTO] Sweeping Axis 5 (Wrist Pitch)...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_left_marker_axis_5.txt

[FULL AUTO] Computing Simultaneous 3-Axis Solution for left arm...
[FULL AUTO] Staged Joint 5 (Pitch) Offset: +0.2360°
[FULL AUTO] Staged Joint 6 (Roll)  Offset: +1.5560°
[FULL AUTO] Orthogonality Residual: 2.585°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 1.5560° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: 0.2360° (click APPLY OFFSET to save).
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: 0.2360°, Joint 6: 1.5560°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-1.9161°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9161° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: -5.9, Y: +30.0, Z: -3.7 mm
  * Bracket Rot: R: -2.70, P: -0.71, Y: +0.37 deg
  * Joint Offsets: Joint 6: -1.30°, Joint 5: +1.87°, Joint 3: -2.20°
  --- LEFT ARM ---
  * Bracket Pos: X: -7.9, Y: +30.0, Z: -4.1 mm
  * Bracket Rot: R: +0.17, P: +0.18, Y: +0.36 deg
  * Joint Offsets: Joint 6: +1.56°, Joint 5: +0.24°, Joint 3: -1.92°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Roll) : 1.5560°
    * Joint 5 (Wrist Pitch): 0.2360°
    * Joint 3 (Elbow)      : -1.9161°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Roll) : -1.3008°
    * Joint 5 (Wrist Pitch): 1.8688°
    * Joint 3 (Elbow)      : -2.2040°
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
Auto base head pose (deg): [ 0. -0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
[Sample 1] Captured marker: R_pos=[117.1  -4.9 170.4]mm, L_pos=[-95.2  -6.1 173.5]mm
Auto motion done: J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]
[Sample 2] Captured marker: R_pos=[112.4 -10.1 160.5]mm, L_pos=[-90.6 -11.4 163.3]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]
[Sample 3] Captured marker: R_pos=[122.2  -9.6 177.7]mm, L_pos=[-100.2  -10.9  180.8]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]
[Sample 4] Captured marker: R_pos=[1.222e+02 1.000e-01 1.804e+02]mm, L_pos=[-100.2   -1.   183.4]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]
[Sample 5] Captured marker: R_pos=[117.2   6.5 173.4]mm, L_pos=[-95.2   5.4 176.3]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]
[Sample 6] Captured marker: R_pos=[117.1 -16.1 167. ]mm, L_pos=[-95.2 -17.4 169.8]mm
Auto motion done: Restore Baseline Pose
[Sample 7] Captured marker: R_pos=[117.1  -4.9 170.4]mm, L_pos=[-95.2  -6.1 173.4]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 8] Captured marker: R_pos=[120.2  -5.  171.7]mm, L_pos=[-92.2  -6.2 171.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 9] Captured marker: R_pos=[123.3  -5.2 172.8]mm, L_pos=[-89.2  -6.3 170.4]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 10] Captured marker: R_pos=[114.1  -4.9 169.1]mm, L_pos=[-98.3  -6.1 174.7]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 11] Captured marker: R_pos=[111.1  -4.9 167.7]mm, L_pos=[-101.3   -6.2  175.9]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 12] Captured marker: R_pos=[122.7 -11.1 168.5]mm, L_pos=[-100.9  -12.3  171.4]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 13] Captured marker: R_pos=[128.3 -17.  166.3]mm, L_pos=[-106.6  -18.3  169.3]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 14] Captured marker: R_pos=[111.7   1.4 172.3]mm, L_pos=[-89.7   0.3 175.3]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 15] Captured marker: R_pos=[106.3   7.8 174. ]mm, L_pos=[-84.3   6.7 176.9]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 16] Captured marker: R_pos=[119.7  -8.  166.3]mm, L_pos=[-92.8  -3.1 177.5]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 17] Captured marker: R_pos=[122.4 -11.2 162.4]mm, L_pos=[-9.060e+01 -1.000e-01  1.817e+02]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 18] Captured marker: R_pos=[114.7  -1.8 174.6]mm, L_pos=[-97.8  -9.2 169.5]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 19] Captured marker: R_pos=[112.6   1.1 179. ]mm, L_pos=[-100.6  -12.3  165.8]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 20] Captured marker: R_pos=[105.7   1.  175.9]mm, L_pos=[-105.8  -12.5  167.7]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 21] Captured marker: R_pos=[117.2 -11.1 160.2]mm, L_pos=[-97.7  -0.2 184.7]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[119.8   0.8 182. ]mm, L_pos=[-95.4 -12.5 163.3]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[127.8 -11.5 164.4]mm, L_pos=[-83.8  -0.4 178.4]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 24] Captured marker: R_pos=[ 99.4   7.2 170.2]mm, L_pos=[-91.4   7.  180.4]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Joint 2-4 Decouple (+3.0,-3.0)deg
[Sample 25] Captured marker: R_pos=[109.4   3.8 169.4]mm, L_pos=[-81.5  10.6 181.7]mm
Auto motion done: Joint 2-4 Decouple (-3.0,+3.0)deg
[Sample 26] Captured marker: R_pos=[125.4 -13.7 171.5]mm, L_pos=[-109.7  -21.8  164.4]mm
Auto motion done: Restore Baseline Pose
[Sample 27] Captured marker: R_pos=[117.1  -4.9 170.4]mm, L_pos=[-95.3  -6.1 173.4]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 28] Captured marker: R_pos=[122.5  -5.3 179.9]mm, L_pos=[-100.6   -6.6  183. ]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 29] Captured marker: R_pos=[128.1  -5.4 189.4]mm, L_pos=[-106.2   -6.8  192.6]mm
Auto motion done: Elbow Flexion Low (J3 -3deg, J5 +3deg)
[Sample 30] Captured marker: R_pos=[109.4  -3.6 155.8]mm, L_pos=[-87.4  -4.7 158.5]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[113.1   1.7 181.6]mm, L_pos=[-85.3   8.7 194.1]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[105.6  10.4 180.7]mm, L_pos=[-72.1  26.7 201.7]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[117.1  -4.9 170.4]mm, L_pos=[-95.3  -6.1 173.4]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[122.2  -9.5 177.7]mm, L_pos=[-100.   -10.9  180.5]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[122.2  -9.4 177.7]mm, L_pos=[-99.9 -10.9 180.3]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[122.2  -9.8 177.7]mm, L_pos=[-100.4  -10.9  180.9]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[122.   -9.9 177.5]mm, L_pos=[-100.5  -10.8  181. ]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[124.8  -9.4 177.7]mm, L_pos=[-102.8  -10.7  180.7]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[127.5  -9.  178.1]mm, L_pos=[-105.5  -10.5  181. ]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[119.5 -10.  177.6]mm, L_pos=[-97.6 -11.1 180.7]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[116.8 -10.3 177.8]mm, L_pos=[-94.9 -11.3 180.9]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[122.2 -12.7 177.6]mm, L_pos=[-100.1   -8.   180.6]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[122.4 -15.7 177.8]mm, L_pos=[-100.    -5.2  180.6]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[122.1  -6.6 177.8]mm, L_pos=[-100.3  -13.8  180.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[122.   -3.7 178.1]mm, L_pos=[-100.4  -16.6  181.3]mm
Auto motion done: Pos: (-0.030,0.000,0.000)
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Pos: (0.030,0.000,0.000)
[Sample 46] Captured marker: R_pos=[121.2 -13.5 206.1]mm, L_pos=[-99.2 -14.9 209. ]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 47] Captured marker: R_pos=[125.9  -9.4 184.3]mm, L_pos=[-96.4 -11.2 175. ]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 48] Captured marker: R_pos=[129.   -9.3 192. ]mm, L_pos=[-92.3 -11.5 170.3]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 49] Captured marker: R_pos=[118.3 -10.1 171.8]mm, L_pos=[-103.8  -10.8  187.4]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 50] Captured marker: R_pos=[114.  -10.5 166.9]mm, L_pos=[-106.8  -10.7  194.8]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 51] Captured marker: R_pos=[122.5  -6.  174.5]mm, L_pos=[-100.4   -7.3  177.3]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 52] Captured marker: R_pos=[122.7  -2.3 172.3]mm, L_pos=[-100.5   -3.8  175. ]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 53] Captured marker: R_pos=[122.1 -13.6 181.5]mm, L_pos=[-100.1  -14.6  184.9]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 54] Captured marker: R_pos=[121.9 -17.5 186.3]mm, L_pos=[-99.8 -18.3 189.7]mm
Auto motion done: Head Pan: -3.50deg
[Sample 55] Captured marker: R_pos=[108.6  -9.7 184. ]mm, L_pos=[-113.8  -11.   173.7]mm
Auto motion done: Head Pan: -1.75deg
[Sample 56] Captured marker: R_pos=[115.5  -9.8 180.8]mm, L_pos=[-107.1  -11.   177.2]mm
Auto motion done: Head Pan: +1.75deg
[Sample 57] Captured marker: R_pos=[129.   -9.9 173.9]mm, L_pos=[-93.4 -10.9 184. ]mm
Auto motion done: Head Pan: +3.50deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Head Tilt: -3.50deg
[Sample 58] Captured marker: R_pos=[122.4   3.9 181.1]mm, L_pos=[-100.2    2.9  184.3]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 59] Captured marker: R_pos=[122.4  -3.  179.4]mm, L_pos=[-100.3   -4.   182.6]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 60] Captured marker: R_pos=[122.3 -16.6 175.3]mm, L_pos=[-100.3  -17.8  178.6]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 61] Captured marker: R_pos=[122.3 -23.3 173. ]mm, L_pos=[-100.3  -24.6  176.1]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260828_065350.npz
Auto motions sequence completed.
[Step2] Calculate requested.

[Step2] Calculate requested (Active Mode: 'live').
[Step2] Using live recorded dataset (61 samples in memory).
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [np.float64(0.06110506099909379), np.float64(0.029999999999999787), np.float64(-0.0037184528946057273), np.float64(90.70803789986162), np.float64(-2.6978691977018348), np.float64(-89.6592595447885)]
[INFO] Using calibrated marker bracket values for left: [np.float64(0.05911930895015072), np.float64(0.029999999999999492), np.float64(-0.00413554907432049), np.float64(89.82037331216189), np.float64(0.16628920204652753), np.float64(-89.64469322737098)]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.2039956272497325, 'joint5': 1.8687617405465176, 'joint6': -1.3008088869818764}, 'left': {'joint3': -1.916072094322157, 'joint5': 0.23595425020634764, 'joint6': 1.5559680501386643}}

[INFO] === SEQUENTIAL 3-STAGE JOINT-CAMERA CALIBRATION WORKFLOW ===
[STAGE 1/3] Right Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 2/3] Left Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 3/3] Dual-Arm Unified Fine Integration (Anchored Head/Camera, max_iter=50, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=2deg, sigma_pos=1mm
Right arm joint offset (deg): [-2.39727476 -2.39053781 -5.0613798   2.25399563 -0.20420342 -1.81876174
  1.26866605]
Left arm joint offset (deg): [-0.28554234  2.55987901 -8.17286747  1.86607209 11.02543744 -0.28595425
 -1.60596805]
Head joint offset (deg): [-3.16166872  0.38538884]
mount_to_cam xi: [ 2.03401884e-04 -2.43552495e-04  2.41649325e-03  4.99999960e-03
 -1.19363446e-08 -5.79634805e-09]
mount_to_cam_new: [0.0418267047712819, 0.004790008953262153, 0.06237531464845218, -90.05836055973687, 0.008470409519253945, -89.91620056552534]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260828_065350.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -2.3973° | Baseline =  +0.0476° | Diff = 2.4449°
   J1: Calc =  -2.3905° | Baseline =  +2.4544° | Diff = 4.8449°
   J2: Calc =  -5.0614° | Baseline =  -0.0004° | Diff = 5.0609°
   J3: Calc =  +2.2540° | Baseline =  +2.0550° | Diff = 0.1990°
   J4: Calc =  -0.2042° | Baseline =  +0.0007° | Diff = 0.2049°
   J5: Calc =  -1.8188° | Baseline =  +0.0125° | Diff = 1.8313°
   J6: Calc =  +1.2687° | Baseline =  +0.0002° | Diff = 1.2684°
 [LEFT ARM]
   J0: Calc =  -0.2855° | Baseline =  +0.0550° | Diff = 0.3406°
   J1: Calc =  +2.5599° | Baseline =  -2.7585° | Diff = 5.3184°
   J2: Calc =  -8.1729° | Baseline =  +0.0002° | Diff = 8.1731°
   J3: Calc =  +1.8661° | Baseline =  +1.9499° | Diff = 0.0838°
   J4: Calc = +11.0254° | Baseline =  +0.0169° | Diff = 11.0085°
   J5: Calc =  -0.2860° | Baseline =  +0.0723° | Diff = 0.3582°
   J6: Calc =  -1.6060° | Baseline =  -0.0363° | Diff = 1.5697°
=========================================================

Optimization finished successfully.
[Step2] Apply Home Offset requested.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260828_065350.json
Arm: both
Right move offset (deg): [2.397274761869728, 2.390537813488805, 5.061379801601944, -2.2539956272107258, 0.2042034204418966, 1.8187617405719252, -1.2686660528542002]
Left move offset (deg): [0.28554234048266003, -2.5598790091408277, 8.172867470074282, -1.866072094508674, -11.025437442581481, 0.28595425000369185, 1.605968050119179]
Head move offset (deg): [3.1616687230058056, -0.38538884427615994]
Preview move complete. Inspect the robot pose before applying.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Baseline Check Position =====
JSON: /home/nvidia/camera_ws/config/home_reset_baseline.json
Arm: both
Right move offset (deg): [-0.047643680383663366, -2.454410968440594, 0.00043510210396039605, -2.054987237004951, -0.0006591796874999999, -0.0125244140625, -0.00021972656249999998]
Left move offset (deg): [-0.0550404161509901, 2.7585473391089117, -0.00021755105198019802, -1.949910078898515, -0.016918945312499996, -0.0722900390625, 0.0362548828125]
Head move offset (deg): [-0.0, -0.00021972656249999998]
Preview move complete. Inspect the robot pose before applying.
