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

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for right arm (Pass 1/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: +1.1533°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: -3.1359°
[FULL AUTO] [Phase 1] Orthogonality Residual: 1.785°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -3.1359° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 1.1533° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for right arm...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 1.1533°, Joint 6: -3.1359°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving right arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.9830°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0800°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -2.1704°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 5/6] Sweeping physically with staged offset -2.1068°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0487° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -2.1068°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1068° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for RIGHT Arm:
  * Joint 6 Change      : 3.1359°
  * Joint 5 Change      : 1.1533°
  * Joint 3 Change      : 2.1068°
  * Bracket Pos Change  : 0.0000 mm
  * Bracket Rot Change  : 0.8738°
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

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for right arm (Pass 2/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: +1.9049°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: -1.5930°
[FULL AUTO] [Phase 1] Orthogonality Residual: 0.752°
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_roll_v13. Staged: -1.5930° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for RIGHT wrist_pitch_v13. Staged: 1.9049° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for right arm...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 1.9049°, Joint 6: -1.5930°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-2.1068°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: -2.1068° (click APPLY OFFSET to save).
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

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for left arm (Pass 1/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: +0.5533°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: +0.0591°
[FULL AUTO] [Phase 1] Orthogonality Residual: 1.002°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 0.0591° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: 0.5533° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for left arm...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: 0.5533°, Joint 6: 0.0591°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving left arm to elbow Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE

[ITERATION 1/6] Sweeping physically with staged offset 0.0000°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 2/6] Sweeping physically with staged offset -1.8545°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 3/6] Sweeping physically with staged offset -2.0912°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[ITERATION 4/6] Sweeping physically with staged offset -1.9647°...
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0330° < 0.06° (reached resolution limit)
  * Recommended Absolute Offset: -1.9647°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/result/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9647° (click APPLY OFFSET to save).

[PASS 1 EVALUATION] Staged parameter changes for LEFT Arm:
  * Joint 6 Change      : 0.0591°
  * Joint 5 Change      : 0.5533°
  * Joint 3 Change      : 1.9647°
  * Bracket Pos Change  : 0.0000 mm
  * Bracket Rot Change  : 0.2717°
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

[FULL AUTO] [Phase 1] Computing 5·6-Axis Orthogonality Solution for left arm (Pass 2/2)...
[FULL AUTO] [Phase 1] Staged Joint 5 (Pitch) Offset: +1.4661°
[FULL AUTO] [Phase 1] Staged Joint 6 (Roll)  Offset: +1.3035°
[FULL AUTO] [Phase 1] Orthogonality Residual: 2.449°
[INFO] Full Auto: Finished joint calibration for LEFT wrist_roll_v13. Staged: 1.3035° (click APPLY OFFSET to save).
[INFO] Full Auto: Finished joint calibration for LEFT wrist_pitch_v13. Staged: 1.4661° (click APPLY OFFSET to save).
[FULL AUTO] [Phase 2] Computing Pure Marker Bracket Transform for left arm...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: 1.4661°, Joint 6: 1.3035°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO] J3 (Elbow) converged in Pass 1 (-1.9647°). Skipping Pass 2 sweep.
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: -1.9647° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[CALIB REPORT] Final Calibrated Offsets (Relative to Nominal Design):
  --- RIGHT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.0, Z: +0.0 mm
  * Bracket Rot: R: +0.00, P: +0.00, Y: -0.11 deg
  * Joint Offsets: Joint 6: -1.59°, Joint 5: +1.90°, Joint 3: -2.11°
  --- LEFT ARM ---
  * Bracket Pos: X: +0.0, Y: +0.0, Z: +0.0 mm
  * Bracket Rot: R: +0.00, P: +0.00, Y: -0.19 deg
  * Joint Offsets: Joint 6: +1.30°, Joint 5: +1.47°, Joint 3: -1.96°
==================================================

[INFO] Full Auto sequential calibration ended.
[SUCCESS] Full Auto Sequential Calibration completed successfully! Please review the offsets in the table.
[SUCCESS] Saved offsets permanently to setting.yaml!

==================================================
[APPLY] Applied current staged joint offsets for BOTH arms:
  --- LEFT ARM ---
    * Joint 6 (Wrist Roll) : 1.3035°
    * Joint 5 (Wrist Pitch): 1.4661°
    * Joint 3 (Elbow)      : -1.9647°
  --- RIGHT ARM ---
    * Joint 6 (Wrist Roll) : -1.5930°
    * Joint 5 (Wrist Pitch): 1.9049°
    * Joint 3 (Elbow)      : -2.1068°
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
Auto base head pose (deg): [-0.  0.]
[Step2] Auto Motion requested.
Motion plan is missing or empty. Re-building...
Auto Motion started in a background thread. Press Stop to cancel.
Building motion plan based on current pose... (Angle=5.0deg, Pos=0.03m, StepX=0.03m, MaxX=0.4m)
Auto motion done: J0 (-3.5deg) + Head Tilt (-5.0deg) [Center FOV]
[Sample 1] Captured marker: R_pos=[117.1  -4.9 170.5]mm, L_pos=[-95.3  -6.1 173.4]mm
Auto motion done: J0 (-3.5deg) + Head Tilt (-2.5deg) [Upper FOV]
[Sample 2] Captured marker: R_pos=[112.4 -10.2 160.6]mm, L_pos=[-90.6 -11.4 163.2]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+5.0deg) [Center FOV]
[Sample 3] Captured marker: R_pos=[122.1  -9.7 177.7]mm, L_pos=[-100.2  -10.9  180.6]mm
Auto motion done: J0 (+3.5deg) + Head Tilt (+2.5deg) [Lower FOV]
[Sample 4] Captured marker: R_pos=[1.222e+02 1.000e-01 1.804e+02]mm, L_pos=[-100.2   -1.   183.4]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (-3.0deg) [Upper FOV]
[Sample 5] Captured marker: R_pos=[117.2   6.5 173.5]mm, L_pos=[-95.2   5.4 176.3]mm
Auto motion done: J0 ( 0.0deg) + Head Tilt (+3.0deg) [Lower FOV]
[Sample 6] Captured marker: R_pos=[117.1 -16.1 167.1]mm, L_pos=[-95.2 -17.4 169.7]mm
Auto motion done: Restore Baseline Pose
[Sample 7] Captured marker: R_pos=[117.1  -4.9 170.5]mm, L_pos=[-95.2  -6.1 173.4]mm
Auto motion done: Joint 1 Offset: -2.5deg
[Sample 8] Captured marker: R_pos=[120.2  -5.1 171.7]mm, L_pos=[-92.2  -6.2 171.9]mm
Auto motion done: Joint 1 Offset: -5.0deg
[Sample 9] Captured marker: R_pos=[123.3  -5.3 172.8]mm, L_pos=[-89.2  -6.4 170.4]mm
Auto motion done: Joint 1 Offset: +2.5deg
[Sample 10] Captured marker: R_pos=[114.1  -4.9 169.1]mm, L_pos=[-98.3  -6.1 174.7]mm
Auto motion done: Joint 1 Offset: +5.0deg
[Sample 11] Captured marker: R_pos=[111.   -5.  167.7]mm, L_pos=[-101.3   -6.2  175.8]mm
Auto motion done: Joint 2 Offset: -1.5deg
[Sample 12] Captured marker: R_pos=[122.7 -11.1 168.4]mm, L_pos=[-100.9  -12.3  171.4]mm
Auto motion done: Joint 2 Offset: -3.0deg
[Sample 13] Captured marker: R_pos=[128.3 -17.1 166.3]mm, L_pos=[-106.6  -18.4  169.2]mm
Auto motion done: Joint 2 Offset: +1.5deg
[Sample 14] Captured marker: R_pos=[111.7   1.3 172.4]mm, L_pos=[-89.7   0.2 175.2]mm
Auto motion done: Joint 2 Offset: +3.0deg
[Sample 15] Captured marker: R_pos=[106.3   7.7 174. ]mm, L_pos=[-84.3   6.7 176.8]mm
Auto motion done: Joint 4 Offset: -2.5deg
[Sample 16] Captured marker: R_pos=[119.7  -8.  166.4]mm, L_pos=[-92.8  -3.1 177.4]mm
Auto motion done: Joint 4 Offset: -5.0deg
[Sample 17] Captured marker: R_pos=[122.4 -11.2 162.4]mm, L_pos=[-9.070e+01 -1.000e-01  1.817e+02]mm
Auto motion done: Joint 4 Offset: +2.5deg
[Sample 18] Captured marker: R_pos=[114.8  -1.9 174.7]mm, L_pos=[-97.8  -9.2 169.4]mm
Auto motion done: Joint 4 Offset: +5.0deg
[Sample 19] Captured marker: R_pos=[112.6   1.1 179.1]mm, L_pos=[-100.5  -12.3  165.7]mm
Auto motion done: Joint 1+4 (+5.0,+5.0)deg
[Sample 20] Captured marker: R_pos=[105.7   0.9 175.9]mm, L_pos=[-105.8  -12.5  167.7]mm
Auto motion done: Joint 1+4 (+5.0,-5.0)deg
[Sample 21] Captured marker: R_pos=[117.2 -11.1 160.2]mm, L_pos=[-97.7  -0.3 184.7]mm
Auto motion done: Joint 1+4 (-5.0,+5.0)deg
[Sample 22] Captured marker: R_pos=[119.8   0.7 182.1]mm, L_pos=[-95.4 -12.5 163.2]mm
Auto motion done: Joint 1+4 (-5.0,-5.0)deg
[Sample 23] Captured marker: R_pos=[127.7 -11.6 164.4]mm, L_pos=[-83.8  -0.5 178.3]mm
Auto motion done: Joint 1+2 (+5.0,+3.0)deg
[Sample 24] Captured marker: R_pos=[ 99.4   7.1 170.3]mm, L_pos=[-91.4   7.  180.5]mm
Auto motion done: Joint 1+2 (-5.0,-3.0)deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Joint 2-4 Decouple (+3.0,-3.0)deg
[Sample 25] Captured marker: R_pos=[109.3   3.7 169.4]mm, L_pos=[-81.5  10.5 181.7]mm
Auto motion done: Joint 2-4 Decouple (-3.0,+3.0)deg
[Sample 26] Captured marker: R_pos=[125.4 -13.7 171.5]mm, L_pos=[-109.7  -21.8  164.4]mm
Auto motion done: Restore Baseline Pose
[Sample 27] Captured marker: R_pos=[117.1  -4.9 170.5]mm, L_pos=[-95.3  -6.2 173.3]mm
Auto motion done: Elbow Extension Low (J3 +2deg, J5 -2deg)
[Sample 28] Captured marker: R_pos=[122.5  -5.4 180. ]mm, L_pos=[-100.7   -6.6  183. ]mm
Auto motion done: Elbow Extension Mid (J3 +4deg, J5 -4deg)
[Sample 29] Captured marker: R_pos=[128.1  -5.5 189.5]mm, L_pos=[-106.2   -6.8  192.6]mm
Auto motion done: Elbow Flexion Low (J3 -3deg, J5 +3deg)
[Sample 30] Captured marker: R_pos=[109.4  -3.7 155.9]mm, L_pos=[-87.5  -4.7 158.5]mm
Auto motion done: Elbow Extension + Outward Yaw (+3deg)
[Sample 31] Captured marker: R_pos=[113.1   1.6 181.6]mm, L_pos=[-85.3   8.7 194.1]mm
Auto motion done: Elbow Extension + Outward Wide Yaw (+6deg)
[Sample 32] Captured marker: R_pos=[105.5  10.4 180.7]mm, L_pos=[-72.1  26.7 201.7]mm
Auto motion done: Restore Baseline Pose
[Sample 33] Captured marker: R_pos=[117.1  -5.  170.5]mm, L_pos=[-95.3  -6.1 173.4]mm
Auto motion done: RPY: (-2.50,0.00,0.00)
[Sample 34] Captured marker: R_pos=[122.2  -9.6 177.7]mm, L_pos=[-100.   -10.9  180.5]mm
Auto motion done: RPY: (-5.00,0.00,0.00)
[Sample 35] Captured marker: R_pos=[122.2  -9.5 177.8]mm, L_pos=[-99.9 -11.  180.3]mm
Auto motion done: RPY: (2.50,0.00,0.00)
[Sample 36] Captured marker: R_pos=[122.2  -9.8 177.7]mm, L_pos=[-100.4  -10.9  180.8]mm
Auto motion done: RPY: (5.00,0.00,0.00)
[Sample 37] Captured marker: R_pos=[122.   -9.9 177.6]mm, L_pos=[-100.5  -10.8  180.9]mm
Auto motion done: RPY: (0.00,-2.50,0.00)
[Sample 38] Captured marker: R_pos=[124.8  -9.4 177.7]mm, L_pos=[-102.8  -10.7  180.6]mm
Auto motion done: RPY: (0.00,-5.00,0.00)
[Sample 39] Captured marker: R_pos=[127.5  -9.1 178.1]mm, L_pos=[-105.5  -10.5  180.9]mm
Auto motion done: RPY: (0.00,2.50,0.00)
[Sample 40] Captured marker: R_pos=[119.4 -10.  177.6]mm, L_pos=[-97.6 -11.1 180.6]mm
Auto motion done: RPY: (0.00,5.00,0.00)
[Sample 41] Captured marker: R_pos=[116.8 -10.3 177.8]mm, L_pos=[-94.9 -11.3 180.8]mm
Auto motion done: RPY: (0.00,0.00,-2.50)
[Sample 42] Captured marker: R_pos=[122.2 -12.7 177.6]mm, L_pos=[-100.1   -8.   180.6]mm
Auto motion done: RPY: (0.00,0.00,-5.00)
[Sample 43] Captured marker: R_pos=[122.3 -15.7 177.8]mm, L_pos=[-100.1   -5.2  180.6]mm
Auto motion done: RPY: (0.00,0.00,2.50)
[Sample 44] Captured marker: R_pos=[122.1  -6.6 177.8]mm, L_pos=[-100.3  -13.8  180.9]mm
Auto motion done: RPY: (0.00,0.00,5.00)
[Sample 45] Captured marker: R_pos=[122.1  -3.7 178.2]mm, L_pos=[-100.4  -16.6  181.2]mm
Auto motion done: Pos: (-0.030,0.000,0.000)
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Pos: (0.030,0.000,0.000)
[Sample 46] Captured marker: R_pos=[121.2 -13.5 206.2]mm, L_pos=[-99.2 -15.  208.9]mm
Auto motion done: Pos: (0.000,-0.015,0.000)
[Sample 47] Captured marker: R_pos=[125.9  -9.5 184.4]mm, L_pos=[-96.4 -11.2 175. ]mm
Auto motion done: Pos: (0.000,-0.030,0.000)
[Sample 48] Captured marker: R_pos=[129.   -9.3 192. ]mm, L_pos=[-92.3 -11.5 170.3]mm
Auto motion done: Pos: (0.000,0.015,0.000)
[Sample 49] Captured marker: R_pos=[118.3 -10.1 171.9]mm, L_pos=[-103.8  -10.8  187.3]mm
Auto motion done: Pos: (0.000,0.030,0.000)
[Sample 50] Captured marker: R_pos=[114.  -10.5 167. ]mm, L_pos=[-106.8  -10.7  194.8]mm
Auto motion done: Pos: (0.000,0.000,-0.015)
[Sample 51] Captured marker: R_pos=[122.5  -6.1 174.6]mm, L_pos=[-100.4   -7.3  177.3]mm
Auto motion done: Pos: (0.000,0.000,-0.030)
[Sample 52] Captured marker: R_pos=[122.6  -2.3 172.4]mm, L_pos=[-100.5   -3.8  175. ]mm
Auto motion done: Pos: (0.000,0.000,0.015)
[Sample 53] Captured marker: R_pos=[122.1 -13.6 181.6]mm, L_pos=[-100.1  -14.6  184.8]mm
Auto motion done: Pos: (0.000,0.000,0.030)
[Sample 54] Captured marker: R_pos=[121.8 -17.5 186.3]mm, L_pos=[-99.8 -18.3 189.6]mm
Auto motion done: Head Pan: -3.50deg
[Sample 55] Captured marker: R_pos=[108.6  -9.7 184.1]mm, L_pos=[-113.8  -11.   173.6]mm
Auto motion done: Head Pan: -1.75deg
[Sample 56] Captured marker: R_pos=[115.5  -9.8 181. ]mm, L_pos=[-107.1  -11.   177.2]mm
Auto motion done: Head Pan: +1.75deg
[Sample 57] Captured marker: R_pos=[129.   -9.9 174. ]mm, L_pos=[-93.4 -10.9 183.9]mm
Auto motion done: Head Pan: +3.50deg
Marker not detected.
Capture failed after motion. This pose is skipped.
[WARNING] Step capture failed (1/3). Skipping this pose...
Auto motion done: Head Tilt: -3.50deg
[Sample 58] Captured marker: R_pos=[122.4   3.9 181.2]mm, L_pos=[-100.2    2.9  184.3]mm
Auto motion done: Head Tilt: -1.75deg
[Sample 59] Captured marker: R_pos=[122.3  -3.  179.5]mm, L_pos=[-100.2   -4.   182.6]mm
Auto motion done: Head Tilt: +1.75deg
[Sample 60] Captured marker: R_pos=[122.3 -16.7 175.4]mm, L_pos=[-100.3  -17.8  178.5]mm
Auto motion done: Head Tilt: +3.50deg
[Sample 61] Captured marker: R_pos=[122.2 -23.3 173.1]mm, L_pos=[-100.3  -24.6  176.1]mm
Auto motions completed.
[Auto-Save] Dataset saved/updated in: /home/nvidia/camera_ws/result/result_step2/dataset_20260828_091054.npz
Auto motions sequence completed.
[Step2] Calculate requested.

[Step2] Calculate requested (Active Mode: 'live').
[Step2] Using live recorded dataset (61 samples in memory).
[Step2] Optimization calculation started in background thread...
[INFO] Using calibrated marker bracket values for right: [0.067, 0.0, 0.0, 90.0, 0.0, -90.11461075180154]
[INFO] Using calibrated marker bracket values for left: [0.067, 0.0, 0.0, 90.0, 0.0, -90.18934878410286]
[INFO] Applying joint offset bounds: {'right': {'joint3': -2.1068183791715804, 'joint5': 1.9049322974313867, 'joint6': -1.5930296967096047}, 'left': {'joint3': -1.9647243093149314, 'joint5': 1.4660861375750045, 'joint6': 1.3034550154426687}}

[INFO] === SEQUENTIAL 3-STAGE JOINT-CAMERA CALIBRATION WORKFLOW ===
[STAGE 1/3] Right Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 2/3] Left Arm + Head + Camera Alignment (max_iter=50, eps=1e-7)...
[STAGE 3/3] Dual-Arm Unified Fine Integration (Anchored Head/Camera, max_iter=50, eps=1e-7)...

===== RESULT =====
lambda_cam_pos = 1.0
lambda_cam_rot = 1000000.0
measurement_noise = sigma_rot=1.264deg, sigma_pos=0.1784mm
Right arm joint offset (deg): [-0.32655893  2.10926192  0.39731291  2.06919632  0.34359455 -1.90832759
  1.59580931]
Left arm joint offset (deg): [ 0.09361254 -2.8673015   0.3323095   2.01472424 -1.68860978 -1.41608638
 -1.25345521]
Head joint offset (deg): [0.00198428 0.01648143]
mount_to_cam xi: [ 0.00155294  0.00056691  0.00063931 -0.00131015  0.005       0.0002363 ]
mount_to_cam_new: [0.04205285818238152, 0.011090616917538362, 0.057367353210168955, -89.98095967728673, -0.09338485338775401, -89.96249805440384]
Result saved to /home/nvidia/camera_ws/result/result_step2/result_20260828_091054.json
History appended to /home/nvidia/camera_ws/result/result_step2/calibration_history.txt

=========================================================
  BASE LINE COMPARISON (config/home_reset_baseline.json)
=========================================================
 [RIGHT ARM]
   J0: Calc =  -0.3266° | Baseline =  +0.0476° | Diff = 0.3742°
   J1: Calc =  +2.1093° | Baseline =  +2.4544° | Diff = 0.3451°
   J2: Calc =  +0.3973° | Baseline =  -0.0004° | Diff = 0.3977°
   J3: Calc =  +2.0692° | Baseline =  +2.0550° | Diff = 0.0142°
   J4: Calc =  +0.3436° | Baseline =  +0.0007° | Diff = 0.3429°
   J5: Calc =  -1.9083° | Baseline =  +0.0125° | Diff = 1.9209°
   J6: Calc =  +1.5958° | Baseline =  +0.0002° | Diff = 1.5956°
 [LEFT ARM]
   J0: Calc =  +0.0936° | Baseline =  +0.0550° | Diff = 0.0386°
   J1: Calc =  -2.8673° | Baseline =  -2.7585° | Diff = 0.1088°
   J2: Calc =  +0.3323° | Baseline =  +0.0002° | Diff = 0.3321°
   J3: Calc =  +2.0147° | Baseline =  +1.9499° | Diff = 0.0648°
   J4: Calc =  -1.6886° | Baseline =  +0.0169° | Diff = 1.7055°
   J5: Calc =  -1.4161° | Baseline =  +0.0723° | Diff = 1.4884°
   J6: Calc =  -1.2535° | Baseline =  -0.0363° | Diff = 1.2172°
=========================================================

Optimization finished successfully.

[Check Position] Step 1: Moving to Joint Ready Pose...
[Check Position] Step 2: Moving to Check Pose with Offsets...

===== HOME OFFSET PREVIEW: Optimized Check Position =====
JSON: /home/nvidia/camera_ws/result/result_step2/result_20260828_091054.json
Arm: both
Right move offset (deg): [0.32655892685678745, -2.1092619159508, -0.3973129051829935, -2.0691963209157613, -0.343594550857574, 1.908327592469066, -1.5958093086945628]
Left move offset (deg): [-0.09361254472284046, 2.867301501753658, -0.3323095012826546, -2.014724240489763, 1.688609777985842, 1.4160863824711833, 1.2534552093974558]
Head move offset (deg): [-0.0019842793360661825, -0.01648143156533191]
Preview move complete. Inspect the robot pose before applying.
