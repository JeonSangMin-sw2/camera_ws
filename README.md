[INFO] Starting Joint Sweep: WRIST_ROLL_V13

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: WRIST_ROLL_V13
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 74 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 72 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 74 dense raw coordinate frames during Joint A motion... downsampled to 74 for optimization.
Swept 72 dense raw coordinate frames during Joint B motion... downsampled to 72 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration] Target r3=443.2698 mm, Predicted r3(0)=438.7574 mm. Solved offset=-0.0000°
  * Angle Error (Deviation)     : 2.1223°
  * Circle Size Error (r_A-r_B) : 281.4538 mm
  * Center Distance Error       : 310.2350 mm
  * Max Fitting Error Metric    : 310.2350 mm
  * Updated Absolute Offset     : -0.0000°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: -0.0000° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: -0.0000°

[VALIDATION SWEEP] Running final validation sweep with recommended offset -0.0000°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -0.0000°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 70 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 74 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 70 dense raw coordinate frames during Joint A motion... downsampled to 70 for optimization.
Swept 74 dense raw coordinate frames during Joint B motion... downsampled to 74 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration] Target r3=442.5298 mm, Predicted r3(0)=438.9764 mm. Solved offset=2.6042°
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png
------------------------------
  [1] Calibration Target: wrist_roll_v13
      Estimated Optimal Offset: -0.000 deg
------------------------------

[CALIBRATION COMPLETE]

      [INFO] Detailed 2x2 comparison plot already exists: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png

==================================================
   [SUCCESS] 3-STEP POLARITY CALIBRATION CONVERGED SUCCESSFULLY!
   * Recommended Absolute Offset : -0.0000°
   * Current Active Offset       : 0.0000° (REVERTED)
   --> Click 'APPLY OFFSET' on the UI panel to apply this new calibration.
==================================================


==================================================
       JOINT CALIBRATION ESTIMATED RESULTS
==================================================

[1] Calibration Target: wrist_roll_v13
    - Target Swept Joint       : Joint 5 (Wrist Pitch)
    - Estimated Optimal Offset : -0.0000 deg

[2] Suggested Joint Home Offset update:
  Add offset: -0.0000 deg to calibration config.
