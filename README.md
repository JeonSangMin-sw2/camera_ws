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
    -> Swept 75 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 72 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 75 dense raw coordinate frames during Joint A motion... downsampled to 75 for optimization.
Swept 72 dense raw coordinate frames during Joint B motion... downsampled to 72 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration] Target r3=442.2598 mm, Predicted r3(0)=438.4784 mm. Solved offset=2.7790°
  * Angle Error (Deviation)     : 2.0075°
  * Forearm Length (Center Dist): 308.9600 mm
  * Radii Difference (r3 - r5)  : 280.0059 mm
  * Updated Absolute Offset     : 2.7790°

[ITERATION 2/8] Sweeping with staged offset 2.7790°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 2.7790°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 78 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 73 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 78 dense raw coordinate frames during Joint A motion... downsampled to 78 for optimization.
Swept 73 dense raw coordinate frames during Joint B motion... downsampled to 73 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration] Target r3=446.2586 mm, Predicted r3(0)=442.6788 mm. Solved offset=2.8119°
  * Angle Error (Deviation)     : 1.8470°
  * Forearm Length (Center Dist): 309.6785 mm
  * Radii Difference (r3 - r5)  : 284.6535 mm
  * Updated Absolute Offset     : 5.5910°

[ITERATION 3/8] Sweeping with staged offset 5.5910°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 5.5910°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 75 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 79 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 75 dense raw coordinate frames during Joint A motion... downsampled to 75 for optimization.
Swept 79 dense raw coordinate frames during Joint B motion... downsampled to 79 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration] Target r3=448.9210 mm, Predicted r3(0)=446.0238 mm. Solved offset=2.4159°
  * Angle Error (Deviation)     : 1.2201°
  * Forearm Length (Center Dist): 308.7725 mm
  * Radii Difference (r3 - r5)  : 286.9389 mm
  * Updated Absolute Offset     : 8.0068°

[ITERATION 4/8] Sweeping with staged offset 8.0068°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 8.0068°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 72 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
[STOP] Stop requested by user.
[STOP] Sending cancel_control to robot!
[ERROR] Joint B sweep motion failed or was cancelled.
[ERROR] Iteration 4 sweep failed. Aborting calibration.
