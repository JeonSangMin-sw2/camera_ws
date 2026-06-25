[INFO] Starting Joint Sweep: WRIST_PITCH_V13

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: WRIST_PITCH_V13
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0002°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 0.0002°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 69 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 68 dense raw coordinate frames during Joint A motion... downsampled to 68 for optimization.
Swept 69 dense raw coordinate frames during Joint B motion... downsampled to 69 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [J5 center-match cross-check scan]
    delta= -10.0deg -> |c_FK-c_cam| = 14.184 mm <-min
    delta=  -5.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  -3.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  -1.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  +0.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  +1.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  +3.0deg -> |c_FK-c_cam| = 14.184 mm
    delta=  +5.0deg -> |c_FK-c_cam| = 14.184 mm
    delta= +10.0deg -> |c_FK-c_cam| = 14.184 mm
  [v1.3 Joint 5 Calibration  — DIRECT vector method]
    reference frames used        : 6 (±2° window)
    J5 offset (direct, primary)  : 49.6574deg
    J5 offset (scan minimum)     : -10.0deg  (dist=14.184mm)
  [Marker Design Values from Sweep A reference frame]
    r_A (sweep-fit)              : 161.464 mm
    r_A (direct, vec|cam-perp|)  : 348.508 mm
    axial offset along J5 axis   : -29.871 mm  (sweep B: -9.752 mm)
  * Angle Error (Deviation from 90°) : 88.7355°
  * Perpendicular Distance (After)   : 14.1840 mm
  * Perpendicular Distance (Before)  : 14.1836 mm
  * Updated Absolute Offset     : 49.6577°

[ITERATION 2/8] Sweeping with staged offset 49.6577°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 49.6577°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
[ERROR] Joint A sweep motion failed or was cancelled.
[ERROR] Iteration 2 sweep failed. Aborting calibration.
