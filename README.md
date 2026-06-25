[INFO] Starting Joint Sweep: WRIST_PITCH_V13

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: WRIST_PITCH_V13
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 68 dense raw coordinate frames during Joint A motion... downsampled to 68 for optimization.
Swept 68 dense raw coordinate frames during Joint B motion... downsampled to 68 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration]
    perp_dist(δ=0)   = 10.4867 mm  (c_B ~ Joint6 axis, before)
    solved δ         = 2.1546°  →  applied offset = -2.1546°
    perp_dist(δ_opt) = 9.2787 mm  (after)
  [Bracket Design Verification]
    r_A  (Joint6 sweep radius, lateral offset from axis) = 132.194 mm
    axial offset (c_B along Joint6 axis)                 = -91.958 mm
  * Angle Error (Deviation)     : 89.1547°
  * Circle Size Error (r_A-r_B) : 29.4284 mm
  * Center Distance Error       : 999.0000 mm
  * Max Fitting Error Metric    : 999.0000 mm
  * Updated Absolute Offset     : -2.1546°

[ITERATION 2/8] Sweeping with staged offset -2.1546°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -2.1546°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 69 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 66 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 69 dense raw coordinate frames during Joint A motion... downsampled to 69 for optimization.
Swept 66 dense raw coordinate frames during Joint B motion... downsampled to 66 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration]
    perp_dist(δ=0)   = 12.8415 mm  (c_B ~ Joint6 axis, before)
    solved δ         = 4.3441°  →  applied offset = -4.3441°
    perp_dist(δ_opt) = 8.2375 mm  (after)
  [Bracket Design Verification]
    r_A  (Joint6 sweep radius, lateral offset from axis) = 132.028 mm
    axial offset (c_B along Joint6 axis)                 = -93.839 mm
  * Angle Error (Deviation)     : 89.1535°
  * Circle Size Error (r_A-r_B) : 29.9678 mm
  * Center Distance Error       : 999.0000 mm
  * Max Fitting Error Metric    : 999.0000 mm
  * Updated Absolute Offset     : -6.4987°

[ITERATION 3/8] Sweeping with staged offset -6.4987°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -6.4987°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 69 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
[STOP] Stop requested by user.
[STOP] Sending cancel_control to robot!
[ERROR] Joint B sweep motion failed or was cancelled.
[ERROR] Iteration 3 sweep failed. Aborting calibration.
