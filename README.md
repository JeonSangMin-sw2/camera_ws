[INFO] Starting Joint Sweep: WRIST_ROLL_V13

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: WRIST_ROLL_V13
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 91 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 88 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 6 debug points to sweep_points_right_joint_A_axis_6.txt
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_B_axis_5.txt
Swept 91 dense raw coordinate frames during Joint A motion... downsampled to 91 for optimization.
Swept 88 dense raw coordinate frames during Joint B motion... downsampled to 88 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration — Direct Geometric Method]
    J6 encoder->plane slope     : 1.0010 (Ideal: 1.0)
    J6 encoder=0° Ref Angle     : -37.7726°
    J6 Design Nominal Angle      : -45.3716°
    ★ J6 Calibration Angle       : -7.5991°
    Uncertainty (±1σ)            : ±0.0152°
  [Reference: Old perp_dist Method]
    perp_dist(c_B to J6 axis)   : 4.7043 mm (Reference Only)
  [Bracket Design Verification]
    r_A (Joint6 sweep radius, lateral offset from axis) = 132.242 mm
    r_B (Joint5 sweep radius, total marker-J5 distance) = 161.586 mm
    axial offset (c_B along Joint6 axis)                 = -92.561 mm
  [Marker Design Values (Back-calculated)]
    Geometric Assumptions: J6 roll axis = ee_right X, J5 pitch axis = ee_right Y
    z_marker (= r_A, J6 axis perp. dist) : 132.242 mm
    x_marker (= sqrt(r_B²-r_A²))         : 92.855 mm  * Excludes URDF J6->ee offset
    |axial_offset| (c_A->c_B axial)      : 92.561 mm
  [Geometric Consistency Verification]
    r_B Predicted (sqrt(r_A²+axial²))    : 161.417 mm
    r_B Measured                         : 161.586 mm
    Error                                : 0.169 mm (0.1%)  ✓ OK
  * Angle Error (Deviation from 90°) : 0.7691°
  * Perpendicular Distance (After)   : 0.0000 mm
  * Perpendicular Distance (Before)  : 4.7043 mm
  * Updated Absolute Offset     : -5.0000°

[SUCCESS] Calibration CONVERGED successfully:
  * Circle Normals Angle Error: 0.7691° <= 0.1°
  * Center Distance Error: 0.0000 mm <= 0.1 mm
  * Recommended Absolute Offset: -5.0000°

[VALIDATION SWEEP] Running final validation sweep with recommended offset -5.0000°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -5.0000°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 87 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 88 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 87 dense raw coordinate frames during Joint A motion... downsampled to 87 for optimization.
Swept 88 dense raw coordinate frames during Joint B motion... downsampled to 88 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration — Direct Geometric Method]
    J6 encoder->plane slope     : 1.0008 (Ideal: 1.0)
    J6 encoder=0° Ref Angle     : -37.7374°
    J6 Design Nominal Angle      : -45.2786°
    ★ J6 Calibration Angle       : -7.5412°
    Uncertainty (±1σ)            : ±0.0146°
  [Reference: Old perp_dist Method]
    perp_dist(c_B to J6 axis)   : 16.2992 mm (Reference Only)
  [Bracket Design Verification]
    r_A (Joint6 sweep radius, lateral offset from axis) = 132.431 mm
    r_B (Joint5 sweep radius, total marker-J5 distance) = 161.680 mm
    axial offset (c_B along Joint6 axis)                 = -94.340 mm
  [Marker Design Values (Back-calculated)]
    Geometric Assumptions: J6 roll axis = ee_right X, J5 pitch axis = ee_right Y
    z_marker (= r_A, J6 axis perp. dist) : 132.431 mm
    x_marker (= sqrt(r_B²-r_A²))         : 92.750 mm  * Excludes URDF J6->ee offset
    |axial_offset| (c_A->c_B axial)      : 94.340 mm
  [Geometric Consistency Verification]
    r_B Predicted (sqrt(r_A²+axial²))    : 162.598 mm
    r_B Measured                         : 161.680 mm
    Error                                : 0.917 mm (0.6%)  ✓ OK
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png
------------------------------
  [1] Calibration Target: wrist_roll_v13
      Estimated Optimal Offset: -5.000 deg
------------------------------

[CALIBRATION COMPLETE]


==================================================
   [SUCCESS] 3-STEP POLARITY CALIBRATION CONVERGED SUCCESSFULLY!
   * Recommended Absolute Offset : -5.0000°
   * Current Active Offset       : 0.0000° (REVERTED)
   --> Click 'APPLY OFFSET' on the UI panel to apply this new calibration.
==================================================


==================================================
       JOINT CALIBRATION ESTIMATED RESULTS
==================================================

[1] Calibration Target: wrist_roll_v13
    - Target Swept Joint       : Joint 6 (Wrist Roll)
    - Estimated Optimal Offset : -5.0000 deg

[2] Suggested Joint Home Offset update:
  Add offset: -5.0000 deg to calibration config.

[3] Bracket Design Verification (Based on Joint 6 Axis)
    - c_B ~ Joint 6 axis perp. dist (before) : 4.7043 mm
    - c_B ~ Joint 6 axis perp. dist (after)  : 0.0000 mm
    - Sweep A fitting radius (r_A, lateral marker offset) : 132.242 mm
    - Axial marker offset (c_B along Joint 6 axis)  : -92.561 mm
    - Lateral marker offset (c_B perp Joint 6 axis)  : 4.704 mm
    * Design Reference Offset Axis: X-axis
==================================================
