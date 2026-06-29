[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING SEQUENTIAL CALIBRATION FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for right arm...
[FULL AUTO] Moving right arm to ready pose...
[INFO] Moving right arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 4 (duration=15s)...
    -> Swept 177 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 6 (duration=15s)...
    -> Swept 174 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 5 (duration=15s)...
    -> Swept 174 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...

[FULL AUTO] Calibrating J6 (Wrist Roll) first...

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration — Direct Geometric Method]
    J6 encoder->plane slope     : 1.0007 (Ideal: 1.0)
    J6 encoder=0° Ref Angle     : -57.5374°
    J6 Design Nominal Angle      : -98.6961°
    ★ J6 Calibration Angle       : -41.1587°
    Uncertainty (±1σ)            : ±0.0042°
  [Reference: Old perp_dist Method]
    perp_dist(c_B to J6 axis)   : 20.7071 mm (Reference Only)
  [Bracket Design Verification]
    r_A (Joint6 sweep radius, lateral offset from axis) = 132.049 mm
    r_B (Joint5 sweep radius, total marker-J5 distance) = 162.280 mm
    axial offset (c_B along Joint6 axis)                 = -91.976 mm
  [Marker Design Values (Back-calculated)]
    Geometric Assumptions: J6 roll axis = ee_right X, J5 pitch axis = ee_right Y
    z_marker (= r_A, J6 axis perp. dist) : 132.049 mm
    x_marker (= sqrt(r_B²-r_A²))         : 94.328 mm  * Excludes URDF J6->ee offset
    |axial_offset| (c_A->c_B axial)      : 91.976 mm
  [Geometric Consistency Verification]
    r_B Predicted (sqrt(r_A²+axial²))    : 160.924 mm
    r_B Measured                         : 162.280 mm
    Error                                : 1.356 mm (0.8%)  ✓ OK
[ERROR] Full Auto sequential calibration failed: 'recommended_joint_offset'
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/additional_calib_ui.py", line 800, in run
    opt_roll = joint_res_roll["recommended_joint_offset"]
KeyError: 'recommended_joint_offset'

[INFO] Full Auto sequential calibration ended.
