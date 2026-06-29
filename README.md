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
[INFO] Active joint 14 limits: min=-180.00°, max=180.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 4 (duration=15s)...
    -> Swept 98 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Active joint 16 limits: min=-90.00°, max=90.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 6 (duration=15s)...
    -> Swept 92 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Active joint 15 limits: min=-50.00°, max=50.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 5 (duration=15s)...
    -> Swept 90 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Computing unified marker bracket calibration...
[INFO] Full Auto: Staged joint offsets for RIGHT Arm - Joint 5: 15.0229°, Joint 6: -17.9889°
[INFO] Full Auto: Finished bracket calibration for RIGHT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO 2/2] Starting Joint Calibration for right arm...
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving to right Pitch/Head Ready Pose (Mode: elbow)...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: ELBOW
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 119 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 116 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 119 dense raw coordinate frames during Joint A motion... downsampled to 119 for optimization.
Swept 116 dense raw coordinate frames during Joint B motion... downsampled to 116 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2208  0.2327 -0.9472]
  [DEBUG] Physically aligned n_B: [ 0.2589  0.2413 -0.9353]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0087, actual=-0.6280
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.6367
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.6367°. Applying damped correction step: 0.6049°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 2.3400 deg
  * Circle Sizes (r_A / r_B)               : 164.93 / 163.56 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.3750 mm
  * Estimated Circle Center Distance       : 4.2979 mm
  * Calculated Offset Correction           : 0.604874 deg
==================================================
  * Angle Error (Deviation)     : 2.3400°
  * Circle Size Error (r_A-r_B) : 1.3750 mm
  * Center Distance Error       : 4.2979 mm
  * Max Fitting Error Metric    : 4.2979 mm
  [SAFETY WARNING] Staged offset 0.6049° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 2/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 122 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 122 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 122 dense raw coordinate frames during Joint A motion... downsampled to 122 for optimization.
Swept 122 dense raw coordinate frames during Joint B motion... downsampled to 122 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2437  0.2317 -0.9418]
  [DEBUG] Physically aligned n_B: [ 0.245   0.2448 -0.9381]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0087, actual=-0.7795
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.7882
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.7882°. Applying damped correction step: 0.7488°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 0.7838 deg
  * Circle Sizes (r_A / r_B)               : 164.96 / 163.36 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.5950 mm
  * Estimated Circle Center Distance       : 4.4467 mm
  * Calculated Offset Correction           : 0.748798 deg
==================================================
  * Angle Error (Deviation)     : 0.7838°
  * Circle Size Error (r_A-r_B) : 1.5950 mm
  * Center Distance Error       : 4.4467 mm
  * Max Fitting Error Metric    : 4.4467 mm
  [SAFETY WARNING] Staged offset 0.7488° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 3/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 117 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 118 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 117 dense raw coordinate frames during Joint A motion... downsampled to 117 for optimization.
Swept 118 dense raw coordinate frames during Joint B motion... downsampled to 118 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2168  0.2347 -0.9476]
  [DEBUG] Physically aligned n_B: [ 0.2337  0.247  -0.9404]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0085, actual=-0.7770
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.7854
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.7854°. Applying damped correction step: 0.7462°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.2652 deg
  * Circle Sizes (r_A / r_B)               : 165.27 / 163.90 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.3672 mm
  * Estimated Circle Center Distance       : 4.3726 mm
  * Calculated Offset Correction           : 0.746173 deg
==================================================
  * Angle Error (Deviation)     : 1.2652°
  * Circle Size Error (r_A-r_B) : 1.3672 mm
  * Center Distance Error       : 4.3726 mm
  * Max Fitting Error Metric    : 4.3726 mm
  [SAFETY WARNING] Staged offset 0.7462° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 4/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 119 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 118 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 119 dense raw coordinate frames during Joint A motion... downsampled to 119 for optimization.
Swept 118 dense raw coordinate frames during Joint B motion... downsampled to 118 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2726  0.2278 -0.9348]
  [DEBUG] Physically aligned n_B: [ 0.2248  0.2457 -0.9429]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.8902
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.8985
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.8985°. Applying damped correction step: 0.8535°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 2.9583 deg
  * Circle Sizes (r_A / r_B)               : 164.89 / 163.63 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.2586 mm
  * Estimated Circle Center Distance       : 4.4226 mm
  * Calculated Offset Correction           : 0.853534 deg
==================================================
  * Angle Error (Deviation)     : 2.9583°
  * Circle Size Error (r_A-r_B) : 1.2586 mm
  * Center Distance Error       : 4.4226 mm
  * Max Fitting Error Metric    : 4.4226 mm
  [SAFETY WARNING] Staged offset 0.8535° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 5/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 115 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 114 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 115 dense raw coordinate frames during Joint A motion... downsampled to 115 for optimization.
Swept 114 dense raw coordinate frames during Joint B motion... downsampled to 114 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2375  0.2305 -0.9436]
  [DEBUG] Physically aligned n_B: [ 0.2404  0.2455 -0.9391]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.8944
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.9027
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.9027°. Applying damped correction step: 0.8576°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 0.9114 deg
  * Circle Sizes (r_A / r_B)               : 165.17 / 163.34 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.8310 mm
  * Estimated Circle Center Distance       : 4.5580 mm
  * Calculated Offset Correction           : 0.857562 deg
==================================================
  * Angle Error (Deviation)     : 0.9114°
  * Circle Size Error (r_A-r_B) : 1.8310 mm
  * Center Distance Error       : 4.5580 mm
  * Max Fitting Error Metric    : 4.5580 mm
  [SAFETY WARNING] Staged offset 0.8576° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 6/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 116 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 120 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 116 dense raw coordinate frames during Joint A motion... downsampled to 116 for optimization.
Swept 120 dense raw coordinate frames during Joint B motion... downsampled to 120 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2515  0.2291 -0.9403]
  [DEBUG] Physically aligned n_B: [ 0.2368  0.243  -0.9407]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.7735
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.7817
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.7817°. Applying damped correction step: 0.7426°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.1589 deg
  * Circle Sizes (r_A / r_B)               : 165.11 / 163.49 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.6256 mm
  * Estimated Circle Center Distance       : 4.4606 mm
  * Calculated Offset Correction           : 0.742636 deg
==================================================
  * Angle Error (Deviation)     : 1.1589°
  * Circle Size Error (r_A-r_B) : 1.6256 mm
  * Center Distance Error       : 4.4606 mm
  * Max Fitting Error Metric    : 4.4606 mm
  [SAFETY WARNING] Staged offset 0.7426° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 7/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 118 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 120 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 118 dense raw coordinate frames during Joint A motion... downsampled to 118 for optimization.
Swept 120 dense raw coordinate frames during Joint B motion... downsampled to 120 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.2527  0.2303 -0.9397]
  [DEBUG] Physically aligned n_B: [ 0.2411  0.2427 -0.9397]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.6948
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.7031
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.7031°. Applying damped correction step: 0.6679°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 0.9733 deg
  * Circle Sizes (r_A / r_B)               : 165.19 / 163.42 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.7676 mm
  * Estimated Circle Center Distance       : 4.4990 mm
  * Calculated Offset Correction           : 0.667920 deg
==================================================
  * Angle Error (Deviation)     : 0.9733°
  * Circle Size Error (r_A-r_B) : 1.7676 mm
  * Center Distance Error       : 4.4990 mm
  * Max Fitting Error Metric    : 4.4990 mm
  [SAFETY WARNING] Staged offset 0.6679° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 8/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 113 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 120 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 113 dense raw coordinate frames during Joint A motion... downsampled to 113 for optimization.
Swept 120 dense raw coordinate frames during Joint B motion... downsampled to 120 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.227   0.2314 -0.946 ]
  [DEBUG] Physically aligned n_B: [ 0.2455  0.2438 -0.9382]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.7881
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.7964
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.7964°. Applying damped correction step: 0.7565°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.3514 deg
  * Circle Sizes (r_A / r_B)               : 164.91 / 163.40 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.5087 mm
  * Estimated Circle Center Distance       : 4.4030 mm
  * Calculated Offset Correction           : 0.756550 deg
==================================================
  * Angle Error (Deviation)     : 1.3514°
  * Circle Size Error (r_A-r_B) : 1.5087 mm
  * Center Distance Error       : 4.4030 mm
  * Max Fitting Error Metric    : 4.4030 mm
  [SAFETY WARNING] Staged offset 0.7565° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[VALIDATION SWEEP] Running final validation sweep with recommended offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 121 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 117 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 121 dense raw coordinate frames during Joint A motion... downsampled to 121 for optimization.
Swept 117 dense raw coordinate frames during Joint B motion... downsampled to 117 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [ 0.237   0.2311 -0.9436]
  [DEBUG] Physically aligned n_B: [ 0.2567  0.2435 -0.9353]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0083, actual=-0.7942
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.8024
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.8024°. Applying damped correction step: 0.7623°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.4164 deg
  * Circle Sizes (r_A / r_B)               : 165.14 / 163.36 mm
  * Circle Size Error (abs: r_A - r_B)     : 1.7839 mm
  * Estimated Circle Center Distance       : 4.5314 mm
  * Calculated Offset Correction           : 0.762321 deg
==================================================
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for RIGHT elbow. Staged: 0.0000° (click APPLY OFFSET to save).
[INFO] RIGHT arm sequential calibration completed successfully.

==================================================
   STARTING SEQUENTIAL CALIBRATION FOR LEFT ARM
==================================================

[INFO] Detected Robot Version: 1.3 (is_v1.3: True)
[FULL AUTO 1/2] Starting Marker Bracket Calibration for left arm...
[FULL AUTO] Moving left arm to ready pose...
[INFO] Moving left arm to marker Ready Pose...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.
[FULL AUTO] Sweeping Axis 4...

==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Active joint 21 limits: min=-180.00°, max=180.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 4 (duration=15s)...
    -> Swept 93 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Returning to Initial Starting Pose...
[FULL AUTO] Sweeping Axis 6...

==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Active joint 23 limits: min=-90.00°, max=90.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 6 (duration=15s)...
    -> Swept 88 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Sweeping Axis 5...

==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Active joint 22 limits: min=-50.00°, max=50.00°
[INFO] Moving to start sweep position...
[INFO] Commencing Continuous Sweep on Marker Axis 5 (duration=15s)...
    -> Swept 89 dense raw coordinate frames.

[INFO] Sweep complete. Returning to initial ready pose...
[FULL AUTO] Computing unified marker bracket calibration...
[INFO] Full Auto: Staged joint offsets for LEFT Arm - Joint 5: -15.0428°, Joint 6: 16.9136°
[INFO] Full Auto: Finished bracket calibration for LEFT arm. Values staged in UI (click APPLY BRACKETS to save).
[FULL AUTO 2/2] Starting Joint Calibration for left arm...
[FULL AUTO] Sweeping Elbow (Joint 3)...
[INFO] Moving to left Pitch/Head Ready Pose (Mode: elbow)...
[INFO] Moving inactive arm to zero pose first...
[INFO] Moving active arm, torso, and head to ready pose...
[INFO] Ready Pose Reached.

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: LEFT | Joint Target: ELBOW
============================================================


[ITERATION 1/8] Sweeping with staged offset -0.1010°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -0.1010°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 117 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 118 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 117 dense raw coordinate frames during Joint A motion... downsampled to 117 for optimization.
Swept 118 dense raw coordinate frames during Joint B motion... downsampled to 118 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [-0.2408  0.2184 -0.9457]
  [DEBUG] Physically aligned n_B: [-0.2573  0.2224 -0.9404]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0080, actual=-0.8456
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.8536
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.8536°. Applying damped correction step: 0.8109°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.0173 deg
  * Circle Sizes (r_A / r_B)               : 163.75 / 163.60 mm
  * Circle Size Error (abs: r_A - r_B)     : 0.1499 mm
  * Estimated Circle Center Distance       : 1.7220 mm
  * Calculated Offset Correction           : 0.810924 deg
==================================================
  * Angle Error (Deviation)     : 1.0173°
  * Circle Size Error (r_A-r_B) : 0.1499 mm
  * Center Distance Error       : 1.7220 mm
  * Max Fitting Error Metric    : 1.7220 mm
  [SAFETY WARNING] Staged offset 0.7099° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[ITERATION 2/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 120 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 122 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 120 dense raw coordinate frames during Joint A motion... downsampled to 120 for optimization.
Swept 122 dense raw coordinate frames during Joint B motion... downsampled to 122 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [-0.2314  0.2236 -0.9468]
  [DEBUG] Physically aligned n_B: [-0.2319  0.2235 -0.9467]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0078, actual=-0.0123
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=-0.0201
  [DEBUG] Mathematically resolved offset direction sign: 1.0
  [ANGLE CONTROL] Using angle-based calibration error: -0.0201°. Applying damped correction step: 0.0191°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 0.0297 deg
  * Circle Sizes (r_A / r_B)               : 164.08 / 163.73 mm
  * Circle Size Error (abs: r_A - r_B)     : 0.3543 mm
  * Estimated Circle Center Distance       : 1.1205 mm
  * Calculated Offset Correction           : 0.019123 deg
==================================================
  * Angle Error (Deviation)     : 0.0297°
  * Circle Size Error (r_A-r_B) : 0.3543 mm
  * Center Distance Error       : 1.1205 mm
  * Max Fitting Error Metric    : 1.1205 mm
  [SAFETY WARNING] Staged offset 0.0191° exceeds safe bounds [-3.0°, 0.0°]. Clamping.
  * Updated Absolute Offset     : 0.0000°

[SUCCESS] Calibration CONVERGED successfully:
  * Step Correction: 0.0191° < 0.05° (reached resolution limit)
  * Recommended Absolute Offset: 0.0000°

[VALIDATION SWEEP] Running final validation sweep with recommended offset 0.0000°...

==================================================
   STARTING ELBOW CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 2, duration=20.0s) ---
    -> Swept 121 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 4, duration=20.0s) ---
    -> Swept 121 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 121 dense raw coordinate frames during Joint A motion... downsampled to 121 for optimization.
Swept 121 dense raw coordinate frames during Joint B motion... downsampled to 121 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [DEBUG] Physically aligned n_A: [-0.2805  0.2116 -0.9362]
  [DEBUG] Physically aligned n_B: [-0.2532  0.2219 -0.9416]
  [DEBUG] Nominal angles in perp plane (deg): nominal=0.0078, actual=0.6465
  [DEBUG] Perpendicular plane angle difference (deg): diff_angle=0.6387
  [DEBUG] Mathematically resolved offset direction sign: -1.0
  [ANGLE CONTROL] Using angle-based calibration error: 0.6387°. Applying damped correction step: -0.6068°

==================================================
   SWEEP ANALYSIS & RESULTS (ELBOW - CONTINUOUS)
==================================================
  * Camera Circle Normals Angle (Reference): 1.7008 deg
  * Circle Sizes (r_A / r_B)               : 163.58 / 163.77 mm
  * Circle Size Error (abs: r_A - r_B)     : 0.1937 mm
  * Estimated Circle Center Distance       : 1.2074 mm
  * Calculated Offset Correction           : -0.606775 deg
==================================================
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_left_elbow_joint_calib.png
[INFO] Full Auto: Finished joint calibration for LEFT elbow. Staged: 0.0000° (click APPLY OFFSET to save).
[INFO] LEFT arm sequential calibration completed successfully.

==================================================
   FULL AUTO SEQUENTIAL CALIBRATION COMPLETE!
==================================================

[INFO] Full Auto sequential calibration ended.
