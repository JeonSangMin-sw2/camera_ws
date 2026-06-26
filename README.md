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
    -> Swept 169 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 170 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 6 debug points to sweep_points_right_joint_A_axis_6.txt
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_B_axis_5.txt
Swept 169 dense raw coordinate frames during Joint A motion... downsampled to 169 for optimization.
Swept 170 dense raw coordinate frames during Joint B motion... downsampled to 170 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration — 직접 기하학적 방법]
    J6 encoder→plane 기울기     : 1.0011 (이상: 1.0)
    J6 encoder=0° 기준방향       : -37.5780°
    J5=0° 마커 방향 (J6 평면)   : -37.0632°
    ★ J6 보정각                  : +0.5148°
    불확실성(±1σ)                : ±0.0057°
  [참고: 구 방법 perp_dist]
    perp_dist(c_B to J6 axis)   : 4.8883 mm  (새 방법에서는 참고만)
  [Bracket Design Verification]
    r_A  (Joint6 sweep radius, lateral offset from axis) = 132.167 mm
    axial offset (c_B along Joint6 axis)                 = -92.472 mm
  * Angle Error (Deviation from 90°) : 0.8047°
  * Perpendicular Distance (After)   : 0.0000 mm
  * Perpendicular Distance (Before)  : 4.8883 mm
  * Updated Absolute Offset     : 0.5148°

[SUCCESS] Calibration CONVERGED successfully:
  * Circle Normals Angle Error: 0.8047° <= 0.1°
  * Center Distance Error: 0.0000 mm <= 0.1 mm
  * Recommended Absolute Offset: 0.5148°

[VALIDATION SWEEP] Running final validation sweep with recommended offset 0.5148°...

==================================================
   STARTING WRIST_ROLL_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): 0.5148°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 6, duration=15.0s) ---
    -> Swept 169 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 5, duration=15.0s) ---
    -> Swept 169 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 169 dense raw coordinate frames during Joint A motion... downsampled to 169 for optimization.
Swept 169 dense raw coordinate frames during Joint B motion... downsampled to 169 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 6 Calibration — 직접 기하학적 방법]
    J6 encoder→plane 기울기     : 1.0012 (이상: 1.0)
    J6 encoder=0° 기준방향       : -37.5786°
    J5=0° 마커 방향 (J6 평면)   : -36.5426°
    ★ J6 보정각                  : +1.0360°
    불확실성(±1σ)                : ±0.0064°
  [참고: 구 방법 perp_dist]
    perp_dist(c_B to J6 axis)   : 3.6600 mm  (새 방법에서는 참고만)
  [Bracket Design Verification]
    r_A  (Joint6 sweep radius, lateral offset from axis) = 132.226 mm
    axial offset (c_B along Joint6 axis)                 = -92.746 mm
[SUCCESS] Saved combined calibration comparison plot to: /home/nvidia/camera_ws/core/calibration/result_img/circle_fit_right_wrist_roll_v13_joint_calib.png
------------------------------
  [1] Calibration Target: wrist_roll_v13
      Estimated Optimal Offset: 0.515 deg
------------------------------

[CALIBRATION COMPLETE]


==================================================
   [SUCCESS] 3-STEP POLARITY CALIBRATION CONVERGED SUCCESSFULLY!
   * Recommended Absolute Offset : 0.5148°
   * Current Active Offset       : 0.0000° (REVERTED)
   --> Click 'APPLY OFFSET' on the UI panel to apply this new calibration.
==================================================


==================================================
       JOINT CALIBRATION ESTIMATED RESULTS
==================================================

[1] Calibration Target: wrist_roll_v13
    - Target Swept Joint       : Joint 6 (Wrist Roll)
    - Estimated Optimal Offset : 0.5148 deg

[2] Suggested Joint Home Offset update:
  Add offset: 0.5148 deg to calibration config.

[3] Bracket Design Verification (Joint 6 회전축 기준)
    - c_B ~ Joint 6 axis perp. dist (before) : nan mm
    - c_B ~ Joint 6 axis perp. dist (after)  : nan mm
    - Sweep A 피팅원 반경 (r_A, lateral 마커 오프셋)    : nan mm
    - Axial  마커 오프셋 (c_B along Joint 6 axis)  : nan mm
    - Lateral 마커 오프셋 (c_B perp Joint 6 axis)  : nan mm
    ※ 설계 기준 오프셋 방향: X축
==================================================
