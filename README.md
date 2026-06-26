[INFO] Starting Joint Sweep: WRIST_PITCH_V13

============================================================
   STARTING ITERATIVE JOINT CALIBRATION SEQUENCE
   Target Arm: RIGHT | Joint Target: WRIST_PITCH_V13
============================================================


[ITERATION 1/8] Sweeping with staged offset 0.0000°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 72 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
[DEBUG] Saved Axis 5 debug points to sweep_points_right_joint_A_axis_5.txt
[DEBUG] Saved Axis 3 debug points to sweep_points_right_joint_B_axis_3.txt
Swept 68 dense raw coordinate frames during Joint A motion... downsampled to 68 for optimization.
Swept 72 dense raw coordinate frames during Joint B motion... downsampled to 72 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration — 삼각형 법칙]
    L_J5M (r_A, J5→Marker 거리)      : 161.815 mm
    r_J3M (r_B, J3→Marker 거리)      : 443.114 mm
    L_J3J5 (URDF FK, pitch 평면)     : 311.048 mm  [FK OK]
    cos(φ_actual)                    : -0.729306  ← 유효
    φ_actual (코사인 법칙)            : 136.8282°
    φ_nominal (FK J5=0 예측)         : 137.8365°  [OK]
    ★ J5 오프셋 (삼각형 법칙)         : -1.0083°  [PRIMARY]
    J5 오프셋 (벡터 비교 cross-check): -2.3494°  [cross-check]
    최종 사용 방법                    : triangle_law
  [삼각형 일관성 검증]
    r_J3M 실측                       : 443.114 mm
    r_J3M 예측 (δ5=0)                : 444.467 mm  → 잔차 1.353 mm
    r_J3M 예측 (δ5=-1.01°)        : 445.790 mm  → 잔차 2.675 mm  ⚠ 개선 없음
  [Marker Design Values (Sweep A 기반)]
    r_A sweep (= L_J5M)              : 161.815 mm
    r_A direct (|cam-perp|)          : 161.816 mm
    axial offset along J5 axis       : 0.020 mm  (sweep B: 15.109 mm)
  * Angle Error (Deviation from 90°) : 88.0430°
  * Perpendicular Distance (After)   : 2.6753 mm
  * Perpendicular Distance (Before)  : 1.3527 mm
  * Updated Absolute Offset     : -1.0083°

[ITERATION 2/8] Sweeping with staged offset -1.0083°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -1.0083°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 70 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 70 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 70 dense raw coordinate frames during Joint A motion... downsampled to 70 for optimization.
Swept 70 dense raw coordinate frames during Joint B motion... downsampled to 70 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration — 삼각형 법칙]
    L_J5M (r_A, J5→Marker 거리)      : 161.610 mm
    r_J3M (r_B, J3→Marker 거리)      : 442.656 mm
    L_J3J5 (URDF FK, pitch 평면)     : 311.049 mm  [FK OK]
    cos(φ_actual)                    : -0.726855  ← 유효
    φ_actual (코사인 법칙)            : 136.6234°
    φ_nominal (FK J5=0 예측)         : 137.8372°  [OK]
    ★ J5 오프셋 (삼각형 법칙)         : -1.2139°  [PRIMARY]
    J5 오프셋 (벡터 비교 cross-check): -2.2969°  [cross-check]
    최종 사용 방법                    : triangle_law
  [삼각형 일관성 검증]
    r_J3M 실측                       : 442.656 mm
    r_J3M 예측 (δ5=0)                : 444.287 mm  → 잔차 1.631 mm
    r_J3M 예측 (δ5=-1.21°)        : 445.874 mm  → 잔차 3.218 mm  ⚠ 개선 없음
  [Marker Design Values (Sweep A 기반)]
    r_A sweep (= L_J5M)              : 161.610 mm
    r_A direct (|cam-perp|)          : 161.621 mm
    axial offset along J5 axis       : 0.050 mm  (sweep B: 14.256 mm)
  * Angle Error (Deviation from 90°) : 88.1487°
  * Perpendicular Distance (After)   : 3.2180 mm
  * Perpendicular Distance (Before)  : 1.6307 mm
  * Updated Absolute Offset     : -2.2222°

[ITERATION 3/8] Sweeping with staged offset -2.2222°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -2.2222°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 71 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 71 dense raw coordinate frames during Joint A motion... downsampled to 71 for optimization.
Swept 68 dense raw coordinate frames during Joint B motion... downsampled to 68 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration — 삼각형 법칙]
    L_J5M (r_A, J5→Marker 거리)      : 161.694 mm
    r_J3M (r_B, J3→Marker 거리)      : 439.625 mm
    L_J3J5 (URDF FK, pitch 평면)     : 311.048 mm  [FK OK]
    cos(φ_actual)                    : -0.699626  ← 유효
    φ_actual (코사인 법칙)            : 134.3970°
    φ_nominal (FK J5=0 예측)         : 137.8377°  [OK]
    ★ J5 오프셋 (삼각형 법칙)         : -3.4407°  [PRIMARY]
    J5 오프셋 (벡터 비교 cross-check): -2.3086°  [cross-check]
    최종 사용 방법                    : triangle_law
  [삼각형 일관성 검증]
    r_J3M 실측                       : 439.625 mm
    r_J3M 예측 (δ5=0)                : 444.361 mm  → 잔차 4.736 mm
    r_J3M 예측 (δ5=-3.44°)        : 448.748 mm  → 잔차 9.123 mm  ⚠ 개선 없음
  [Marker Design Values (Sweep A 기반)]
    r_A sweep (= L_J5M)              : 161.694 mm
    r_A direct (|cam-perp|)          : 161.728 mm
    axial offset along J5 axis       : 0.069 mm  (sweep B: -9.469 mm)
  * Angle Error (Deviation from 90°) : 88.7434°
  * Perpendicular Distance (After)   : 9.1227 mm
  * Perpendicular Distance (Before)  : 4.7360 mm
  * Updated Absolute Offset     : -5.6629°

[ITERATION 4/8] Sweeping with staged offset -5.6629°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -5.6629°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 68 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
    -> Swept 70 dense raw coordinate frames during Joint B motion.

[INFO] Sweep finished. Returning arm to initial pose...
Swept 68 dense raw coordinate frames during Joint A motion... downsampled to 68 for optimization.
Swept 70 dense raw coordinate frames during Joint B motion... downsampled to 70 for optimization.

--- [3] Starting Offline Direct Angle Calculation (Fitted Circle Normal Orthogonality - Camera Frame) ---
  [v1.3 Joint 5 Calibration — 삼각형 법칙]
    L_J5M (r_A, J5→Marker 거리)      : 161.501 mm
    r_J3M (r_B, J3→Marker 거리)      : 434.389 mm
    L_J3J5 (URDF FK, pitch 평면)     : 311.047 mm  [FK OK]
    cos(φ_actual)                    : -0.655529  ← 유효
    φ_actual (코사인 법칙)            : 130.9598°
    φ_nominal (FK J5=0 예측)         : 137.8376°  [OK]
    ★ J5 오프셋 (삼각형 법칙)         : -6.8778°  [PRIMARY]
    J5 오프셋 (벡터 비교 cross-check): -2.1761°  [cross-check]
    최종 사용 방법                    : triangle_law
  [삼각형 일관성 검증]
    r_J3M 실측                       : 434.389 mm
    r_J3M 예측 (δ5=0)                : 444.191 mm  → 잔차 9.802 mm
    r_J3M 예측 (δ5=-6.88°)        : 452.598 mm  → 잔차 18.210 mm  ⚠ 개선 없음
  [Marker Design Values (Sweep A 기반)]
    r_A sweep (= L_J5M)              : 161.501 mm
    r_A direct (|cam-perp|)          : 161.692 mm
    axial offset along J5 axis       : 0.185 mm  (sweep B: -22.533 mm)
  * Angle Error (Deviation from 90°) : 87.0150°
  * Perpendicular Distance (After)   : 18.2098 mm
  * Perpendicular Distance (Before)  : 9.8020 mm
  * Updated Absolute Offset     : -10.6629°

[ITERATION 5/8] Sweeping with staged offset -10.6629°...

==================================================
   STARTING WRIST_PITCH_V13 CONTINUOUS OFFSET CALIBRATION SWEEP
   [Baseline Shift (Current Applied Offset): -10.6629°]
==================================================

--- [1/2] Commencing Continuous Sweep on Joint A (Index 5, duration=15.0s) ---
    -> Swept 71 dense raw coordinate frames during Joint A motion.

--- [2/2] Commencing Continuous Sweep on Joint B (Index 3, duration=15.0s) ---
[STOP] Stop requested by user.
[STOP] Sending cancel_control to robot!
[ERROR] Joint B sweep motion failed or was cancelled.
[ERROR] Iteration 5 sweep failed. Aborting calibration.
