[INFO] Starting Unified Marker Sweep (Axis 6 & 5) (Head Tracking: False)

==================================================
   [Stage 1/3] Sweeping Axis 4 (Wrist Yaw)...
==================================================


==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving Marker Axis 4 to start sweep position...
[INFO] Commencing continuous sweep on Marker Axis 4 (duration=15.0s)...
    -> Swept 150 dense raw coordinate frames during Marker Axis 4 motion.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 4 marker sweep debug points to sweep_points_right_marker_axis_4.txt

==================================================
   [Stage 2/3] Returning to Initial Starting Pose...
==================================================


==================================================
   [Stage 2/3] Sweeping Axis 6 (Roll)...
==================================================


==================================================
   STARTING 6 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving Marker Axis 6 to start sweep position...
[INFO] Commencing continuous sweep on Marker Axis 6 (duration=15.0s)...
    -> Swept 151 dense raw coordinate frames during Marker Axis 6 motion.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 6 marker sweep debug points to sweep_points_right_marker_axis_6.txt

==================================================
   [Stage 3/3] Sweeping Axis 5 (Pitch)...
==================================================


==================================================
   STARTING 5 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving Marker Axis 5 to start sweep position...
[INFO] Commencing continuous sweep on Marker Axis 5 (duration=15.0s)...
    -> Swept 151 dense raw coordinate frames during Marker Axis 5 motion.

[INFO] Sweep complete. Returning to initial ready pose...
[DEBUG] Saved Axis 5 marker sweep debug points to sweep_points_right_marker_axis_5.txt

[PROCESSING] Computing unified bracket calibration parameters...

==================================================
       UNIFIED BRACKET CALIBRATION RESULTS
==================================================

[1] Cartesian Offset (EE Link Frame)
    - X-Offset: 0.00 mm
    - Y-Offset: -54.68 mm
    - Z-Offset: 10.00 mm
       * (L_5_ee: 126.1 mm, R6: 54.31 mm, R5: 174.45 mm, R4: 180.95 mm)

[2] Angular Misalignment (EE Link Frame)
    - Roll : 89.73 deg
    - Pitch: 0.07 deg
    - Yaw  : 179.07 deg

[3] setting.yaml Config Update values:
  Tf_to_marker_right: [0.0000, -0.0547, 0.0100, 89.73, 0.07, 179.07]

[4] Confidence Metrics:
    - Orthogonality Error  : 2.221 deg
==================================================
