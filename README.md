[INFO] Starting Unified Marker Sweep (Axis 6 & 5) (Head Tracking: False)

==================================================
   [Stage 1/3] Sweeping Axis 4 (Wrist Yaw)...
==================================================


==================================================
   STARTING 4 CONTINUOUS MARKER SWEEP
==================================================
[INFO] Moving Marker Axis 4 to start sweep position...
[INFO] Commencing continuous sweep on Marker Axis 4 (duration=15.0s)...
    -> Swept 152 dense raw coordinate frames during Marker Axis 4 motion.

[INFO] Sweep complete. Returning to initial ready pose...
[ERROR] Worker exception: name 'ee_name' is not defined
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/additional_calib_ui.py", line 465, in run
    res_4 = self.calibrator.perform_calibration_sweep(
  File "/home/nvidia/camera_ws/core/calibration/MarkerCalibrator.py", line 221, in perform_calibration_sweep
    arm_side, axis_mode, dataset, initial_joint_pos, ee_name, dyn_model, T_t5_to_cam_fixed, "marker", log_callback
NameError: name 'ee_name' is not defined

[ERROR] Marker sweep failed.
