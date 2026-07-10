[INFO] Starting Full Auto Sequential Calibration (Right -> Left Arm)...
Starting FULL AUTO sequential calibration...

==================================================
   STARTING PASS 1/2 FOR RIGHT ARM
==================================================

[INFO] Detected Robot Version: 1.2 (is_v1.3: False)
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
[ERROR] Full Auto sequential calibration failed: invalid index to scalar variable.
Traceback (most recent call last):
  File "/home/nvidia/calibration_ws/main_ui.py", line 1713, in run
    res_4 = self.marker_calibrator.perform_calibration_sweep(
  File "/home/nvidia/calibration_ws/core/calibration/MarkerCalibrator.py", line 119, in perform_calibration_sweep
    initial_check = self.marker_st.get_marker_transform(sampling_time=2.0, side=arm_side)
  File "/home/nvidia/calibration_ws/core/marker_detection.py", line 749, in get_marker_transform
    marker_transforms = self.marker_detection.detect(color_img, lpf=lpf, depth_image=depth_img, use_filter=use_filter)
  File "/home/nvidia/calibration_ws/core/marker_detection.py", line 494, in detect
    valid_indices = [i for i, mid in enumerate(ids) if mid[0] in self.marker_id]
  File "/home/nvidia/calibration_ws/core/marker_detection.py", line 494, in <listcomp>
    valid_indices = [i for i, mid in enumerate(ids) if mid[0] in self.marker_id]
IndexError: invalid index to scalar variable.

[INFO] Full Auto sequential calibration ended.
