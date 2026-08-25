============================================================
  UNIFIED ROBOT CALIBRATION SUITE LOADED
============================================================
[RECOMMENDED SEQUENCE]
  1. Calibrate camera intrinsics first if needed (Step 1 > Camera tab).
  2. Calibrate joint offsets using Joint subtab.
  3. Perform marker bracket sweeps using Marker subtab.
  4. Control head and verify offsets as a final check.
============================================================
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[Camera] Initialized camera exposure to AUTO mode.
[INFO] Power is already ON.
[INFO] Servos are ON.
[INFO] Control manager is already enabled. Re-enabling with unlimited_mode_enabled=True...
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=0.0000°, J5=0.0000°, J6=0.0000°] L[J3=0.0000°, J5=0.0000°, J6=0.0000°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Home reset baseline saved to: /home/nvidia/camera_ws/config/home_reset_baseline.json
Starting Home Offset Reset from current pose...
[WARN] Joint limit validation exception: 'rby1_sdk._bindings.Model_A' object has no attribute 'get_dynamics_model'
Reset right arm joint OK: right_arm_0
Reset right arm joint OK: right_arm_1
Reset right arm joint OK: right_arm_2
Reset right arm joint OK: right_arm_3
Reset right arm joint OK: right_arm_4
Reset right arm joint OK: right_arm_5
Reset right arm joint OK: right_arm_6
Reset left arm joint OK: left_arm_0
Reset left arm joint OK: left_arm_1
Reset left arm joint OK: left_arm_2
Reset left arm joint OK: left_arm_3
Reset left arm joint OK: left_arm_4
Reset left arm joint OK: left_arm_5
Reset left arm joint OK: left_arm_6
Reset head joint OK: head_0
Reset head joint OK: head_1
All selected joints reset successfully!
Disabling control manager and waiting 2 seconds...
Powering off overall power (.*)...
[SUCCESS] Saved offsets permanently to setting.yaml!
Re-connecting and initializing robot...
[INFO] Disconnecting from robot...
[INFO] Loaded joint offsets from setting.yaml: R[J3=0.0000°, J5=0.0000°, J6=0.0000°] L[J3=0.0000°, J5=0.0000°, J6=0.0000°]
[INFO] Robot disconnected.
[INFO] Power is not ON. Turning power (.*) on...
[ERROR] Power configuration failed: Failed to turn power on.
[INFO] Turning servos (.*) on...
[ERROR] Servo configuration failed: Failed to turn servos on.
[INFO] Enabling control manager with unlimited_mode_enabled=True...
[INFO] Connected robot model version string: 'v1.2'
[INFO] Loaded joint offsets from setting.yaml: R[J3=0.0000°, J5=0.0000°, J6=0.0000°] L[J3=0.0000°, J5=0.0000°, J6=0.0000°]
[INFO] Loaded Tf_to_marker values for both arms and synced to calibrator memory
[INFO] Automatically switched Step 2 Mode to 'live' because camera and robot are connected.
[INFO] Robot successfully connected and initialized (Classified Version: 1.2).
Home Offset Reset complete!
[Step2] Check Calibration State requested.
[Check State] Step 1: Moving to Joint Ready Pose...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
[Check State] Skipping Joint Ready Pose (Subsequent Move)...
[Check State] Step 2: Moving to Cartesian Symmetrical Checking Pose...
[Check State] Symmetrical move completed successfully.
