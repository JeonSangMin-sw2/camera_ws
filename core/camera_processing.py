"""Camera backends. Marker processing remains in marker_detection.py."""
import threading
import time
import numpy as np
import pyrealsense2 as rs

class CameraUnavailableError(RuntimeError):
    pass


class RealSenseCamera:
    # serial_number : Use camera with this serial, if not specified, use the first camera
    """Camera serial number can be searched via realsense_check.py"""
    def __init__(self, serial_number=None):
        # Search for connected cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise CameraUnavailableError("No RealSense connected")
        # Camera selection: use specified serial number if given, otherwise use the first device
        self.device_number = None
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number) or serial_number is None:
                self.device_number = i
                break

        if self.device_number is None:
            raise CameraUnavailableError(f"RealSense serial {serial_number!r} was not found")
        selected_serial = devices[self.device_number].get_info(rs.camera_info.serial_number)

        # Reconnect selected camera for safe usage
        print("Resetting Realsense device...")
        devices[self.device_number].hardware_reset()
        # Wait for camera to reconnect
        time.sleep(3)

        # Re-verify camera info (hardware reset performed)
        ctx = rs.context()
        devices = ctx.query_devices()

        # Enumeration order can change after a USB reset. Re-select by identity,
        # never silently switch to a different connected camera.
        self.device_number = next((i for i, device in enumerate(devices)
                                   if device.get_info(rs.camera_info.serial_number) == selected_serial), None)
        if self.device_number is None:
            raise CameraUnavailableError(f"RealSense {selected_serial!r} did not reconnect after reset")

        self.device_name = devices[self.device_number].get_info(rs.camera_info.name)
        self.serial_number = devices[self.device_number].get_info(rs.camera_info.serial_number)
        print("Using camera is : ", self.device_name)

        # Depth scale check: D435 is 1mm, D405 is 0.1mm, depending on model
        depth_sensor = devices[self.device_number].first_depth_sensor()
        if depth_sensor.supports(rs.option.thermal_compensation):
            depth_sensor.set_option(rs.option.thermal_compensation, 1.0) # Thermal compensation On
            depth_sensor.set_option(rs.option.visual_preset, 3) # High Accuracy
        depth_scale = depth_sensor.get_depth_scale()
        print("depth scale : ", depth_scale)
        

        # Parameter configuration for running camera
        self.depth_resolution = depth_scale*1000                # Depth value resolution of each pixel
        self.pipeline = rs.pipeline()                           # Pipeline for camera streaming
        self.config = rs.config()                               # Camera config structure
        self.spatial = rs.spatial_filter()                      # Spatial filter (noise reduction)
        self.spatial.set_option(rs.option.filter_magnitude, 2)  # Set spatial filter strength
        self.temporal = rs.temporal_filter()                    # Temporal filter (flicker prevention)
        self.hole_filling = rs.hole_filling_filter()            # Hole filling filter

        # Status flags
        self.camera_running = False                             # Camera running status flag
        self.camera_monitoring = False                          # Camera monitoring status flag
        self.Infrared = True                                    # Infrared camera usage flag

        # Image storage variables
        self.color_image = None
        self.depth_image = None
        # Default resolution
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30

        # Camera intrinsic parameters: used for calculating depth map and marker coordinates
        self.fx = 0.0                                           # Focal length x
        self.fy = 0.0                                           # Focal length y
        self.principal_point = [0.0, 0.0]                       # Principal point (image center)
        self.intrinsics = None                                  # Intrinsic matrix
        self.profile = None                                     # Camera profile
        self.baseline = 0.065                                   # Stereo camera baseline (m)
        self.dist_coeffs = None                                 # Distortion coefficients
        self.actual_exposure = 6000.0                           # Actual measured exposure from frame metadata (μs)

        # Lock for thread synchronization
        self.lock = threading.RLock()
        self.io_lock = threading.RLock()
        self.frame_id = 0
        self.frame_timestamp = None
        self.temperature = None
        self.temperature_timestamp = None
        self.last_error = ""
        self.thread = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        
        try:
            self.config.enable_device(self.serial_number)
            # Enable streaming for the selected camera types. Depth is always used.
            # To reduce CPU load, ir1/ir2 are streamed when IR is used, otherwise color is streamed.
            # self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            # if self.Infrared:
            #     self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
            #     self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
            
            # Start pipeline
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to start pipeline with {self.width}x{self.height}@{self.fps}. Error: {e}")
            print("Attempting fallback resolution (848x480 @ 30fps)...")
            try:
                self.config = rs.config() # Reset config
                self.config.enable_device(self.serial_number)
                self.width, self.height, self.fps = 848, 480, 30
                # self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.profile = self.pipeline.start(self.config)
            except Exception as e2:
                print(f"Fallback 1 failed: {e2}. Attempting 640x480 @ 30fps...")
                try:
                    self.config = rs.config()
                    self.config.enable_device(self.serial_number)
                    self.width, self.height, self.fps = 640, 480, 30
                    # self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                    self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                    self.profile = self.pipeline.start(self.config)
                except Exception as e3:
                    print(f"All profile attempts failed: {e3}")
                    raise e3

        try:
            # Discard first 10 frames to allow camera exposure to stabilize
            for i in range(10):
                self.pipeline.wait_for_frames()
            
            # [NEW] Sensor Auto Exposure configuration - adaptive to diverse environments
            device = self.profile.get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1) # Enable auto exposure (previously manual 0)
                # Comment out previous manual settings
                # if sensor.supports(rs.option.exposure):
                #     try:
                #         sensor.set_option(rs.option.exposure, 6000) # 6ms (exposure time)
                #     except Exception as e:
                #         print(f"Warning: Failed to set exposure on sensor {sensor.get_info(rs.camera_info.name)}: {e}")
                # if sensor.supports(rs.option.gain):
                #     try:
                #         sensor.set_option(rs.option.gain, 80) # Increase gain for brightness
                #     except Exception as e:
                #         print(f"Warning: Failed to set gain on sensor {sensor.get_info(rs.camera_info.name)}: {e}")

            # Get depth camera intrinsics: used for baseline, fx, fy, principal_point
            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

            # left_ir_stream = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            # right_ir_stream = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            # 
            # extrinsics = left_ir_stream.get_extrinsics_to(right_ir_stream)
            # self.baseline = abs(extrinsics.translation[0]) # m
            self.intrinsics = color_stream.get_intrinsics()
            self.fx = self.intrinsics.fx
            self.fy = self.intrinsics.fy
            self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy] #pixel
            self.dist_coeffs = np.array(self.intrinsics.coeffs)

            print(f"Successfully initialized: {self.width}x{self.height} @ {self.fps}fps")
            print(f"Focal Length: fx={self.fx}, fy={self.fy}")
            print(f"Principal Point: {self.principal_point[0]}, {self.principal_point[1]}")
            print("Baseline: ", self.baseline)
            self.camera_running = True
        except Exception as e:
            print(f"Camera didn't initialize post-start: {e}")
            self.pipeline.stop()
            raise e

    # Turn streaming on/off with monitoring function
    def monitoring(self, Flag=True):
        if Flag:
            self.start_capture()
        else:
            self.stream_off()

    def start_capture(self):
        """One producer owns the camera pipeline. No GUI work runs here."""
        if self.thread is not None and self.thread.is_alive():
            return
        if not self.camera_running:
            raise CameraUnavailableError("Camera pipeline is not initialized")
        self.camera_monitoring = True
        self.thread = threading.Thread(target=self.stream_on, name="camera-capture", daemon=True)
        self.thread.start()

    def stream_on(self, fps=30):
        next_temperature = 0.0
        while self.camera_running:
            self.capture_image()
            now = time.monotonic()
            if now >= next_temperature:
                temperature = self._read_camera_temperature()
                with self.lock:
                    self.temperature = temperature
                    self.temperature_timestamp = time.time()
                next_temperature = now + 5.0
            # SDK frame wait controls cadence; avoid spinning on disconnection.
            time.sleep(0.001)

    def stream_off(self):
        self.camera_running = False
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Camera capture thread did not stop")
        self.camera_monitoring = False
        with self.io_lock:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass

    def capture_image(self):
        if not self.camera_running:
            return
        # Compatibility callers may request capture, but only the producer reads
        # the pipeline once continuous capture is running.
        if self.camera_monitoring and threading.current_thread() is not self.thread:
            return
        try:
            with self.io_lock:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                frame = frames.get_color_frame()
                if not frame:
                    return
                pixels = np.asanyarray(frame.get_data()).copy()
                exposure = self.actual_exposure
                if frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
                    exposure = float(frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure))
            with self.lock:
                self.color_image = pixels
                self.actual_exposure = exposure
                self.frame_id += 1
                self.frame_timestamp = time.time()
                self.last_error = ""
        except Exception as error:
            with self.lock:
                self.last_error = str(error)


    # Functions that must be called for calculation
    def get_color_image(self):
        with self.lock:
            if self.color_image is None:
                return None
            return self.color_image.copy()

    def get_depth_image(self):
        with self.lock:
            if self.depth_image is None:
                return None
            return self.depth_image.copy()

    def get_infrared_images(self):
        with self.lock:
            if not hasattr(self, 'left_ir_image') or not hasattr(self, 'right_ir_image'):
                return None, None
            return self.left_ir_image.copy(), self.right_ir_image.copy()

    def get_principal_point_and_focal_length(self):
        return [self.principal_point[0], self.principal_point[1], self.fx, self.fy]

    def get_depth_resolution(self):
        return self.depth_resolution

    def get_baseline(self):
        return self.baseline

    def _read_camera_temperature(self):
        with self.io_lock:
            try:
                if not self.camera_running or self.profile is None:
                    return None
                device = self.profile.get_device()
                depth_sensor = device.first_depth_sensor()
                if depth_sensor.supports(rs.option.projector_temperature):
                    proj_temp = depth_sensor.get_option(rs.option.projector_temperature)
                    return proj_temp
                # If the model supports ASIC temperature (e.g. D435 supports either or both)
                elif depth_sensor.supports(rs.option.asic_temperature):
                    asic_temp = depth_sensor.get_option(rs.option.asic_temperature)
                    return asic_temp
            except Exception as e:
                print(f"Failed to get temperature: {e}")
                return None

    def set_exposure(self, exposure_val, auto_exposure=False):
        """
        exposure_val: exposure time in microseconds (e.g. 100 ~ 100000)
        auto_exposure: True for auto exposure, False for manual exposure
        """
        with self.io_lock:
            try:
                if not self.camera_running or self.profile is None:
                    return False
                device = self.profile.get_device()
                for sensor in device.query_sensors():
                    if auto_exposure:
                        if sensor.supports(rs.option.enable_auto_exposure):
                            sensor.set_option(rs.option.enable_auto_exposure, 1)
                    else:
                        if sensor.supports(rs.option.enable_auto_exposure):
                            sensor.set_option(rs.option.enable_auto_exposure, 0)
                        if sensor.supports(rs.option.exposure):
                            sensor.set_option(rs.option.exposure, float(exposure_val))
                return True
            except Exception as e:
                print(f"[Camera] Failed to set exposure (auto={auto_exposure}, val={exposure_val}): {e}")
                return False

    def get_exposure(self):
        """
        Returns (auto_exposure: bool, exposure_val: float)
        """
        with self.io_lock:
            try:
                if not self.camera_running or self.profile is None:
                    return True, 6000.0
                device = self.profile.get_device()
                for sensor in device.query_sensors():
                    if sensor.supports(rs.option.enable_auto_exposure):
                        is_auto = bool(sensor.get_option(rs.option.enable_auto_exposure) > 0.5)
                        exp_val = sensor.get_option(rs.option.exposure) if sensor.supports(rs.option.exposure) else 6000.0
                        return is_auto, float(exp_val)
                return True, 6000.0
            except Exception as e:
                print(f"[Camera] Failed to get exposure: {e}")
                return True, 6000.0
            return None

    def get_actual_exposure(self):
        """Returns the actual measured exposure (in microseconds) from the latest frame metadata."""
        with self.lock:
            return float(self.actual_exposure)

    def get_dist_coeffs(self):
        return self.dist_coeffs

    def get_camera_temperature(self):
        with self.lock:
            return self.temperature

    def get_monitor_snapshot(self):
        """Return a detached, consistent snapshot; never perform camera I/O."""
        with self.lock:
            return {
                "image": None if self.color_image is None else self.color_image.copy(),
                "frame_id": self.frame_id,
                "frame_timestamp": self.frame_timestamp,
                "temperature": self.temperature,
                "temperature_timestamp": self.temperature_timestamp,
                "exposure": self.actual_exposure,
                "connected": self.camera_running,
                "error": self.last_error,
            }
