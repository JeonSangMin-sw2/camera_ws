"""Camera acquisition backends, independent of marker detection/calibration.

Additional backends should implement the CameraDevice protocol. Intrinsics
selection and PnP remain owned by marker_detection, not by this device layer.
RealSense imports its optional SDK only when a device is requested.
"""
import threading
import time
from typing import Protocol
import numpy as np
import cv2


class CameraDevice(Protocol):
    """Minimal frame/exposure contract for a marker observation source."""
    baseline: float
    camera_monitoring: bool

    def initialize_camera(self, set_width: int, set_height: int, set_fps: int): ...
    def capture_image(self, timeout_ms: int = 100): ...
    def get_color_image(self): ...
    def get_depth_image(self): ...
    def get_depth_resolution(self): ...
    def set_exposure(self, exposure_val, auto_exposure=False): ...
    def get_exposure(self): ...
    def get_actual_exposure(self): ...
    def get_camera_temperature(self): ...
    def stream_off(self): ...


class CameraUnavailableError(RuntimeError):
    """No camera hardware or RealSense driver is available."""

class RealSenseCamera:
    # serial_number : Use camera with this serial, if not specified, use the first camera
    """Camera serial number can be searched via realsense_check.py"""
    def __init__(self, serial_number=None):
        global rs
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            if exc.name != 'pyrealsense2':
                raise
            raise CameraUnavailableError('RealSense driver is unavailable') from exc
        # Search for connected cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise CameraUnavailableError("No RealSense connected")
        # Camera selection: use specified serial number if given, otherwise use the first device
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number) or serial_number is None:
                self.device_number = i
                break

        if not hasattr(self, 'device_number'):
            raise CameraUnavailableError(f'RealSense serial {serial_number} is unavailable')

        # Reconnect selected camera for safe usage
        print("Resetting Realsense device...")
        devices[self.device_number].hardware_reset()
        # Wait for camera to reconnect
        time.sleep(3)

        # Re-verify camera info (hardware reset performed)
        ctx = rs.context()
        devices = ctx.query_devices()

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
        self.color_frame_received_at = None
        self.depth_image = None
        self.depth_frame_received_at = None
        self.left_ir_image = None
        self.right_ir_image = None
        self.infrared_frame_received_at = None
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
        self.lock = threading.Lock()
        self.thread = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        self.camera_running = False
        self.camera_monitoring = False
        self._invalidate_frame()
        
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
            raise RuntimeError(f'Camera profile {self.width}x{self.height}@{self.fps} rejected: {e}') from e

        try:
            for i in range(10):
                self.pipeline.wait_for_frames(timeout_ms=1000)
            
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
            raise e

    # Turn streaming on/off with monitoring function
    def monitoring(self, Flag=True):
        self.camera_monitoring = Flag
        if self.camera_monitoring:
            
            self.thread = threading.Thread(target=self.stream_on)
            self.thread.start()
        else:
            self.stream_off()
            if self.thread is not None:
                self.thread.join()

    def stream_on(self , fps = 30):
        align_to = rs.stream.color
        align = rs.align(align_to)
        frame_sleep = 1/fps
        try:
            while self.camera_running:
                self.capture_image()
                 # Visualization Logic
                result_list = []
                if self.color_image is not None:
                    result_list.append(self.color_image)
                if self.depth_image is not None:
                    if np.max(self.depth_image) > 0:
                        # min_dist = float(np.min(self.depth_image[self.depth_image > 0]))
                        # max_dist = float(np.max(self.depth_image))
                        min_dist = 700.0
                        max_dist = 5000.0
                        alpha = (0.0 - 255.0) / (max_dist - min_dist)
                        beta = 255.0 - (min_dist * alpha)
                    else:
                        min_dist = 700.0
                        max_dist = 5000.0
                        alpha = (0.0 - 255.0) / (max_dist - min_dist)
                        beta = 255.0 - (min_dist * alpha)
                    depth_re_img = self.depth_image.astype(np.float32)
                    depth_re_img = depth_re_img * alpha + beta
                    depth_re_img = np.clip(depth_re_img, 0, 255).astype(np.uint8)
                    # depth_re_img = depth_re_img.astype(np.uint8)
                    depth_re_img[self.depth_image == 0] = 0
                    depth_re_img_bgr = cv2.cvtColor(depth_re_img, cv2.COLOR_GRAY2BGR)
                    #depth_re_img_bgr = cv2.applyColorMap(depth_re_img, cv2.COLORMAP_BONE)
                    result_list.append(depth_re_img_bgr)
                if self.Infrared == True and self.left_ir_image is not None and self.right_ir_image is not None:
                    result_list.append(cv2.cvtColor(self.left_ir_image, cv2.COLOR_GRAY2BGR))
                    result_list.append(cv2.cvtColor(self.right_ir_image, cv2.COLOR_GRAY2BGR))
                
                if len(result_list) > 0:
                    if len(result_list) == 1:
                        # If only one image, window size is 1/2 default resolution
                        resize_height = self.height // 2
                        resize_width = self.width // 2
                    else:
                        # If two or more images, window size is 1/n default resolution
                        resize_height = self.height // len(result_list)
                        resize_width = (self.width // len(result_list)) * len(result_list)
                    concat_image = cv2.hconcat(result_list)
                    concat_image = cv2.resize(concat_image, (resize_width, resize_height))
                    cv2.imshow("Preview", concat_image)
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord('q'): # ESC or q
                        raise KeyboardInterrupt
                    
                    if cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt
                time.sleep(frame_sleep)
        except RuntimeError as e:
            print(f"Error: {e}")

    def stream_off(self):
        self.camera_monitoring = False
        self.camera_running = False
        if self.thread is not None:
            self.thread.join()
        try:
            self.pipeline.stop()
        except:
            pass
        self._invalidate_frame()

    def capture_image(self, timeout_ms=100):
        # Completely remove complex thread checking logic.
        if not self.camera_running:
            self._invalidate_frame()
            return

        # This function runs solely inside the stream_on background thread.
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            
            # [LEGACY] RealSense CPU Depth Align & Filter (Not used for PnP detection but causes 25ms bottleneck -> Commented out)
            # align_to = rs.stream.color
            # align = rs.align(align_to)
            # aligned_frames = align.process(frames)
            # color_frame = aligned_frames.get_color_frame()
            # depth_frame = aligned_frames.get_depth_frame()
            # depth_frame = self.spatial.process(depth_frame)
            # depth_frame = self.temporal.process(depth_frame)
            # depth_frame = self.hole_filling.process(depth_frame)

            # [OPTIMIZED] Directly retrieve Color and Depth frames to remove CPU bottleneck
            color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()
            if not color_frame: # or not depth_frame:
                raise RuntimeError('No color frame received')
            color_data = np.asanyarray(color_frame.get_data())
            
            # Read actual exposure metadata from color frame
            cur_act_exp = self.actual_exposure
            try:
                if color_frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
                    cur_act_exp = float(color_frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure))
            except Exception:
                pass
            # depth_data = np.asanyarray(depth_frame.get_data())
        
            with self.lock:
                self.color_image = color_data
                self.color_frame_received_at = time.monotonic()
                self.actual_exposure = cur_act_exp
                # self.depth_image = depth_data
                # if self.Infrared:
                #     ir_frame_left = frames.get_infrared_frame(1)
                #     ir_frame_right = frames.get_infrared_frame(2)
                #     if ir_frame_left and ir_frame_right:
                #         self.left_ir_image = np.asanyarray(ir_frame_left.get_data())
                #         self.right_ir_image = np.asanyarray(ir_frame_right.get_data())
        except Exception:
            self._invalidate_frame()

    def _invalidate_frame(self):
        with self.lock:
            self.color_image = None
            self.depth_image = None
            self.left_ir_image = None
            self.right_ir_image = None
            self.color_frame_received_at = None
            self.depth_frame_received_at = None
            self.infrared_frame_received_at = None

    def _frame_is_fresh(self, received_at):
        maximum_age = max(.1, 3.0 / max(float(getattr(self, 'fps', 30)), 1.0))
        return (self.camera_running and received_at is not None
                and time.monotonic() - received_at <= maximum_age)


    # Functions that must be called for calculation
    def get_color_image(self):
        with self.lock:
            if self.color_image is None or not self._frame_is_fresh(self.color_frame_received_at):
                return None
            return self.color_image.copy()

    def get_depth_image(self):
        with self.lock:
            if self.depth_image is None or not self._frame_is_fresh(self.depth_frame_received_at):
                return None
            return self.depth_image.copy()

    def get_infrared_images(self):
        with self.lock:
            if (getattr(self, 'left_ir_image', None) is None
                    or getattr(self, 'right_ir_image', None) is None
                    or not self._frame_is_fresh(getattr(self, 'infrared_frame_received_at', None))):
                return None, None
            return self.left_ir_image.copy(), self.right_ir_image.copy()

    def get_principal_point_and_focal_length(self):
        return [self.principal_point[0], self.principal_point[1], self.fx, self.fy]

    def get_depth_resolution(self):
        return self.depth_resolution

    def get_baseline(self):
        return self.baseline

    def get_camera_temperature(self):
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
