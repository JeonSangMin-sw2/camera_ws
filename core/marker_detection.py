import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading
import os, yaml

#debugging flag : must be all false in production
imshow_when_detect = False
tcpip_send = False
use_calib_int = False # Whether to use the finely calibrated intrinsics file (camera_intrinsics.yaml)
# see_depth_sensors_depth = False
# see_stereo_depth = False

# Utility classes
class TCPClient:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        try:
            self.sock.connect((ip, port))
            print("Connected to Python Server!")
            self.connected = True
        except ConnectionRefusedError:
            print("Connection Failed. (Is Python Server running?)")
        except Exception as e:
            print(f"Connection Error: {e}")

    def __del__(self):
        if self.connected:
            self.sock.close()

    def send_pose(self, T):
        if not self.connected:
            return
        
        # T is expected to be a flat list or numpy array of 16 floats
        if isinstance(T, np.ndarray):
            T = T.flatten().tolist()
            
        # Pack 16 floats (4 bytes each) -> 64 bytes
        try:
            packed_data = struct.pack('16f', *T)
            self.sock.send(packed_data)
        except Exception as e:
            print(f"Send Error: {e}")
            self.connected = False
        
# Camera class
"""
This class is only compatible with RealSense cameras
If using another camera, it is recommended to implement functions with the same signature.
"""
class RealSenseCamera:
    # serial_number : Use camera with this serial, if not specified, use the first camera
    """Camera serial number can be searched via realsense_check.py"""
    def __init__(self, serial_number=None):
        # Search for connected cameras
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise RuntimeError("No RealSense connected")
        # Camera selection: use specified serial number if given, otherwise use the first device
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number) or serial_number is None:
                self.device_number = i
                break

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

        # Lock for thread synchronization
        self.lock = threading.Lock()
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
        self.camera_running = False
        if self.thread is not None:
            self.thread.join()
        try:
            self.pipeline.stop()
        except:
            pass

    def capture_image(self):
        # Completely remove complex thread checking logic.
        if not self.camera_running:
            return

        # This function runs solely inside the stream_on background thread.
        try:
            frames = self.pipeline.wait_for_frames()
            
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
                print("no frame")
                return
            color_data = np.asanyarray(color_frame.get_data())
            # depth_data = np.asanyarray(depth_frame.get_data())
        
            with self.lock:
                self.color_image = color_data
                # self.depth_image = depth_data
                # if self.Infrared:
                #     ir_frame_left = frames.get_infrared_frame(1)
                #     ir_frame_right = frames.get_infrared_frame(2)
                #     if ir_frame_left and ir_frame_right:
                #         self.left_ir_image = np.asanyarray(ir_frame_left.get_data())
                #         self.right_ir_image = np.asanyarray(ir_frame_right.get_data())
        except Exception as e:
            pass # Ignore intermittent frame drops


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
        return None

    def get_dist_coeffs(self):
        return self.dist_coeffs


class Marker_Detection:
    def __init__(self):
        # Define which markers to detect
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        # Parameter tuning for improving marker detection precision
        # 1. Maximize sub-pixel precision
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01

        # 2. Refine binarization (for lighting and shadow robust)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 3  # Fine scan
        self.parameters.adaptiveThreshConstant = 7     # Adjusted lower for noise

        # 3. Shape approximation and filtering
        self.parameters.polygonalApproxAccuracyRate = 0.01 # Stricter square check
        self.parameters.minDistanceToBorder = 3
        self.parameters.minMarkerPerimeterRate = 0.01

        # 4. Enhance internal bit sampling
        self.parameters.perspectiveRemovePixelPerCell = 12 # Fine bit extraction

        # Intrinsic parameters used for calculation
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.dist_coeffs = None
        self.depth_resolution = 1
        self.rpy = [0, 0, 0]
        
        self.lpf_alpha = 0.5
        self.prev_pts_dict = {}

        self.focal_scale = 1.0# 0.99 # Focal length scaling factor for fine-tuning

        # Marker type and ID to detect
        self.marker_type = None
        self.marker_id = None
        
        self.marker_size_mm = 36.0 # Default value, overwritten by config
        self.markers_config = {}
        if tcpip_send:
            self.tcp_client = TCPClient("127.0.0.1", 5000)

        self.marker_depth = 0
        self.stereo_depth = 0

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.detector = None

    # Set camera parameters required for calculation
    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.fx = param[2]
        self.fy = param[3]

    def set_depth_resolution(self, depth_resolution):
        self.depth_resolution = depth_resolution

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_dist_coeffs(self, dist_coeffs):
        self.dist_coeffs = dist_coeffs

    def set_marker_type(self, marker_type="plate"):
        self.marker_type = marker_type
        if marker_type == "plate":
            self.plate_left_ids = self.markers_config.get("plate", {}).get("left_ids", [])
            self.plate_right_ids = self.markers_config.get("plate", {}).get("right_ids", [])
            self.marker_id = self.plate_left_ids + self.plate_right_ids
            self.marker_size_mm = self.markers_config.get("plate", {}).get("plate_size_mm", 100.0) * 0.8
        else:
            self.marker_id = []

    def get_depth_from_depth_img(self, depth_image, center_pixel):
        x, y = int(center_pixel[0]), int(center_pixel[1])
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            # Use a small window (e.g., 3x3) to get a more stable depth value
            roi = depth_image[max(0, y-1):min(depth_image.shape[0], y+2),
                              max(0, x-1):min(depth_image.shape[1], x+2)]
            valid_depths = roi[roi > 0]
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))*self.depth_resolution
            return float(depth_image[y, x])*self.depth_resolution
        return 0.0

    # Center coordinates of markers (4x4 matrix)
    def detect(self, color_image, lpf = False, logging = False, depth_image = None, use_filter = True):
        # [LEGACY] Image-wide lens undistortion (causes CPU processing bottleneck -> Commented out)
        # pnp_dist_coeffs = self.dist_coeffs
        # if self.dist_coeffs is not None and np.any(self.dist_coeffs != 0):
        #     base_cam_mat = np.array([
        #         [self.fx, 0, self.principal_point[0]],
        #         [0, self.fy, self.principal_point[1]],
        #         [0, 0, 1]
        #     ], dtype=np.float32)
        #     # Image-wide lens undistortion (to restore marker edges to straight lines)
        #     color_image = cv2.undistort(color_image, base_cam_mat, self.dist_coeffs, None, base_cam_mat)
        #     # Since distortion is already corrected, ignore distortion params in solvePnP to avoid double correction
        #     pnp_dist_coeffs = None

        # [OPTIMIZED] Pass distortion parameters to solvePnP instead of doing full undistort (takes <0.01ms)
        pnp_dist_coeffs = self.dist_coeffs
            
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
        # Step 2: Enforce sub-pixel precision
        if corners is not None and len(corners) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            for i in range(len(corners)):
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)
                
        marker_centers_result = []
        

        # Filter only registered markers (create a new list since tuple and ndarray don't support pop)
        if ids is not None and len(ids) > 0 and self.marker_id is not None:
            valid_indices = []
            for i, mid in enumerate(ids):
                val = mid[0] if isinstance(mid, (np.ndarray, list)) else mid
                if val in self.marker_id:
                    valid_indices.append(i)
            if len(valid_indices) > 0:
                corners = tuple(corners[i] for i in valid_indices)
                ids = np.array([ids[i] for i in valid_indices])
            else:
                corners, ids = (), None

        # debugging (Adjust positions so only filtered markers are displayed on screen)
        if imshow_when_detect:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            cv2.imshow("Detected Markers", color_image)
            cv2.waitKey(1)
            
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            unique_ids, counts = np.unique(ids_flat, return_counts=True)
            duplicate_ids = unique_ids[counts > 1]
            if len(duplicate_ids) > 0:
                print(f"Warning: Duplicate marker IDs detected: {duplicate_ids}. Ignoring these duplicates in this frame.")
                
            comp_fx = self.fx * self.focal_scale
            comp_fy = self.fy * self.focal_scale
            cam_mat = np.array([
                [comp_fx, 0, self.principal_point[0]],
                [0, comp_fy, self.principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            half_m = self.marker_size_mm / 2.0

            obj_pts = np.array([
                [-half_m, -half_m, 0], [ half_m, -half_m, 0],
                [ half_m,  half_m, 0], [-half_m,  half_m, 0]
            ], dtype=np.float32)
            
            for i in range(len(ids)):
                marker_id = ids_flat[i]
                if marker_id in duplicate_ids:
                    continue
                c = corners[i][0]
                
                # # Check depth if see_depth_sensors_depth is True and it's a plate
                # if see_depth_sensors_depth and self.marker_type == "plate" and depth_image is not None:
                #     center_px = np.mean(c, axis=0)
                #     self.marker_depth = self.get_depth_from_depth_img(depth_image, center_px)
                #     print(f"[Depth] Marker ID {marker_id} Center Depth: {self.marker_depth:.1f} mm")

                success, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_mat, pnp_dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                if not success:
                    continue
                rot_matrix, _ = cv2.Rodrigues(rvec)
                center_pos = tvec.flatten().tolist()
                
                # --- EMA & Slerp Smoothing ---
                if use_filter or lpf:
                    alpha = self.lpf_alpha
                    if marker_id in self.prev_pts_dict:
                        prev_pos, prev_rot = self.prev_pts_dict[marker_id]
                        
                        # Adaptive EMA: Calculate position delta
                        dist = np.linalg.norm(np.array(center_pos) - np.array(prev_pos))
                        # If movement is larger than 2mm per frame (fast motor movement), disable smoothing to avoid lag
                        if dist > 2.0:
                            alpha = 1.0
                            
                        # 1. Position EMA
                        center_pos = [
                            alpha * center_pos[0] + (1 - alpha) * prev_pos[0],
                            alpha * center_pos[1] + (1 - alpha) * prev_pos[1],
                            alpha * center_pos[2] + (1 - alpha) * prev_pos[2]
                        ]
                        # 2. Rotation Slerp
                        try:
                            from scipy.spatial.transform import Rotation as R
                            r_curr = R.from_matrix(rot_matrix)
                            r_prev = R.from_matrix(prev_rot)
                            
                            delta_rot = r_curr * r_prev.inv()
                            r_smoothed = R.from_rotvec(delta_rot.as_rotvec() * alpha) * r_prev
                            rot_matrix_smoothed = r_smoothed.as_matrix()
                        except ImportError:
                            rot_matrix_smoothed = rot_matrix
                    else:
                        rot_matrix_smoothed = rot_matrix
                        
                    # Update previous state
                    self.prev_pts_dict[marker_id] = (center_pos, rot_matrix_smoothed)
                else:
                    rot_matrix_smoothed = rot_matrix
                
                transform = [
                    rot_matrix_smoothed[0][0], rot_matrix_smoothed[0][1], rot_matrix_smoothed[0][2], center_pos[0],
                    rot_matrix_smoothed[1][0], rot_matrix_smoothed[1][1], rot_matrix_smoothed[1][2], center_pos[1],
                    rot_matrix_smoothed[2][0], rot_matrix_smoothed[2][1], rot_matrix_smoothed[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                # Plate group identification
                if marker_id in getattr(self, 'plate_left_ids', []):
                    marker_centers_result.append(("plate_left", transform))
                elif marker_id in getattr(self, 'plate_right_ids', []):
                    marker_centers_result.append(("plate_right", transform))
                else:
                    marker_centers_result.append((marker_id, transform))
                # if tcpip_send:
                #     self.tcp_client.send_pose(transform)
                
        return marker_centers_result



class Marker_Transform:
    def __init__(self, serial_number = None):
        # Initialize
        self.camera = RealSenseCamera(serial_number=serial_number)
        self.marker_detection = Marker_Detection()
        
        # Load configs globally in the wrapper class
        self._load_all_configs()
        
        # Setup Transforms
        tf_vec_l = self.markers_config.get("Tf_to_marker_left", self.markers_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        tf_vec_r = self.markers_config.get("Tf_to_marker_right", self.markers_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        head_base_vec = self.camera_config.get("head_base_to_cam", [0.009, -0.09, -0.085, 159.0, 0.0, 180.0])
        print(tf_vec_l)
        
        self.Tf_to_marker_tf_left = self.make_transform(tf_vec_l)
        self.Tf_to_marker_tf_right = self.make_transform(tf_vec_r)
        self.head_base_to_cam_tf = self.make_transform(head_base_vec)
        
        self.width = self.camera_config.get("width", 1280)
        self.height = self.camera_config.get("height", 720)
        self.fps = self.camera_config.get("fps", 30)
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

        depth_resolution = self.camera.get_depth_resolution()
        self.marker_detection.set_depth_resolution(depth_resolution)

        dist_coeffs = self.camera.get_dist_coeffs()
        self.marker_detection.set_dist_coeffs(dist_coeffs)

        self.marker_detection.set_baseline(self.camera.baseline)
        

        # [NEW] Apply calibrated camera intrinsics setting (camera_intrinsics.yaml)
        if use_calib_int:
            # Search for config in current dir or parent dir (to support both source and installed structures)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(base_dir, "config", "camera_intrinsics.yaml")
            if not os.path.exists(calib_file):
                calib_file = os.path.join(os.path.dirname(base_dir), "config", "camera_intrinsics.yaml")
            if os.path.exists(calib_file):
                try:
                    with open(calib_file, "r") as f:
                        calib_data = yaml.safe_load(f)
                    
                    mtx = np.array(calib_data["camera_matrix"])
                    dist = np.array(calib_data["dist_coeffs"])
                    
                    calib_w = calib_data.get("width")
                    calib_h = calib_data.get("height")
                    
                    # Proportionally adjust scale if resolution differs
                    if calib_w and calib_h and (calib_w != self.width or calib_h != self.height):
                        scale_x = self.width / calib_w
                        scale_y = self.height / calib_h
                        
                        if abs(scale_x - scale_y) > 0.03:
                            print(f"\n[WARNING] Aspect ratio mismatch! Calibration: {calib_w}x{calib_h}, Current: {self.width}x{self.height}")
                        
                        mtx[0,0] *= scale_x # fx
                        mtx[1,1] *= scale_y # fy
                        mtx[0,2] *= scale_x # ppx
                        mtx[1,2] *= scale_y # ppy
                        print(f"\n[INFO] Scaled intrinsics from {calib_w}x{calib_h} to {self.width}x{self.height} (Scale X:{scale_x:.2f}, Y:{scale_y:.2f})")

                    # Inject calibrated parameters to Marker_Detection
                    # New interface [ppx, ppy, fx, fy]
                    new_intrinsics = [mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]]
                    self.marker_detection.set_intrinsics_param(new_intrinsics)
                    self.marker_detection.set_dist_coeffs(dist)
                    
                    print(f"[INFO] --- Loaded Calibrated Intrinsics from {calib_file} ---")
                    print(f"       fx: {mtx[0,0]:.2f}, fy: {mtx[1,1]:.2f}, ppx: {mtx[0,2]:.2f}, ppy: {mtx[1,2]:.2f}")
                    print(f"       dist: {dist}")
                except Exception as e:
                    print(f"\n[ERROR] Failed to load {calib_file}: {e}")
            else:
                print(f"\n[WARNING] Calibrated Intrinsics file {calib_file} NOT FOUND. Using factory defaults.")
        
        self.temp_history = []

    def _load_all_configs(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        setting_config_path = os.path.join(base_dir, "config", "setting.yaml")
        if not os.path.exists(setting_config_path):
            setting_config_path = os.path.join(os.path.dirname(base_dir), "config", "setting.yaml")
            
        try:
            with open(setting_config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
                
            camera_config = config_data.get("camera", {})
            yaml_device_name = camera_config.get("device_name")
            connected_device_name = getattr(self.camera, 'device_name', None)
            
            self.camera_model = None
            if connected_device_name:
                known_models = ["D405", "D435I", "D435", "D455", "D415"]
                for model in known_models:
                    if model.lower() in connected_device_name.lower():
                        self.camera_model = model
                        break
            
            if self.camera_model and self.camera_model != yaml_device_name:
                print(f"[INFO] Connected camera '{connected_device_name}' (matched as '{self.camera_model}') differs from setting.yaml '{yaml_device_name}'. Updating...")
                extrinsics_file = os.path.join(os.path.dirname(setting_config_path), "camera_extrinsics.yaml")
                if os.path.exists(extrinsics_file):
                    with open(extrinsics_file, "r") as ef:
                        extrinsics_data = yaml.safe_load(ef) or {}
                    
                    if self.camera_model in extrinsics_data:
                        cam_ext = extrinsics_data[self.camera_model]
                        camera_config["device_name"] = self.camera_model
                        camera_config["head_base_to_cam"] = cam_ext.get("head_base_to_cam", [0.0, 0.0, 0.0, -90.0, 0.0, -90.0])
                        camera_config["mount_to_cam"] = cam_ext.get("mount_to_cam", [0.0, 0.0, 0.0, -90.0, 0.0, -90.0])
                        camera_config["camera_mount_link"] = cam_ext.get("camera_mount_link", "link_head_2")
                        config_data["camera"] = camera_config
                        
                        with open(setting_config_path, "w") as wf:
                            yaml.safe_dump(config_data, wf, default_flow_style=False, sort_keys=False)
                        print(f"[INFO] Updated setting.yaml extrinsics for {self.camera_model} from camera_extrinsics.yaml")
                    else:
                        print(f"[WARNING] Match '{self.camera_model}' not found in camera_extrinsics.yaml")
                else:
                    print(f"[WARNING] camera_extrinsics.yaml not found at {extrinsics_file}")
            
            self.camera_config = camera_config
            self.markers_config = config_data.get("marker", config_data)
            self.marker_detection.markers_config = self.markers_config
            print(f"- Loaded Setting Config from {os.path.basename(setting_config_path)}")
            
            # Check camera intrinsics model mismatch
            self.intrinsics_mismatch = False
            self.calib_device_name = ""
            calib_file = os.path.join(base_dir, "config", "camera_intrinsics.yaml")
            if not os.path.exists(calib_file):
                calib_file = os.path.join(os.path.dirname(base_dir), "config", "camera_intrinsics.yaml")
            if os.path.exists(calib_file):
                try:
                    with open(calib_file, "r") as f:
                        calib_data = yaml.safe_load(f) or {}
                    self.calib_device_name = calib_data.get("device_name", "")
                    if self.calib_device_name and self.camera_model and self.calib_device_name.lower() != self.camera_model.lower():
                        self.intrinsics_mismatch = True
                        print(f"[WARNING] Camera intrinsics model mismatch detected (Connected: {self.camera_model}, Calibrated: {self.calib_device_name})")
                except Exception as e:
                    print(f"[WARNING] Failed to parse camera_intrinsics.yaml: {e}")
        except Exception as e:
            print(f"- Warning: Could not load {setting_config_path}: {e}")
            self.camera_config = {}
            self.markers_config = {}
    def set_marker_type(self, marker_type="plate"):
        self.marker_detection.set_marker_type(marker_type)
    def make_transform(self, data):
        # data: [x, y, z, roll, pitch, yaw] (x,y,z in meters, r,p,y in degrees)
        x, y, z = data[0]*1000, data[1]*1000, data[2]*1000 
        roll = data[3] * math.pi / 180
        pitch = data[4] * math.pi / 180
        yaw = data[5] * math.pi / 180
        
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)
        
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = cy * cp
        m[0, 1] = sr * sp * cy - cr * sy
        m[0, 2] = cr * sp * cy + sr * sy
        m[0, 3] = x
        
        m[1, 0] = sy * cp
        m[1, 1] = sr * sp * sy + cr * cy
        m[1, 2] = cr * sp * sy - sr * cy
        m[1, 3] = y
        
        m[2, 0] = -sp
        m[2, 1] = cp * sr
        m[2, 2] = cp * cr
        m[2, 3] = z
        
        return m

    def calc_cam_to_tool(self, camera_to_marker_tf, side="left"):
        try:
            target_tf = self.Tf_to_marker_tf_left if side == "left" else self.Tf_to_marker_tf_right
            # target_tf is in meters now. camera_to_marker_tf is also in meters.
            tf_to_marker_inv = np.linalg.inv(target_tf)
        
            if tcpip_send:
                cam_to_tool_tf = camera_to_marker_tf @ tf_to_marker_inv
            else:
                cam_to_tool_tf = camera_to_marker_tf
            cam_to_tool_vec = cam_to_tool_tf.flatten()
            
            cam_to_tool_vec[3] /= 1000
            cam_to_tool_vec[7] /= 1000
            cam_to_tool_vec[11] /= 1000
            if tcpip_send and len(cam_to_tool_vec) > 0:
                self.marker_detection.tcp_client.send_pose(cam_to_tool_vec) 
            return cam_to_tool_vec
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot invert")
            return None

    def get_marker_transform(self, sampling_time=0, side="left", use_filter=None):
        if use_filter is None:
            use_filter = (sampling_time == 0)
        lpf = False
        # Collection array for sampling -> dict of lists
        collected_transforms = {} # { marker_id: [tf_vectors...] }
        sampled_temps = []
        start_time = time.time()

        if sampling_time > 0:
            self.marker_detection.prev_pts_dict = {}
            lpf = True

        while True:
            try:
                if not self.camera.camera_monitoring:
                    self.camera.capture_image()
                color_img = self.camera.get_color_image()
                depth_img = self.camera.get_depth_image()
                if color_img is None:
                    time.sleep(0.01)
                    if sampling_time == 0 : return None
                    continue
                
                marker_transforms = self.marker_detection.detect(color_img, lpf=lpf, depth_image=depth_img, use_filter=use_filter)
                for marker_id_or_group, tf_list in marker_transforms:
                    if marker_id_or_group not in collected_transforms:
                        collected_transforms[marker_id_or_group] = []
                    collected_transforms[marker_id_or_group].append(tf_list)
                # -------------------------------------------------------------
                
                # Check timeout if sampling
                if sampling_time == 0 or (sampling_time > 0 and (time.time() - start_time > sampling_time)):
                    break
                        
            except KeyboardInterrupt:
                raise
            
            # Small sleep to reduce CPU utilization
            time.sleep(0.01)
            
        final_results = {}
        # Post-processing for sampling
        if sampling_time > 0:
            if not collected_transforms:
                return None
            
            for marker_id, tfs in collected_transforms.items():
                data = np.array(tfs) # Shape (N, 16)
                
                # Separate translation and rotation for CAMERA_TO_MARKER (NOT inverted yet)
                translations = data[:, [3, 7, 11]]
                
                # Median for translation is robust
                final_translation = np.median(translations, axis=0)
                
                # Average rotations using SVD (chordal L2 mean) to maintain orthogonality
                rotations = []
                for vec in data:
                    R = np.array([
                        [vec[0], vec[1], vec[2]],
                        [vec[4], vec[5], vec[6]],
                        [vec[8], vec[9], vec[10]]
                    ])
                    rotations.append(R)
                
                sum_R = np.sum(rotations, axis=0)
                U, S, Vt = np.linalg.svd(sum_R)
                final_R = U @ Vt
                
                # Ensure det(R) = 1 (proper rotation)
                if np.linalg.det(final_R) < 0:
                    U[:, 2] *= -1
                    final_R = U @ Vt
                
                avg_cam_to_marker_tf = np.eye(4, dtype=np.float32)
                avg_cam_to_marker_tf[0:3, 0:3] = final_R
                avg_cam_to_marker_tf[0:3, 3] = final_translation
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                
                cam_to_tool_vec = self.calc_cam_to_tool(avg_cam_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        elif sampling_time == 0:
            for marker_id, tfs in collected_transforms.items():
                # For sampling_time == 0, there is only one frame of transforms
                # and thus tfs should have only length 1
                camera_to_marker_tf = np.array(tfs[-1], dtype=np.float32).reshape(4, 4)
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                cam_to_tool_vec = self.calc_cam_to_tool(camera_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        
        if len(final_results) > 0:
            if self.marker_detection.marker_type == "plate":
                if side == "left":
                    res = final_results.get("plate_left")
                    return [res] if res is not None else None
                elif side == "right":
                    res = final_results.get("plate_right")
                    return [res] if res is not None else None
                elif side == "all":
                    out = []
                    res_r = final_results.get("plate_right")
                    res_l = final_results.get("plate_left")
                    if res_r is not None: out.append(res_r)
                    if res_l is not None: out.append(res_l)
                    return out if out else None
            return final_results
        else:
            return None
