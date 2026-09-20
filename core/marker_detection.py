from core.storage import ConfigStorage
from core.storage import FileStorage
from core.camera_processing import RealSenseCamera, CameraUnavailableError
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading
import sys
import os
import re, yaml

#debugging flag : must be all false in production
imshow_when_detect = False
tcpip_send = False
# Which values from the calibrated intrinsics file (camera_intrinsics.yaml) override the factory ones:
#   "off"             : factory fx, fy, cx, cy and distortion
#   "principal_point" : calibrated cx, cy only (factory focal length and distortion kept)
#   "full"            : calibrated fx, fy, cx, cy and distortion
# The principal point sets where the optical axis is in the image, so it directly biases the head
# tilt/pan pointing estimate (2026-09-15: factory cy 347.9 vs calibrated 350.6 px = 0.24 deg of tilt).
calib_intrinsics_mode = "full"
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
        self.prev_pnp_rot = {}

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

    def reset_tracking(self):
        self.prev_pts_dict = {}
        self.prev_pnp_rot = {}

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

                # --- IPPE Planar Ambiguity Disambiguation ---
                pnp_res = cv2.solvePnPGeneric(obj_pts, c, cam_mat, pnp_dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                num_sol = pnp_res[0] if len(pnp_res) > 0 else 0
                if num_sol == 0 or len(pnp_res[1]) == 0:
                    continue
                
                rvecs = pnp_res[1]
                tvecs = pnp_res[2]
                reproj_errs = pnp_res[3] if len(pnp_res) > 3 else None

                # Pick the lower-reprojection-error solution independently per frame (same as
                # cv2.solvePnP(SOLVEPNP_IPPE), used up to 2026-09-07). Choosing the solution
                # closest to the previous frame (introduced 2026-09-13) locks onto a wrong
                # mirrored solution for whole runs of frames near a fronto-parallel view, which
                # the per-frame spike filter in CalibratorBase.filter_sweep_dataset cannot remove.
                best_idx = 0
                if num_sol > 1 and reproj_errs is not None and len(reproj_errs) > 1:
                    best_idx = int(np.argmin(np.asarray(reproj_errs).ravel()))
                rvec = rvecs[best_idx]
                tvec = tvecs[best_idx]
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
    def __init__(self, serial_number=None, *, sim=False, robot=None, robot_version='1.2', camera_factory=RealSenseCamera):
        self.sim = bool(sim)
        self.camera = None
        self.robot = robot
        self.robot_version = str(robot_version).removeprefix('v')
        self.rng = np.random.default_rng(42)
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

        if self.sim:
            self.simulation_model = SimulationModel.create(self.robot_version)
        else:
            try:
                self.camera = camera_factory(serial_number=serial_number)
            except (CameraUnavailableError, RuntimeError) as e:
                print(f"[WARN] Camera unavailable: {e}. Falling back to simulation model.")
                self.sim = True
                self.simulation_model = SimulationModel.create(self.robot_version)

            if self.camera is not None:
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
                if calib_intrinsics_mode in ("principal_point", "full"):
                    from core.storage import CONFIG_PATHS
                    calib_file = CONFIG_PATHS.get("camera_intrinsics")
                    if not calib_file or not os.path.exists(calib_file):
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        calib_file = os.path.join(base_dir, "config", "camera_intrinsics.yaml")
                        if not os.path.exists(calib_file):
                            calib_file = os.path.join(os.path.dirname(base_dir), "config", "camera_intrinsics.yaml")
                    if os.path.exists(calib_file):
                        try:
                            calib_data = ConfigStorage.load(calib_file)
                            
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
                            # Interface [ppx, ppy, fx, fy]
                            factory = list(intrinsics)
                            if calib_intrinsics_mode == "full":
                                new_intrinsics = [mtx[0,2], mtx[1,2], mtx[0,0], mtx[1,1]]
                                self.marker_detection.set_dist_coeffs(dist)
                            else:
                                new_intrinsics = [mtx[0,2], mtx[1,2], factory[2], factory[3]]
                            self.marker_detection.set_intrinsics_param(new_intrinsics)

                            print(f"[INFO] --- Calibrated intrinsics ({calib_intrinsics_mode}) from {calib_file} ---")
                            print(f"       factory   : fx {factory[2]:.2f}, fy {factory[3]:.2f}, ppx {factory[0]:.2f}, ppy {factory[1]:.2f}")
                            print(f"       in use    : fx {new_intrinsics[2]:.2f}, fy {new_intrinsics[3]:.2f}, ppx {new_intrinsics[0]:.2f}, ppy {new_intrinsics[1]:.2f}")
                            if calib_intrinsics_mode == "full":
                                print(f"       dist: {dist}")
                        except Exception as e:
                            print(f"\n[ERROR] Failed to load {calib_file}: {e}")
                    else:
                        print(f"\n[WARNING] Calibrated Intrinsics file {calib_file} NOT FOUND. Using factory defaults.")
                
                # Always default to Auto Exposure on initialization
                self.camera.set_exposure(6000.0, auto_exposure=True)

        self.temp_supported = bool(self.camera is not None and getattr(self, 'temp_supported', False))
        self.temp_history = []
        self.set_marker_type("plate")

    def bind_robot(self, robot, robot_version):
        version = str(robot_version).removeprefix('v')
        if self.sim and (getattr(self, 'simulation_model', None) is None or self.simulation_model.version != version):
            self.simulation_model = SimulationModel.create(version)
            self.rng = np.random.default_rng(self.simulation_model.config.get('seed', 42))
        self.robot, self.robot_version = robot, version

    def set_marker_type(self, marker_type):
        if self.marker_detection is not None:
            self.marker_detection.set_marker_type(marker_type)

    def set_camera_exposure(self, exposure_val, auto_exposure=False):
        if self.camera is None: return False
        return self.camera.set_exposure(exposure_val, auto_exposure)

    def get_camera_exposure(self):
        if self.camera is None: return True, 6000.0
        return self.camera.get_exposure()

    def get_actual_exposure(self):
        if self.camera is None: return 6000.0
        return self.camera.get_actual_exposure()

    def _load_all_configs(self):
        from core.storage import CONFIG_PATHS
        setting_config_path = CONFIG_PATHS.get("setting_yaml")
        if not setting_config_path or not os.path.exists(setting_config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            setting_config_path = os.path.join(base_dir, "config", "setting.yaml")
            if not os.path.exists(setting_config_path):
                setting_config_path = os.path.join(os.path.dirname(base_dir), "config", "setting.yaml")
            
        try:
            config_data = ConfigStorage.load(setting_config_path)
                
            camera_config = config_data.get("camera", {})
            yaml_device_name = camera_config.get("device_name")
            connected_device_name = getattr(self.camera, 'device_name', None)
            
            self.camera_model = parse_camera_model(connected_device_name)
            
            self.temp_supported = False
            info_file = CONFIG_PATHS.get("camera_info")
            if not info_file or not os.path.exists(info_file):
                info_file = os.path.join(os.path.dirname(setting_config_path), "camera_info.yaml")
                if not os.path.exists(info_file) and getattr(sys, 'frozen', False):
                    info_file = os.path.join(sys._MEIPASS, "config", "camera_info.yaml")

            info_data = {}
            if os.path.exists(info_file):
                try:
                    info_data = ConfigStorage.load(info_file)
                except Exception as e:
                    print(f"[WARNING] Failed to read camera_info.yaml: {e}")

            target_model = self.camera_model or yaml_device_name
            matched_info_key = lookup_camera_info_key(info_data, target_model)

            if matched_info_key:
                self.temp_supported = info_data[matched_info_key].get("temp_supported", False)
                cam_ext = info_data[matched_info_key]
                info_head_base = cam_ext.get("head_base_to_cam")
                info_mount = cam_ext.get("mount_to_cam")
                info_mount_link = cam_ext.get("camera_mount_link", "link_head_2")

                current_dev = camera_config.get("device_name")
                current_head_base = camera_config.get("head_base_to_cam")
                current_mount = camera_config.get("mount_to_cam")
                current_mount_link = camera_config.get("camera_mount_link")

                def is_diff(v1, v2):
                    if v1 is None or v2 is None:
                        return v1 != v2
                    try:
                        return not np.allclose(v1, v2, atol=1e-5)
                    except Exception:
                        return v1 != v2

                dev_diff = (self.camera_model is not None and current_dev != self.camera_model)
                pos_diff = is_diff(current_head_base, info_head_base) or is_diff(current_mount, info_mount) or (current_mount_link != info_mount_link)
                missing = current_head_base is None or current_mount is None or current_mount_link is None

                # camera_info.yaml holds each camera model's CAD nominal. Load it only when the camera
                # model actually changed or setting.yaml has no extrinsics yet. A value that merely
                # differs from the nominal is a calibrated (Step 2 Apply) or hand-edited one and must
                # survive a restart -- overwriting it on every launch silently threw both away.
                if not (dev_diff or missing) and pos_diff:
                    print(f"[INFO] setting.yaml camera extrinsics differ from the '{matched_info_key}' nominal in "
                          f"camera_info.yaml; keeping setting.yaml (calibrated or edited values).")

                if dev_diff or missing:
                    if dev_diff:
                        print(f"[INFO] Connected camera '{connected_device_name}' (matched as '{self.camera_model}') differs from setting.yaml '{yaml_device_name}'. Loading its nominal extrinsics...")
                    if missing:
                        print(f"[INFO] setting.yaml has no camera extrinsics yet. Loading the '{matched_info_key}' nominal from camera_info.yaml...")

                    camera_config["device_name"] = self.camera_model or matched_info_key
                    changes = {("camera", "device_name"): camera_config["device_name"]}
                    if info_head_base is not None:
                        camera_config["head_base_to_cam"] = info_head_base
                        changes[("camera", "head_base_to_cam")] = list(info_head_base)
                    if info_mount is not None:
                        camera_config["mount_to_cam"] = info_mount
                        changes[("camera", "mount_to_cam")] = list(info_mount)
                    if info_mount_link is not None:
                        camera_config["camera_mount_link"] = info_mount_link
                        changes[("camera", "camera_mount_link")] = info_mount_link
                    config_data["camera"] = camera_config
                    ConfigStorage.update_values(setting_config_path, changes)
                    print(f"[INFO] Updated setting.yaml extrinsics for {matched_info_key} from camera_info.yaml (head_base_to_cam: {camera_config.get('head_base_to_cam')}, mount_to_cam: {camera_config.get('mount_to_cam')})")
            else:
                if not os.path.exists(info_file):
                    print(f"[WARNING] camera_info.yaml not found at {info_file}")
                elif target_model:
                    print(f"[WARNING] Match '{target_model}' not found in camera_info.yaml")
            
            self.camera_config = camera_config
            self.markers_config = config_data.get("marker", {}) or {}
            # Legacy fallback
            for lk in ["Tf_to_marker_left", "Tf_to_marker_right", "Tf_to_marker_left_v12", "Tf_to_marker_right_v12", "Tf_to_marker_left_v13", "Tf_to_marker_right_v13"]:
                if lk not in self.markers_config and lk in camera_config:
                    self.markers_config[lk] = camera_config[lk]
            self.marker_detection.markers_config = self.markers_config
            print(f"- Loaded Setting Config from {os.path.basename(setting_config_path)}")
            print(f"  * head_base_to_cam: {camera_config.get('head_base_to_cam')}")
            print(f"  * mount_to_cam: {camera_config.get('mount_to_cam')}")
            
            # Check camera intrinsics model mismatch
            self.intrinsics_mismatch = False
            self.calib_device_name = ""
            calib_file = CONFIG_PATHS.get("camera_intrinsics")
            if not calib_file or not os.path.exists(calib_file):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                calib_file = os.path.join(base_dir, "config", "camera_intrinsics.yaml")
                if not os.path.exists(calib_file):
                    calib_file = os.path.join(os.path.dirname(base_dir), "config", "camera_intrinsics.yaml")
            if os.path.exists(calib_file):
                try:
                    calib_data = ConfigStorage.load(calib_file)
                    self.calib_device_name = calib_data.get("device_name", "")
                    if (self.calib_device_name and self.camera_model
                            and camera_model_family(self.calib_device_name) != camera_model_family(self.camera_model)):
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
            # target_tf is in meters. camera_to_marker_tf is also in meters.
            tf_to_marker_inv = np.linalg.inv(target_tf)
        
            if tcpip_send:
                cam_to_tool_tf = camera_to_marker_tf @ tf_to_marker_inv
            else:
                cam_to_tool_tf = camera_to_marker_tf
            cam_to_tool_vec = cam_to_tool_tf.flatten()
            
            if tcpip_send and len(cam_to_tool_vec) > 0:
                self.marker_detection.tcp_client.send_pose(cam_to_tool_vec) 
            return cam_to_tool_vec
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot invert")
            return None

    def get_marker_transform(self, sampling_time=0, side="left", use_filter=None, *, q_encoder=None):
        if not np.isfinite(sampling_time) or sampling_time < 0:
            raise ValueError('sampling_time must be finite and nonnegative')
        if self.sim and self.robot is None and q_encoder is None:
            raise RuntimeError('A connected SDK robot or q_encoder is required for simulated observations')
        if use_filter is None:
            use_filter = (sampling_time == 0)
        lpf = False
        # Collection array for sampling -> dict of lists
        collected_transforms = {} # { marker_id: [tf_vectors...] }
        sampled_temps = []
        start_time = time.monotonic()

        if sampling_time > 0:
            self.marker_detection.prev_pts_dict = {}
            self.marker_detection.prev_pnp_rot = {}
            lpf = True

        while True:
            try:
                if self.sim:
                    if q_encoder is not None:
                        q = np.asarray(q_encoder)
                    elif self.robot is not None:
                        q = np.asarray(self.robot.get_state().position)
                    else:
                        q = np.zeros(26)
                    sides = ['right', 'left'] if side == 'all' else [side]
                    marker_transforms = [
                        ('plate_' + arm, self.simulation_model.marker_pose(self.robot, q, arm, self.rng).ravel())
                        for arm in sides]
                else:
                    if self.camera is None:
                        time.sleep(0.01)
                        if sampling_time == 0: return None
                        continue
                    if not self.camera.camera_monitoring:
                        self.camera.capture_image()
                    color_img = self.camera.get_color_image()
                    depth_img = self.camera.get_depth_image()
                    if color_img is None:
                        time.sleep(0.01)
                        if sampling_time == 0: return None
                        continue
                    
                    raw_transforms = self.marker_detection.detect(color_img, lpf=lpf, depth_image=depth_img, use_filter=use_filter)
                    # Real detector coordinates are millimetres. Normalize to meters at observation boundary.
                    normalized = []
                    for marker_id, values in raw_transforms:
                        transform = np.asarray(values, dtype=float).reshape(4, 4).copy()
                        transform[:3, 3] /= 1000.0
                        normalized.append((marker_id, transform.ravel()))
                    marker_transforms = normalized

                for marker_id_or_group, tf_list in marker_transforms:
                    if marker_id_or_group not in collected_transforms:
                        collected_transforms[marker_id_or_group] = []
                    collected_transforms[marker_id_or_group].append(tf_list)
                
                # Check timeout if sampling
                if sampling_time == 0 or (sampling_time > 0 and (time.monotonic() - start_time >= sampling_time)):
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
                
                avg_cam_to_marker_tf = np.eye(4, dtype=float)
                avg_cam_to_marker_tf[0:3, 0:3] = final_R
                avg_cam_to_marker_tf[0:3, 3] = final_translation
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                
                cam_to_tool_vec = self.calc_cam_to_tool(avg_cam_to_marker_tf, side=calc_side)
                if cam_to_tool_vec is not None:
                    final_results[marker_id] = cam_to_tool_vec
        elif sampling_time == 0:
            for marker_id, tfs in collected_transforms.items():
                camera_to_marker_tf = np.array(tfs[-1], dtype=float).reshape(4, 4)
                
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


from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def camera_model_family(name):
    """'Intel RealSense D435i' -> 'D435'. Variants of one model (D435i has an IMU, D435f an IR
    filter) share the housing and optics, so they share the mount-to-camera extrinsics; only the
    number identifies the body. D405 and D455 are different housings and stay separate."""
    if not name:
        return None
    match = re.search(r"[Dd](\d{3})", str(name))
    return f"D{match.group(1)}" if match else None


def parse_camera_model(device_name):
    """Full model as reported, e.g. 'D435I' for an 'Intel RealSense D435i'."""
    if not device_name:
        return None
    match = re.search(r"[Dd](\d{3})([A-Za-z]*)", str(device_name))
    if not match:
        return None
    return f"D{match.group(1)}{match.group(2).upper()}"


def lookup_camera_info_key(info_data, target_model):
    """Exact entry for this model if camera_info.yaml has one, otherwise the model family.

    Letting the family answer means a D435f (or any future variant) works without its own entry,
    and a variant only needs an entry when its numbers genuinely differ.
    """
    if not target_model or not info_data:
        return None
    for key in info_data:
        if key.lower() == str(target_model).lower():
            return key
    family = camera_model_family(target_model)
    if family:
        for key in info_data:
            if key.lower() == family.lower():
                return key
    return None


def load_truth_config():
    from core.storage import CONFIG_PATHS
    sim_path = CONFIG_PATHS.get('simulation_yaml')
    if not sim_path or not os.path.exists(sim_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(base_dir, 'config', 'simulation.yaml')
        if not os.path.exists(sim_path):
            sim_path = os.path.join(os.path.dirname(base_dir), 'config', 'simulation.yaml')
    with FileStorage.open(sim_path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def uses_head_camera(camera_config, model):
    mode = camera_config.get('camera_mount_mode', 'head')
    if mode not in ('head', 'fixed'):
        raise ValueError('camera_mount_mode must be head or fixed')
    return mode == 'head' and len(getattr(model, 'head_idx', [])) >= 2


@dataclass(frozen=True)
class SimulationModel:
    version: str
    config_json: str

    @classmethod
    def create(cls, version='1.2', config=None):
        config = load_truth_config() if config is None else config
        version = str(version).removeprefix('v')
        if version not in config['brackets']:
            raise ValueError(f'No simulation geometry for robot v{version}')
        return cls(version, json.dumps(config, sort_keys=True, allow_nan=False))

    @property
    def config(self):
        return json.loads(self.config_json)

    def arm_offsets(self, side):
        values = self.config['offsets'][side]
        return np.deg2rad([values[f'joint{i}' if i != 5 else
                                 ('joint5_v13' if self.version == '1.3' else 'joint5_v12')]
                          for i in range(7)])

    def bracket_transform(self, side):
        try:
            from core.calibration.calibration_optimizer import make_transform
        except ImportError:
            from calibration.calibration_optimizer import make_transform
        cfg = self.config
        nominal = make_transform(cfg['brackets'][self.version][side])
        error = cfg['offsets'][side]
        nominal[:3, 3] += np.asarray(error['bracket_pos'])
        nominal[:3, :3] = make_transform([0, 0, 0] + error['bracket_rpy'])[:3, :3] @ nominal[:3, :3]
        return nominal

    def marker_pose(self, robot, q_encoder, side, rng=None, noisy=True):
        try:
            from core.calibration.calibration_optimizer import compute_fk, make_transform, so3_exp
        except ImportError:
            from calibration.calibration_optimizer import compute_fk, make_transform, so3_exp
        cfg, model = self.config, robot.model() if robot else None
        q = np.array(q_encoder, dtype=float, copy=True)
        if model:
            q[getattr(model, f'{side}_arm_idx')] += self.arm_offsets(side)
            head = uses_head_camera(cfg, model)
            if head:
                offsets = cfg['offsets']['head']
                q[model.head_idx] += np.deg2rad([offsets['pan'], offsets['tilt']])
        base = 'link_head_2' if (model and uses_head_camera(cfg, model)) else 'link_head_0'
        camera = make_transform(cfg['mount_to_cam' if (model and uses_head_camera(cfg, model)) else 'head_base_to_cam'])
        if robot:
            _, fk = compute_fk(robot, robot.get_dynamics(), q, f'ee_{side}', base_link=base)
            result = np.linalg.inv(camera) @ fk @ self.bracket_transform(side)
        else:
            result = np.eye(4)
        if noisy:
            if rng is None:
                raise ValueError('A session RNG is required for reproducible sensor noise')
            result[:3, 3] += rng.normal(0, cfg['position_noise_std_m'], 3)
            result[:3, :3] = so3_exp(np.deg2rad(rng.normal(0, cfg['orientation_noise_std_deg'], 3))) @ result[:3, :3]
        return result

    def metadata(self, urdf_path=None):
        return {'schema_version': 1, 'source': 'simulation_pose_sensor',
                'robot_version': self.version, 'truth': self.config,
                'truth_sha256': hashlib.sha256(self.config_json.encode()).hexdigest(),
                'urdf_sha256': hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest() if urdf_path else None,
                'offset_convention': 'q_physical = q_encoder + delta; correction = -delta',
                'image_detection_verified': False}
