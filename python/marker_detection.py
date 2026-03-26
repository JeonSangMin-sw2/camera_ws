import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading
import os, yaml
from datetime import datetime

#debugging flag : 실사용시 모두 false여야함
imshow_when_detect = False
check_cube_marker_data = False
tcpip_send = False

# 유틸리티 클래스
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

# 데이터를 텍스트 파일로 저장하는 클래스. 디버깅을 위함
class File_Logger:
    def __init__(self, filepath=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "log", "marker_log")
        
        if filepath is None or filepath == "log.txt":
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filepath = os.path.join(log_dir, f"{now_str}.txt")
        else:
            self.filepath = os.path.join(log_dir, filepath)

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")

class CubeMarker:
    def __init__(self, cube_size_mm=60.0, side="left"):
        # Dictionary to store Cube to Marker (CUBE_TO_MARKER) transformation matrices 
        # for left arm (10~14) and right arm (30~34)
        self.cube_to_marker_tf = {}
        half_c = cube_size_mm / 2.0
        
        # Id 10 / 30: +Y Face (Physical orientation derived from photo: bottom corner is c[0])
        R_10 = np.array([
            [-1,  0,  0],
            [ 0,  0, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        t_10 = np.array([0, half_c, 0], dtype=np.float32)
        
        # Id 11 / 31: -X Face
        R_11 = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ], dtype=np.float32)
        t_11 = np.array([-half_c, 0, 0], dtype=np.float32)
        
        # Id 12 / 32: -Z Face
        R_12 = np.array([
            [ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]
        ], dtype=np.float32)
        t_12 = np.array([0, 0, -half_c], dtype=np.float32)
        
        # Id 13 / 33: +X Face (Physical orientation derived from photo)
        R_13 = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0, -1,  0]
        ], dtype=np.float32)
        t_13 = np.array([half_c, 0, 0], dtype=np.float32)
        
        # Id 14 / 34: +Z Face (Physical orientation derived from photo)
        R_14 = np.array([
            [ 0,  1,  0],
            [ 1,  0,  0],
            [ 0,  0, -1]
        ], dtype=np.float32)
        t_14 = np.array([0, 0, half_c], dtype=np.float32)

        if side == "left":
            self.cube_to_marker_tf[10] = self._make_tf(R_10, t_10)
            self.cube_to_marker_tf[11] = self._make_tf(R_11, t_11)
            self.cube_to_marker_tf[12] = self._make_tf(R_12, t_12)
            self.cube_to_marker_tf[13] = self._make_tf(R_13, t_13)
            self.cube_to_marker_tf[14] = self._make_tf(R_14, t_14)
        elif side == "right":
            self.cube_to_marker_tf[30] = self._make_tf(R_10, t_10)
            self.cube_to_marker_tf[31] = self._make_tf(R_11, t_11)
            self.cube_to_marker_tf[32] = self._make_tf(R_12, t_12)
            self.cube_to_marker_tf[33] = self._make_tf(R_13, t_13)
            self.cube_to_marker_tf[34] = self._make_tf(R_14, t_14)
        else:
            raise ValueError("side must be 'left' or 'right'")

    def load_calibration_data(self, calib_data, marker_size_mm=64.0):
        try:
            half_m = marker_size_mm / 2.0
            local_corners = np.array([
                [-half_m, -half_m, 0],
                [ half_m, -half_m, 0],
                [ half_m,  half_m, 0],
                [-half_m,  half_m, 0]
            ], dtype=np.float32)
            c_local = np.mean(local_corners, axis=0)
            local_centered = local_corners - c_local
        
            for m_id_str, corners_list in calib_data.items():
                m_id = int(m_id_str)
                opt_corners = np.array(corners_list, dtype=np.float32)
                
                c_opt = np.mean(opt_corners, axis=0)
                opt_centered = opt_corners - c_opt
                
                H = local_centered.T @ opt_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                if np.linalg.det(R) < 0:
                    Vt[2, :] *= -1
                    R = Vt.T @ U.T
                    
                t = c_opt - R @ c_local
                
                self.cube_to_marker_tf[m_id] = self._make_tf(R, t)
        except Exception as e:
            print(f"- Error formatting calibration data: {e}")

    def _make_tf(self, R, t):
        tf = np.eye(4, dtype=np.float32)
        tf[0:3, 0:3] = R
        tf[0:3, 3] = t
        return tf
        
    def get_transform(self, marker_id):
        if marker_id in self.cube_to_marker_tf:
            return self.cube_to_marker_tf[marker_id]
        return None
        
# 카메라 클래스
"""
해당 클래스는 realsense 카메라만 호환
다른 카메라를 사용할 경우 아래의 get이란 이름이 들어간 함수들의 양식을 동일하게 맞춰주고 구현하는것을 권장
"""
class RealSenseCamera:
    # serial_number : 해당 넘버의 카메라를 사용, 입력하지 않으면 첫번째 카메라 사용
    # serial_number : 해당 넘버의 카메라를 사용, 입력하지 않으면 첫번째 카메라 사용
    """카메라의 시리얼 넘버는 realsense_check.py를 통해 검색할 수 있음"""
    def __init__(self, serial_number=None):
        # 연결되어있는 카메라 검색
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise RuntimeError("No RealSense connected")
        # 카메라 선택 : 시리얼 넘버가 정해져있으면 해당 카메라 사용, 없으면 첫번째 카메라 사용
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number) or serial_number is None:
                self.device_number = i
                break

        # 안전한 사용을 위해 선택한 카메라 재연결
        print("Resetting Realsense device...")
        devices[self.device_number].hardware_reset()
        # 카메라가 다시 연결될 때까지 대기
        time.sleep(3)

        # 카메라 정보 재확인(하드웨어 리셋했으므로 재확인 필요)
        ctx = rs.context()
        devices = ctx.query_devices()

        self.device_name = devices[self.device_number].get_info(rs.camera_info.name)
        self.serial_number = devices[self.device_number].get_info(rs.camera_info.serial_number)
        print("Using camera is : ", self.device_name)

        # depth 해상도 확인 : D435는 1mm, D405 는 0.1mm. 즉 모델에 따라 다름
        depth_sensor = devices[self.device_number].first_depth_sensor()
        if depth_sensor.supports(rs.option.thermal_compensation):
            depth_sensor.set_option(rs.option.thermal_compensation, 1.0) # 온도보정기능 On
            depth_sensor.set_option(rs.option.visual_preset, 3) # High Accuracy
        depth_scale = depth_sensor.get_depth_scale()
        print("depth scale : ", depth_scale)
        

        # 카메라 구동을 위한 파라미터 설정
        self.depth_resolution = depth_scale*1000                # 각 픽셀의 depth 값 해상도
        self.pipeline = rs.pipeline()                           # 카메라 스트리밍을 위한 파이프라인
        self.config = rs.config()                               # 카메라 설정구조체
        self.spatial = rs.spatial_filter()                      # 평활화 필터 (노이즈 제거)
        self.spatial.set_option(rs.option.filter_magnitude, 2)  # 평활화 필터의 강도 설정
        self.temporal = rs.temporal_filter()                    # 깜빡임 방지필터
        self.hole_filling = rs.hole_filling_filter()            # 노이즈에 의한 빈 공간 채우는 필터

        # 상태 flag
        self.camera_running = False                             # 카메라 구동상태 flag
        self.camera_monitoring = False                          # 카메라 모니터링 상태 flag

        #이미지 저장 변수
        self.color_image = None
        self.depth_image = None
        # 기본해상도
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30

        #카메라 내부 파라미터 : 해당 파라미터로 depth map 및 마커좌표 계산됨
        self.fx = 0.0                                           # 초점거리 x
        self.fy = 0.0                                           # 초점거리 y
        self.principal_point = [0.0, 0.0]                       # 주점(이미지 중심)
        self.intrinsics = None                                  # 내부 파라미터 행렬
        self.profile = None                                     # 카메라 프로파일
        self.baseline = 0.065                                   # 스테레오 카메라 베이스라인(m)

        # 스레드 동기화를 위한 Lock
        self.lock = threading.Lock()
        self.thread = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        
        try:
            self.config.enable_device(self.serial_number)
            # 사용할 카메라(컬러, depth, ir) 스트리밍 활성화. depth는 모든상황에 사용됨
            # cpu 부하를 줄이기 위해 적외선 사용 시 ir1, ir2 를, 사용하지 않을 경우 color만 추가로 스트리밍 진행
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # 파이프라인 시작
            self.profile = self.pipeline.start(self.config)

            # 구동 후 카메라 안정화를 위해 10프레임 정도는 버리고 사용
            for i in range(10):
                self.pipeline.wait_for_frames()
            
            # depth 카메라 내부 파라미터 얻기 : baseline, fx, fy, principal_point 를 위해 사용
            # depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            # self.depth_intrinsics = depth_stream.get_intrinsics()

            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_stream.get_intrinsics()
            self.fx = self.intrinsics.fx
            self.fy = self.intrinsics.fy
            self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy] #pixel

            print(f"Focal Length: fx={self.fx}, fy={self.fy}")
            print(f"Principal Point: {self.principal_point[0]}, {self.principal_point[1]}")
            self.camera_running = True
        except Exception as e:
            print(f"Camera didn't initialize: {e}")
            raise e

    # Monitoring 함수로 스트리밍을 켜고 끔
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
                        # 이미지가 한개일 때 화면 크기 기본 해상도의 1/2
                        resize_height = self.height // 2
                        resize_width = self.width // 2
                    else:
                        # 이미지가 두개 이상일 때 화면 크기 기본 해상도의 1/n
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
        # 복잡한 스레드 검사 로직(if self.thread is not None...)을 전부 삭제합니다.
        if not self.camera_running:
            return

        # 이 함수는 stream_on(백그라운드 스레드)에서만 단독으로 실행됩니다.
        try:
            frames = self.pipeline.wait_for_frames()
            
            # 카메라의 depth map을 사용할 경우 컬러 이미지와 해상도, 사이즈를 맞추기 위해 align을 사용.
            align_to = rs.stream.color
            align = rs.align(align_to)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # 추가 필터적용
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)
            if not color_frame or not depth_frame:
                print("no frame")
                return
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
        
            with self.lock:
                self.color_image = color_data
                self.depth_image = depth_data
        except Exception as e:
            pass # 간헐적인 프레임 드랍 무시


    # 연산을 위해 호출돼야하는 함수들
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
            # 만약 asic 온도를 대신 지원하는 모델일 경우 (D435 등은 둘 다 지원하거나 하나만 지원)
            elif depth_sensor.supports(rs.option.asic_temperature):
                asic_temp = depth_sensor.get_option(rs.option.asic_temperature)
                return asic_temp
        except Exception as e:
            print(f"Failed to get temperature: {e}")
            return None
        return None


class Marker_Detection:
    def __init__(self):
        # 어떤 마커를 인식시킬건지 정의
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        # 마커 인식 정밀도 향상을 위한 파라미터 튜닝
        # 1. 서브픽셀 정밀도 극대화
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01

        # 2. 이진화 세밀화 (조명 및 그림자 대응)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 3  # 세밀한 스캔
        self.parameters.adaptiveThreshConstant = 7     # 노이즈 대비 약간 낮춤

        # 3. 형상 근사 및 필터링
        self.parameters.polygonalApproxAccuracyRate = 0.01 # 더 엄격한 사각형 검사
        self.parameters.minDistanceToBorder = 3
        self.parameters.minMarkerPerimeterRate = 0.01

        # 4. 내부 비트 샘플링 향상
        self.parameters.perspectiveRemovePixelPerCell = 12 # 정밀한 비트 추출

        # 계산에 쓰일 카메라 내부 파라미터
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.depth_resolution = 1
        self.rpy = [0, 0, 0]
        
        self.lpf_alpha = 0.3
        self.prev_pts_dict = {}

        # 인식할 마커 타입과 ID
        self.marker_type = None
        self.marker_id = None
        
        self.marker_size_mm = 36.0 # 기본값, config에서 덮어씌워짐
        self.markers_config = {}
        if tcpip_send:
            self.tcp_client = TCPClient("127.0.0.1", 5000)

    # 연산에 필요한 카메라 파라미터 설정
    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.fx = param[2]
        self.fy = param[3]

    def set_depth_resolution(self, depth_resolution):
        self.depth_resolution = depth_resolution

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_marker_type(self, marker_type="plate"):
        self.marker_type = marker_type
        if marker_type == "cube":
            left_ids = self.markers_config.get("cube", {}).get("left_ids", [])
            right_ids = self.markers_config.get("cube", {}).get("right_ids", [])
            self.marker_id = left_ids + right_ids
            self.marker_size_mm = self.markers_config.get("cube", {}).get("cube_size_mm", 60.0) * 0.8
        elif marker_type == "plate":
            self.plate_left_ids = self.markers_config.get("plate", {}).get("left_ids", [])
            self.plate_right_ids = self.markers_config.get("plate", {}).get("right_ids", [])
            self.marker_id = self.plate_left_ids + self.plate_right_ids
            self.marker_size_mm = self.markers_config.get("plate", {}).get("plate_size_mm", 100.0) * 0.8
        else:
            self.marker_id = []
    # 마커들의 중심좌표(4*4행렬)
    def detect(self, color_image, depth_image, lpf = False, logging = False, current_temp = None):
        depth_filtered = depth_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        marker_centers_result = []
        

        # 등록된 마커만 필터링 (tuple과 ndarray는 pop()을 쓸 수 없으므로 새로 리스트를 만듭니다)
        if ids is not None and len(ids) > 0 and self.marker_id is not None:
            valid_indices = [i for i, mid in enumerate(ids) if mid[0] in self.marker_id]
            if len(valid_indices) > 0:
                corners = tuple(corners[i] for i in valid_indices)
                ids = np.array([ids[i] for i in valid_indices])
            else:
                corners, ids = (), None

        # debugging (필터링이 끝난 마커들만 화면에 표시되게 위치를 조정합니다)
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
                
            comp_fx = self.fx
            comp_fy = self.fy
            
            if current_temp is not None and hasattr(self, 'thermal_config') and self.thermal_config.get("enabled", False):
                baseline_temp = self.thermal_config.get("baseline_temp", 32.0)
                alpha = self.thermal_config.get("focal_scale_per_deg", 0.00016)
                pp_x_alpha = self.thermal_config.get("pp_x_shift_per_deg", 0.0)
                pp_y_alpha = self.thermal_config.get("pp_y_shift_per_deg", 0.0)
                
                delta_t = (current_temp - baseline_temp)
                scale_factor = 1.0 + alpha * delta_t
                comp_fx = self.fx * scale_factor
                comp_fy = self.fy * scale_factor
                
                comp_cx = self.principal_point[0] + pp_x_alpha * delta_t
                comp_cy = self.principal_point[1] + pp_y_alpha * delta_t
                
                cam_mat = np.array([
                    [comp_fx, 0, comp_cx],
                    [0, comp_fy, comp_cy],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                cam_mat = np.array([
                    [comp_fx, 0, self.principal_point[0]],
                    [0, comp_fy, self.principal_point[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
            half_m = self.marker_size_mm / 2.0
            
            if self.marker_type == "cube" and hasattr(self, 'cube_markers'):
                current_frame_cube_info = {}
                # 인식된 마커들을 재정렬
                for i in range(len(ids)):
                    marker_id = ids_flat[i]
                    if marker_id in duplicate_ids:
                        continue
                    c = corners[i][0]
                    group_id = "cube"
                    if 10 <= marker_id <= 14: group_id = "cube_left"
                    elif 30 <= marker_id <= 34: group_id = "cube_right"
                    
                    CUBE_TO_MARKER = None
                    if group_id in self.cube_markers:
                        CUBE_TO_MARKER = self.cube_markers[group_id].get_transform(marker_id)
                        
                    if CUBE_TO_MARKER is not None:
                        obj_pts = np.array([
                            [-half_m, -half_m, 0], [ half_m, -half_m, 0],
                            [ half_m,  half_m, 0], [-half_m,  half_m, 0]
                        ], dtype=np.float32)
                        success, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_mat, np.zeros(4))
                        if success:
                            rot_matrix, _ = cv2.Rodrigues(rvec)
                            camera_to_marker_tf = np.eye(4, dtype=np.float32)
                            camera_to_marker_tf[0:3, 0:3] = rot_matrix
                            camera_to_marker_tf[0:3, 3] = tvec.flatten()
                            
                            group_id = "cube"
                            if 10 <= marker_id <= 14: group_id = "cube_left"
                            elif 30 <= marker_id <= 34: group_id = "cube_right"
                                
                            current_frame_cube_info[marker_id] = {
                                'group_id': group_id,
                                'T_cam_to_M': camera_to_marker_tf,
                                'corners': c
                            }
                
                cube_marker_ids = list(current_frame_cube_info.keys())
                valid_ids = set()
                # 마커가 2개이상 인식되면 다중마커에 의한 보정기능 실행
                if len(cube_marker_ids) >= 2:
                    good_connections = {m: 0 for m in cube_marker_ids}
                    for i in range(len(cube_marker_ids)):
                        for j in range(i+1, len(cube_marker_ids)):
                            id1, id2 = cube_marker_ids[i], cube_marker_ids[j]
                            info1, info2 = current_frame_cube_info[id1], current_frame_cube_info[id2]
                            
                            # 인식된 마커들 중 valid한 마커만 사용
                            n1 = info1['T_cam_to_M'][0:3, 2]
                            n2 = info2['T_cam_to_M'][0:3, 2]
                            n1 = n1 / np.linalg.norm(n1)
                            n2 = n2 / np.linalg.norm(n2)
                            dot_val = np.dot(n1, n2)
                            angle_deg = math.degrees(math.acos(np.clip(dot_val, -1.0, 1.0)))
                            angle_error = abs(angle_deg - 90.0)
                            
                            center1 = info1['T_cam_to_M'][0:3, 3]
                            center2 = info2['T_cam_to_M'][0:3, 3]
                            actual_dist = np.linalg.norm(center1 - center2)
                            
                            CUBE_TO_MARKER_1 = self.cube_markers[info1['group_id']].get_transform(id1)
                            CUBE_TO_MARKER_2 = self.cube_markers[info2['group_id']].get_transform(id2)
                            expected_dist = np.linalg.norm(CUBE_TO_MARKER_1[0:3, 3] - CUBE_TO_MARKER_2[0:3, 3])
                            dist_error = abs(actual_dist - expected_dist)
                            
                            if angle_error <= 1.8 and dist_error <= 1.2:
                                good_connections[id1] += 1
                                good_connections[id2] += 1
                    valid_ids = {m for m, count in good_connections.items() if count > 0}
                elif len(cube_marker_ids) == 1:
                    valid_ids.add(cube_marker_ids[0])
                    
                valid_group_ids = set()
                obj_pts_by_group = {}
                img_pts_by_group = {}
                
                for m_id in valid_ids:
                    info = current_frame_cube_info[m_id]
                    group_id = info['group_id']
                    valid_group_ids.add(group_id)
                    if group_id not in obj_pts_by_group:
                        obj_pts_by_group[group_id] = []
                        img_pts_by_group[group_id] = []
                        
                    CUBE_TO_MARKER = self.cube_markers[group_id].get_transform(m_id)
                    local_corners = np.array([
                        [-half_m, -half_m, 0, 1], [ half_m, -half_m, 0, 1],
                        [ half_m,  half_m, 0, 1], [-half_m,  half_m, 0, 1]
                    ], dtype=np.float32)
                    
                    for pts in local_corners:
                        obj_pts_by_group[group_id].append((CUBE_TO_MARKER @ pts)[0:3])
                    for pt_2d in info['corners']:
                        img_pts_by_group[group_id].append(pt_2d)
                        
                for group_id in valid_group_ids:
                    obj_pts_all = np.array(obj_pts_by_group[group_id], dtype=np.float32)
                    img_pts_all = np.array(img_pts_by_group[group_id], dtype=np.float32)
                    if len(obj_pts_all) >= 4:
                        flags = cv2.SOLVEPNP_EPNP if len(obj_pts_all) >= 8 else cv2.SOLVEPNP_IPPE_SQUARE
                        success_m, rvec_m, tvec_m = cv2.solvePnP(obj_pts_all, img_pts_all, cam_mat, np.zeros(4), flags=flags)
                        if success_m:
                            success_m, rvec_m, tvec_m = cv2.solvePnP(obj_pts_all, img_pts_all, cam_mat, np.zeros(4), rvec_m, tvec_m, useExtrinsicGuess=True)
                            rot_matrix_m, _ = cv2.Rodrigues(rvec_m)
                            center_pos_m = tvec_m.flatten().tolist()
                            
                            transform_m = [
                                rot_matrix_m[0][0], rot_matrix_m[0][1], rot_matrix_m[0][2], center_pos_m[0],
                                rot_matrix_m[1][0], rot_matrix_m[1][1], rot_matrix_m[1][2], center_pos_m[1],
                                rot_matrix_m[2][0], rot_matrix_m[2][1], rot_matrix_m[2][2], center_pos_m[2],
                                0.0, 0.0, 0.0, 1.0
                            ]
                            marker_centers_result.append((group_id, transform_m))
                            if tcpip_send:
                                self.tcp_client.send_pose(transform_m)
                return marker_centers_result

            # NON-CUBE logic (e.g. Plate)
            obj_pts = np.array([
                [-half_m, -half_m, 0], [ half_m, -half_m, 0],
                [ half_m,  half_m, 0], [-half_m,  half_m, 0]
            ], dtype=np.float32)
            
            for i in range(len(ids)):
                marker_id = ids_flat[i]
                if marker_id in duplicate_ids:
                    continue
                c = corners[i][0]
                
                success, rvec, tvec = cv2.solvePnP(obj_pts, c, cam_mat, np.zeros(4))
                if not success:
                    continue
                rot_matrix, _ = cv2.Rodrigues(rvec)
                center_pos = tvec.flatten().tolist()
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                # Plate group identification
                if marker_id in getattr(self, 'plate_left_ids', []):
                    marker_centers_result.append(("plate_left", transform))
                elif marker_id in getattr(self, 'plate_right_ids', []):
                    marker_centers_result.append(("plate_right", transform))
                else:
                    marker_centers_result.append((marker_id, transform))
                if tcpip_send:
                    self.tcp_client.send_pose(transform)
                
        return marker_centers_result

class Marker_Transform:
    def __init__(self, serial_number = None, monitoring = False):
        # Initialize
        self.camera = RealSenseCamera(serial_number=serial_number)
        self.marker_detection = Marker_Detection()
        
        # Load configs globally in the wrapper class
        self._load_all_configs()
        
        # Setup Transforms
        tf_vec_l = self.camera_config.get("Tf_to_marker_left", self.camera_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        tf_vec_r = self.camera_config.get("Tf_to_marker_right", self.camera_config.get("Tf_to_marker", [0.022, 0.0, 0.18, 180.0, 0.0, -90.0]))
        t5_vec = self.camera_config.get("T5_to_cam", [0.009, -0.09, -0.085, 159.0, 0.0, 180.0])
        
        self.Tf_to_marker_tf_left = self.make_transform(tf_vec_l)
        self.Tf_to_marker_tf_right = self.make_transform(tf_vec_r)
        self.T5_to_cam_tf = self.make_transform(t5_vec)
        
        self.width = self.camera_config.get("width", 1280)
        self.height = self.camera_config.get("height", 720)
        self.fps = self.camera_config.get("fps", 30)
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        if monitoring and not imshow_when_detect:
            self.camera.monitoring(Flag=True)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

        depth_resolution = self.camera.get_depth_resolution()
        self.marker_detection.set_depth_resolution(depth_resolution)
        
        self.temp_history = []

    def _load_all_configs(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        setting_config_path = os.path.join(base_dir, "config", "setting.yaml")
        try:
            with open(setting_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                self.camera_config = config_data.get("camera", {})
                self.markers_config = config_data.get("marker", config_data) # fallback
                self.marker_detection.markers_config = self.markers_config
                self.marker_detection.thermal_config = config_data.get("thermal_compensation", {})
                print(f"- Loaded Setting Config from {os.path.basename(setting_config_path)}")
        except Exception as e:
            print(f"- Warning: Could not load {setting_config_path}: {e}")
            self.camera_config = {}
            self.markers_config = {}
        cube_size = self.markers_config.get("cube", {}).get("cube_size_mm", 60.0)
        marker_size = cube_size * 0.8
        
        self.marker_detection.cube_markers = {}
        
        calib_left_path = os.path.join(base_dir, "config", "calibrated_cube_left.yaml")
        if os.path.exists(calib_left_path):
            with open(calib_left_path, 'r') as f:
                calib_data = yaml.safe_load(f)
                left_cube = CubeMarker(cube_size_mm=cube_size, side="left")
                left_cube.load_calibration_data(calib_data, marker_size_mm=marker_size)
                self.marker_detection.cube_markers["cube_left"] = left_cube
                self.cube_marker = left_cube # fallback
                print(f"- Loaded Calibration from {os.path.basename(calib_left_path)}")
                
        calib_right_path = os.path.join(base_dir, "config", "calibrated_cube_right.yaml")
        if os.path.exists(calib_right_path):
            with open(calib_right_path, 'r') as f:
                calib_data = yaml.safe_load(f)
                right_cube = CubeMarker(cube_size_mm=cube_size, side="right")
                right_cube.load_calibration_data(calib_data, marker_size_mm=marker_size)
                self.marker_detection.cube_markers["cube_right"] = right_cube
                print(f"- Loaded Phase 1 Calibration from {os.path.basename(calib_right_path)}")
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

        # print(m)
        
        return m

    def calc_cam_to_tool(self, camera_to_marker_tf, side="left"):
        try:
            target_tf = self.Tf_to_marker_tf_left if side == "left" else self.Tf_to_marker_tf_right
            tf_to_marker_inv = np.linalg.inv(target_tf)
            # base_to_tool = base_to_marker * camera_to_marker^-1 * camera_to_tool
            cam_to_tool_tf = camera_to_marker_tf @ tf_to_marker_inv
            cam_to_tool_vec = cam_to_tool_tf.flatten()
            
            # Unit conversion if needed (mm -> m logic from original code)
            if abs(cam_to_tool_vec[3]) > 4 or abs(cam_to_tool_vec[7]) > 4 or abs(cam_to_tool_vec[11]) > 4:
                cam_to_tool_vec[3] /= 1000
                cam_to_tool_vec[7] /= 1000
                cam_to_tool_vec[11] /= 1000
                
            return cam_to_tool_vec
        except np.linalg.LinAlgError:
            print("Singular matrix, cannot invert")
            return None

    def get_marker_transform(self, sampling_time=0, side="left"):
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
                if color_img is None or depth_img is None:
                    time.sleep(0.01)
                    if sampling_time == 0 : return None
                    continue
                
                raw_temp = self.camera.get_camera_temperature() if hasattr(self.camera, 'get_camera_temperature') else None
                smoothed_temp = None
                # 온도에 대한 지수이동평균필터 적용
                if raw_temp is not None:
                    self.temp_history.append(raw_temp)
                    if len(self.temp_history) > 7:
                        self.temp_history.pop(0)
                        
                    if len(self.temp_history) < 7:
                        smoothed_temp = sum(self.temp_history) / len(self.temp_history)
                    else:
                        sorted_hist = sorted(self.temp_history)
                        smoothed_temp = sum(sorted_hist[2:5]) / 3.0
                
                if smoothed_temp is not None:
                    sampled_temps.append(smoothed_temp)
                        
                marker_transforms = self.marker_detection.detect(color_img, depth_img, lpf=lpf, current_temp=smoothed_temp)
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
            
            # CPU 점유율을 낮추기 위한 미세한 대기
            time.sleep(0.01)
            
        if sampled_temps:
            self.current_smoothed_temp = sum(sampled_temps) / len(sampled_temps)
        else:
            self.current_smoothed_temp = None
            
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
                
                # Determine side for transform selection
                calc_side = "left" if "left" in str(marker_id) else "right"
                
                T5_to_tool_vec = self.calc_cam_to_tool(avg_cam_to_marker_tf, side=calc_side)
                if T5_to_tool_vec is not None:
                    final_results[marker_id] = T5_to_tool_vec
        elif sampling_time == 0:
                # For sampling_time == 0, there is only one frame of transforms
                # and thus tfs should have only length 1
                camera_to_marker_tf = np.array(tfs[-1], dtype=np.float32).reshape(4, 4)
                
                calc_side = "left" if "left" in str(marker_id) else "right"
                T5_to_tool_vec = self.calc_cam_to_tool(camera_to_marker_tf, side=calc_side)
                if T5_to_tool_vec is not None:
                    final_results[marker_id] = T5_to_tool_vec
        
        if len(final_results) > 0:
            if self.marker_detection.marker_type == "cube":
                if side == "left":
                    res = final_results.get("cube_left")
                    return [res] if res is not None else None
                elif side == "right":
                    res = final_results.get("cube_right")
                    return [res] if res is not None else None
                elif side == "all":
                    out = []
                    res_r = final_results.get("cube_right")
                    res_l = final_results.get("cube_left")
                    if res_r is not None: out.append(res_r)
                    if res_l is not None: out.append(res_l)
                    return out if out else None
            elif self.marker_detection.marker_type == "plate":
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