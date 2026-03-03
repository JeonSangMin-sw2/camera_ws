import rby1_sdk as rby
import numpy as np
import time
import pyrealsense2 as rs
import cv2
import socket
import struct
import math
import threading
import sys
import select
import json

recorded_data = []   # (q, marker) 같이 저장

def check_key_press():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip()
    return None

# ===============================
# Camere Function
# ===============================


# 카메라 클래스
"""
해당 클래스는 realsense 카메라만 호환
다른 카메라를 사용할 경우 아래의 get이란 이름이 들어간 함수들의 양식을 동일하게 맞춰주고 구현하는것을 권장
"""
class RealSenseCamera:
    # serial_number : 해당 넘버의 카메라를 사용, 입력하지 않으면 첫번째 카메라 사용
    # Stereo : 적외선 카메라를 불러와 스테레오 비전으로 depth map을 직접 계산할지 선택
    """카메라의 시리얼 넘버는 realsense_check.py를 통해 검색할 수 있음"""
    def __init__(self, serial_number=None, Stereo=False):
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

        #적외선카메라 사용여부확인
        self.Infrared = Stereo                                  # 적외선 카메라 사용여부

        # 상태 flag
        self.camera_running = False                             # 카메라 구동상태 flag
        self.camera_monitoring = False                          # 카메라 모니터링 상태 flag

        #이미지 저장 변수
        self.color_image = None
        self.depth_image = None
        self.left_ir_image = None
        self.right_ir_image = None

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
            if self.Infrared:
                print("Infrared camera is used")
                self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
                self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
            else:
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # 파이프라인 시작
            self.profile = self.pipeline.start(self.config)

            # 구동 후 카메라 안정화를 위해 10프레임 정도는 버리고 사용
            for i in range(10):
                self.pipeline.wait_for_frames()
            
            # depth 카메라 내부 파라미터 얻기 : baseline, fx, fy, principal_point 를 위해 사용
            depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_stream.get_intrinsics()

            if self.Infrared:
                left_ir_stream = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
                right_ir_stream = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
                self.intrinsics = left_ir_stream.get_intrinsics()
                
                extrinsics = left_ir_stream.get_extrinsics_to(right_ir_stream)
                self.baseline = abs(extrinsics.translation[0]) # m
                print("Baseline: ", self.baseline)
                self.fx = self.intrinsics.fx
                self.fy = self.intrinsics.fy
                self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy] #pixel

            else:
                color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.intrinsics = color_stream.get_intrinsics()
                self.fx = self.depth_intrinsics.fx
                self.fy = self.depth_intrinsics.fy
                self.principal_point = [self.depth_intrinsics.ppx, self.depth_intrinsics.ppy] #pixel

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
            
            if self.Infrared:
                left_ir_frame = frames.get_infrared_frame(1)
                right_ir_frame = frames.get_infrared_frame(2)
                depth_frame = frames.get_depth_frame()
                
                if not left_ir_frame or not right_ir_frame or not depth_frame:
                    print("no frame")
                    return
                
                depth_frame = self.spatial.process(depth_frame)
                depth_frame = self.temporal.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
                
                left_ir_data = np.asanyarray(left_ir_frame.get_data())
                right_ir_data = np.asanyarray(right_ir_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                
                with self.lock:
                    self.left_ir_image = left_ir_data
                    self.right_ir_image = right_ir_data
                    self.depth_image = depth_data
            else:
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

    def get_left_ir_image(self):
        with self.lock:
            if self.left_ir_image is None:
                return None
            return self.left_ir_image.copy()

    def get_right_ir_image(self):
        with self.lock:
            if self.right_ir_image is None:
                return None
            return self.right_ir_image.copy()

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

    # 연산에 필요한 카메라 파라미터 설정
    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.fx = param[2]
        self.fy = param[3]

    def set_depth_resolution(self, depth_resolution):
        self.depth_resolution = depth_resolution

    def set_baseline(self, baseline):
        self.baseline = baseline

    # 기본 연산함수
    def normalize(self, v):
        m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if m > 1e-9:
            return [v[0]/m, v[1]/m, v[2]/m]
        return v

    def cross(self, a, b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
    
    def convert_pixel2mm(self, center):
        # center: [x, y, z] (z is depth value in mm)
        if not center:
            return center
        
        if self.fx == 0 or self.fy == 0:
            return center 

        z = center[2] * self.depth_resolution
        x = (center[0] - self.principal_point[0]) * z / self.fx
        y = (center[1] - self.principal_point[1]) * z / self.fy
        
        return [x, y, z]
    
    # 스테레오 이미지를 통한 계산은 인자가 다르므로 별도로 구현
    def stereo_cal_corners_3d_mm(self, main_corner, ref_corner):
        corners_3d_mm = []
        for i in range(4):
            # 카메라는 Left가 기준이 되고, Right영상에서 매칭점을 찾습니다.
            # D405 등 카메라 모델에 따라 Left/Right의 물리적 위치가 다를 수 있지만
            # 보통 disparity는 절대값으로 계산하거나 올바른 방향으로 빼주어야 양수 깊이가 나옵니다.-----------------------------------------------
            disparity = abs(main_corner[i][0] - ref_corner[i][0])
            depth = (self.baseline * self.fx) / disparity * 1000 # m -> mm
            x_mm = (main_corner[i][0] - self.principal_point[0]) * depth / self.fx
            y_mm = (main_corner[i][1] - self.principal_point[1]) * depth / self.fy
            corners_3d_mm.append([x_mm, y_mm, depth])
        return corners_3d_mm
    
    # 각 좌표의 법선벡터를 활용하여 회전행렬 계산
    def get_rotation_matrix(self, corners_3d):
        # 정확한 회전을 계산하기 위해 정사각형으로 보정
        pts_3d = self.validate_and_correct_marker_shape(corners_3d)
        """
        corners order in aruco: TL, TR, BR, BL (0, 1, 2, 3)
        x_axis = (tr + br) - (tl + bl)
        y_axis = (bl + br) - (tl + tr)
        """
        x_axis = [
            (pts_3d[1][0] + pts_3d[2][0]) - (pts_3d[0][0] + pts_3d[3][0]),
            (pts_3d[1][1] + pts_3d[2][1]) - (pts_3d[0][1] + pts_3d[3][1]),
            (pts_3d[1][2] + pts_3d[2][2]) - (pts_3d[0][2] + pts_3d[3][2])
        ]
        
        y_axis = [
            (pts_3d[3][0] + pts_3d[2][0]) - (pts_3d[0][0] + pts_3d[1][0]),
            (pts_3d[3][1] + pts_3d[2][1]) - (pts_3d[0][1] + pts_3d[1][1]),
            (pts_3d[3][2] + pts_3d[2][2]) - (pts_3d[0][2] + pts_3d[1][2])
        ]
        
        x_axis = self.normalize(x_axis)
        y_axis = self.normalize(y_axis)
        
        z_axis = self.cross(x_axis, y_axis)
        z_axis = self.normalize(z_axis)
        
        # y_axis 재계산 : z_axis와 x_axis에 수직이 되도록
        y_axis = self.cross(z_axis, x_axis)
        y_axis = self.normalize(y_axis)
        
        # Rotation Matrix (3x3)
        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = x_axis[0]; R[0][1] = y_axis[0]; R[0][2] = z_axis[0]
        R[1][0] = x_axis[1]; R[1][1] = y_axis[1]; R[1][2] = z_axis[1]
        R[2][0] = x_axis[2]; R[2][1] = y_axis[2]; R[2][2] = z_axis[2]
        
        return R

    # 회전행렬을 오일러각으로 변환(디버깅용)
    def get_rpy_from_matrix(self, R):
        sy = math.sqrt(R[0][0]**2 + R[1][0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2][1], R[2][2])
            pitch = math.atan2(-R[2][0], sy)
            yaw = math.atan2(R[1][0], R[0][0])
        else:
            roll = math.atan2(-R[1][2], R[1][1])
            pitch = math.atan2(-R[2][0], sy)
            yaw = 0
        return [roll, pitch, yaw]

    def apply_point_lpf(self, marker_id, pts_3d, spike_threshold=1.0):
        pts_arr = np.array(pts_3d, dtype=np.float32)
        if marker_id not in self.prev_pts_dict:
            # print("First detection")
            self.prev_pts_dict[marker_id] = pts_arr
            return pts_3d
        prev_pts = self.prev_pts_dict[marker_id]
        
        distances = np.linalg.norm(pts_arr - prev_pts, axis=1)
        if np.any(distances > spike_threshold):
            # 너무 크게 튀는 값이 있다면 현재 입력값을 무시하고 이전 값을 반환
            #print("Spike detected")
            return prev_pts.tolist()
            
        new_pts = self.lpf_alpha * pts_arr + (1.0 - self.lpf_alpha) * prev_pts
        self.prev_pts_dict[marker_id] = new_pts
        
        return new_pts.tolist()
    
    def validate_and_correct_marker_shape(self, pts_3d, length_tol=0.2, angle_tol_deg=2.0):
        """
        추출된 마커의 3D 모서리 좌표들이 정사각형에 가까운지 확인 (직각, 길이 동일 여부)
        유사하지 않다고 판단될 경우 오차를 보정한 완벽한 정사각형 좌표를 반환
        """
        pts = np.array(pts_3d)
        if len(pts) != 4:
            return pts_3d
            
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[1]
        v2 = pts[3] - pts[2]
        v3 = pts[0] - pts[3]
        
        edges = [v0, v1, v2, v3]
        lengths = [np.linalg.norm(v) for v in edges]
        if any(l == 0 for l in lengths):
            return pts_3d
            
        mean_len = np.mean(lengths)
        
        # 1. 연결한 직선들의 길이가 같은지 확인
        length_diffs = [abs(l - mean_len) / mean_len for l in lengths]
        is_length_valid = all(d <= length_tol for d in length_diffs)
        
        # 2. 연결한 직선들이 각각 직각인지 확인
        angles_valid = True
        for i in range(4):
            e1 = edges[i]
            e2 = edges[(i+1)%4]
            # e1 방향에서 돌아가는 각도의 내적
            cos_theta = np.dot(-e1, e2) / (lengths[i] * lengths[(i+1)%4])
            angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
            if abs(angle_deg - 90.0) > angle_tol_deg:
                angles_valid = False
                break
                
        is_valid = is_length_valid and angles_valid
        
        if not is_valid:
            # 보정 기능: 직각과 대칭을 갖는 평면상 이상적인 정사각형 코너로 보정
            center = np.mean(pts, axis=0)
            
            x_vec = (v0 - v2) / 2.0
            x_axis = x_vec / np.linalg.norm(x_vec)
            
            y_vec = (v1 - v3) / 2.0
            
            z_axis = np.cross(x_axis, y_vec)
            z_norm = np.linalg.norm(z_axis)
            if z_norm < 1e-6:
                return pts_3d
                
            z_axis = z_axis / z_norm
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            half_l = mean_len / 2.0
            corrected_pts = [
                center - half_l * x_axis - half_l * y_axis, # TL
                center + half_l * x_axis - half_l * y_axis, # TR
                center + half_l * x_axis + half_l * y_axis, # BR
                center - half_l * x_axis + half_l * y_axis  # BL
            ]            
            return [list(p) for p in corrected_pts]
            
        return pts_3d

    def get_marker_plane_equation(self, corners, depth_img):
        # corners: list of [x, y]
        # 바운더리 추출
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # ROI를 10% 축소하여 테두리에 대한 노이즈 방지
        w = max_x - min_x
        h = max_y - min_y
        
        start_x = max(0, int(min_x + w * 0.1))
        end_x = min(depth_img.shape[1], int(max_x - w * 0.1))
        start_y = max(0, int(min_y + h * 0.1))
        end_y = min(depth_img.shape[0], int(max_y - h * 0.1))
        
        if end_x <= start_x or end_y <= start_y:
            return None

        # ROI 내의 depth 값 추출
        # (평면 방정식은 Raw Depth 단위로 추출. 추후에 depth_resolution을 통해 결과값 출력 예정)
        roi_depth = depth_img[start_y:end_y, start_x:end_x]
        
        # 좌표 그리드 생성
        # Note: indices returns (y_grid, x_grid)
        y_grid, x_grid = np.indices(roi_depth.shape)
        y_grid += start_y
        x_grid += start_x
        
        # Flatten everything
        z_flat = roi_depth.flatten()
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        
        # Filter invalid depth (0)
        valid_mask = z_flat > 0
        
        if np.sum(valid_mask) < 50: # Robust fit를 위해 최소 50개의 점 필요
            return None
            
        z_valid = z_flat[valid_mask]
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        
        """
        Robust Plane Fitting: 1/Z vs (u, v) normalized
        A plane in 3D (AX + BY + CZ + D = 0) corresponds to 
        1/Z = -(A/D)*X/Z - (B/D)*Y/Z - (C/D)
        1/Z = a * u_norm + b * v_norm + c
        where u_norm = (u - cx)/fx, v_norm = (v - cy)/fy
        """
        inv_z = 1.0 / z_valid
        
        u_norm = (x_valid - self.principal_point[0]) / self.fx
        v_norm = (y_valid - self.principal_point[1]) / self.fy
        
        # A matrix: [u_norm, v_norm, 1]
        A = np.column_stack((u_norm, v_norm, np.ones_like(u_norm)))
        
        # Solve Ax = B where B is 1/z
        try:
            X, residuals, rank, s = np.linalg.lstsq(A, inv_z, rcond=None)
            
            # Calculate RMSE (Root Mean Squared Error)
            # residuals returns sum of squared residuals
            rmse = 0
            if residuals.size > 0:
                rmse = np.sqrt(residuals[0] / len(z_valid))
            
            return X # [a, b, c]
        except np.linalg.LinAlgError:
            return None

    # 마커들의 중심좌표(4*4행렬)
    def detect(self, color_image, depth_image, lpf = False, logging = False):
        depth_filtered = depth_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        marker_centers_result = []
        
        if ids is not None and len(ids) > 0:
            
            for i in range(len(ids)):
                c = corners[i][0] # 마커의 모서리

                # 마커의 중심점 계산
                xs = [pt[0] for pt in c]
                ys = [pt[1] for pt in c]
                xs.sort()
                ys.sort()
                x_center = (xs[0] + xs[3]) / 2.0
                y_center = (ys[0] + ys[3]) / 2.0
                
                # 평면 방정식 계산 : depth 노이즈를 줄이기 위함
                plane = self.get_marker_plane_equation(c, depth_filtered)
                
                # 만약 노이즈가 너무 심해 평면 방정식을 구하지 못했다면 신뢰할 수 없는 데이터이므로 무시
                if plane is None:
                    continue
                    
                A, B, C_val = plane
                
                # 모서리점들의 z값 보정
                c_list = []
                for pt in c:
                    u_n = (pt[0] - self.principal_point[0]) / self.fx
                    v_n = (pt[1] - self.principal_point[1]) / self.fy
                    inv_z_est = A*u_n + B*v_n + C_val
                    z_est = 1.0 / inv_z_est if abs(inv_z_est) > 1e-6 else 0
                    
                    # 미리 mm 단위 3D 좌표로 변환해서 저장
                    pt_3d_mm = self.convert_pixel2mm([pt[0], pt[1], float(z_est)])
                    c_list.append(pt_3d_mm)

                if logging:
                    string = f"{c[0][0]},{c[0][1]},{c[1][0]},{c[1][1]},{c[2][0]},{c[2][1]},{c[3][0]},{c[3][1]}"
                    self.logger.save(string)

                # 보정된 코너점들에 Point LPF 적용 (마커 ID 별로 추적)
                marker_id = ids[i][0]
                if lpf:
                    c_list_filtered = self.apply_point_lpf(marker_id, c_list)
                else:
                    c_list_filtered = c_list

                # 필터링된 꼭짓점 4개의 중심 좌표 재계산
                center_pos = np.mean(c_list_filtered, axis=0).tolist()
                
                # 필터링된 코너점 기반 회전 행렬 계산
                rot_matrix = self.get_rotation_matrix(c_list_filtered)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                marker_centers_result.append(transform)
                
        return marker_centers_result

    def detect_stereo(self, main_img, ref_img, lpf = False):
        main_corners, main_ids, main_rejected = cv2.aruco.detectMarkers(main_img, self.dictionary, parameters=self.parameters)
        ref_corners, ref_ids, ref_rejected = cv2.aruco.detectMarkers(ref_img, self.dictionary, parameters=self.parameters)
        marker_centers_result = []
        
        if main_ids is not None and ref_ids is not None:
            # 마커 순서가 헷갈리지 않도록 ref_dict에 저장
            ref_dict = {}
            for i, marker_id in enumerate(ref_ids):
                mid = marker_id[0]
                if mid not in ref_dict:
                    ref_dict[mid] = []
                ref_dict[mid].append(ref_corners[i][0])

            for i, marker_id in enumerate(main_ids):
                mid = marker_id[0]
                if mid not in ref_dict:
                    continue

                main_corner = main_corners[i][0]
                candidates = ref_dict[mid]
                
               
                best_idx = -1
                min_y_diff = float('inf')
                
                main_center_y = sum([pt[1] for pt in main_corner]) / 4.0
                
                for idx, ref_c in enumerate(candidates):
                    ref_center_y = sum([pt[1] for pt in ref_c]) / 4.0
                    y_diff = abs(main_center_y - ref_center_y)
                    
                    # 수직으로 50픽셀 이내면 같은 마커로 인식. 인식거리에 따라 threshold는 추후 보정 필요
                    if y_diff < 50: # pixel threshold
                        if y_diff < min_y_diff:
                            min_y_diff = y_diff
                            best_idx = idx
                
                if best_idx == -1:
                    continue
                ref_corner = candidates.pop(best_idx) 
                if len(candidates) == 0:
                    del ref_dict[mid]

                # Get depth
                corners_3d_mm = self.stereo_cal_corners_3d_mm(main_corner, ref_corner)

                if len(corners_3d_mm) != 4:
                    continue
                
                # 코너점들의 형태를 정사각형으로 보정
                c_list_corrected = self.validate_and_correct_marker_shape(corners_3d_mm)

                # 보정된 코너점들에 Point LPF 적용 (마커 ID 별로 추적)
                if lpf:
                    c_list_filtered = self.apply_point_lpf(mid, c_list_corrected)
                else:
                    c_list_filtered = c_list_corrected

                # 필터링된 꼭짓점 4개의 중심 좌표 재계산 (테스트코드와 동일하게 np.mean 사용)
                center_pos = np.mean(c_list_filtered, axis=0).tolist()
                
                # 필터링된 코너점 기반 회전 행렬 계산
                rot_matrix = self.get_rotation_matrix(c_list_filtered)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]

                # print(f"Center [{transform[3]}, {transform[7]}, {transform[11]}]")
                # print(f"rpy    [{self.rpy[0]*180/math.pi}, {self.rpy[1]*180/math.pi}, {self.rpy[2]*180/math.pi}]")
                
                marker_centers_result.append(transform)
                
        return marker_centers_result
        

    




class Marker_Transform:
    def __init__(self, Stereo=False, tool_to_cam = [0,0,0,0,0,0], serial_number = None, monitoring = False):
        self.Stereo = Stereo
        # self.client = TCPClient("127.0.0.1", 5000)
        # Setup Transforms
        # T5_to_marker_data = [0.022, 0.0, 0.18, 180, 0.0, -90.0]
        T5_to_marker_data = [0,0,0,0,0,0]
        # tool_to_cam = [0,0,0,0,0,0]
        #tool_to_cam = [0.009,-0.09,-0.085,144,0,180]
        self.T5_to_marker_tf = self.make_transform(T5_to_marker_data)
        self.tool_to_cam_tf = self.make_transform(tool_to_cam)
        
        # Initialize
        self.camera = RealSenseCamera(serial_number=serial_number, Stereo=Stereo)
        self.marker_detection = Marker_Detection()
        
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        if monitoring:
            self.camera.monitoring(Flag=True)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

        depth_resolution = self.camera.get_depth_resolution()
        self.marker_detection.set_depth_resolution(depth_resolution)
        if self.Stereo:
            baseline = self.camera.get_baseline()
            self.marker_detection.set_baseline(baseline)

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

    def get_marker_transform(self, sampling_time=0, lpf = False):
        T5_to_tool_vec = None
        # Collection list for sampling
        collected_transforms = []
        start_time = time.time()

        if lpf and sampling_time > 0:
            self.marker_detection.prev_pts_dict = {}

        while True:
            # Check timeout if sampling
            if sampling_time > 0 and (time.time() - start_time > sampling_time):
                break
                
            try:
                if not self.camera.camera_monitoring:
                    self.camera.capture_image()
                if self.Stereo:
                    left_ir_img = self.camera.get_left_ir_image()
                    right_ir_img = self.camera.get_right_ir_image()
                    if left_ir_img is None or right_ir_img is None:
                        time.sleep(0.01)
                        if sampling_time == 0 : return None
                        continue
                    marker_transforms = self.marker_detection.detect_stereo(left_ir_img, right_ir_img, lpf = lpf)
                else:
                    color_img = self.camera.get_color_image()
                    depth_img = self.camera.get_depth_image()
                    if color_img is None or depth_img is None:
                        time.sleep(0.01)
                        if sampling_time == 0 : return None
                        continue
                    marker_transforms = self.marker_detection.detect(color_img, depth_img, lpf = lpf)
                
                for tf_list in marker_transforms:
                    # Convert flattened list to 4x4 matrix
                    camera_to_marker_tf = np.array(tf_list, dtype=np.float32).reshape(4, 4)
                    
                    if sampling_time == 0:
                        try:
                            camera_to_marker_inv = np.linalg.inv(camera_to_marker_tf)
                            tool_to_cam_inv = np.linalg.inv(self.tool_to_cam_tf)
                            # base_to_tool = base_to_marker * camera_to_marker^-1 * camera_to_tool
                            T5_to_tool_tf = self.T5_to_marker_tf @ camera_to_marker_inv @ tool_to_cam_inv
                            T5_to_tool_vec = T5_to_tool_tf.flatten()
                            
                            # Unit conversion if needed (mm -> m logic from original code)
                            if abs(T5_to_tool_vec[3]) > 4 or abs(T5_to_tool_vec[7]) > 4 or abs(T5_to_tool_vec[11]) > 4:
                                T5_to_tool_vec[3] = T5_to_tool_vec[3]/1000
                                T5_to_tool_vec[7] = T5_to_tool_vec[7]/1000
                                T5_to_tool_vec[11] = T5_to_tool_vec[11]/1000
                                
                            # self.client.send_pose(T5_to_tool_vec)

                            return T5_to_tool_vec
                        except np.linalg.LinAlgError:
                            print("Singular matrix, cannot invert")
                            continue
                    else:
                        # Append the FORWARD transform, NOT the inverted one
                        collected_transforms.append(tf_list)
                        
            except KeyboardInterrupt:
                raise
            
            # CPU 점유율을 낮추기 위한 미세한 대기
            time.sleep(0.01)
            
            # If not sampling, break after one attempt (handled by return above)
            # If sampling, continue loop
            if sampling_time == 0: 
                break

             
        # Post-processing for sampling
        if sampling_time > 0:
            if not collected_transforms:
                return None
            
            data = np.array(collected_transforms) # Shape (N, 16)
            
            # Separate translation and rotation for CAMERA_TO_MARKER (NOT inverted yet)
            translations = data[:, [3, 7, 11]]
            
            # Median for translation is robust
            final_translation = np.median(translations, axis=0)
            
            # Average rotations using SVD (chordal L2 mean) to maintain orthogonality
            # Extract 3x3 rotation matrices
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
            
            # Reconstruct the averaged 4x4 camera_to_marker transform
            avg_cam_to_marker_tf = np.eye(4, dtype=np.float32)
            avg_cam_to_marker_tf[0:3, 0:3] = final_R
            avg_cam_to_marker_tf[0:3, 3] = final_translation
            
            # NOW compute the inversions and multiplication with stable values
            try:
                camera_to_marker_inv = np.linalg.inv(avg_cam_to_marker_tf)
                tool_to_cam_inv = np.linalg.inv(self.tool_to_cam_tf)
                T5_to_tool_tf = self.T5_to_marker_tf @ camera_to_marker_inv @ tool_to_cam_inv
                final_vec = T5_to_tool_tf.flatten()
                # self.client.send_pose(final_vec)
                # Unit conversion
                if abs(final_vec[3]) > 4 or abs(final_vec[7]) > 4 or abs(final_vec[11]) > 4:
                    final_vec[3] /= 1000
                    final_vec[7] /= 1000
                    final_vec[11] /= 1000
                    
                return final_vec
            except np.linalg.LinAlgError:
                return None
            
        return None


        
import argparse
import numpy as np
import time
import rby1_sdk as rby


# ============================================================
# Lie algebra utilities (그대로 유지)
# ============================================================

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def se3_exp(xi):
    w = xi[:3]
    v = xi[3:]
    R = so3_exp(w)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]) / theta

        V = (np.eye(3)
             + (1-np.cos(theta))/theta * K
             + (theta-np.sin(theta))/theta * (K@K))

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = V @ v
    return T


def so3_log(R):
    cos_theta = (np.trace(R)-1)/2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T)/(2*np.sin(theta))
    return theta*np.array([
        w_hat[2,1], w_hat[0,2], w_hat[1,0]
    ])


def se3_log(T):
    R = T[:3,:3]
    t = T[:3,3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = t
    else:
        w_hat = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]) / theta

        A = (np.eye(3)
             - 0.5*w_hat
             + (1/theta**2)*(1-theta/(2*np.tan(theta/2)))*(w_hat@w_hat))
        v = A @ t

    return np.hstack([w, v])


# ============================================================
# Robot initialization
# ============================================================


def load_npz_dataset(path):
    data = np.load(path)
    return data["q"], data["marker"]


def create_robot(ip):
    robot = rby.create_robot_a(ip)
    robot.connect()
    robot.power_on(".*")
    robot.servo_on("^(?!.*head).*")
    robot.reset_fault_control_manager()
    robot.enable_control_manager(False)
    return robot


# ============================================================
# Capture dataset (기존 기능 유지)
# ============================================================

def capture_dataset(robot, dyn_model, RIGHT_ARM_IDX, marker_transform):

    q_cmd_list = []
    T_meas_list = []

    print("\nPress 'e' + Enter to capture")
    print("Press 'q' + Enter to quit\n")

    while True:
        key = input()

        if key == 'e':
            state = robot.get_state()
            q_full = state.position.copy()
            q_cmd = q_full[RIGHT_ARM_IDX[:7]].copy()

            result = marker_transform.get_marker_transform(sampling_time=0)

            if result is None:
                print("Marker not detected.")
                continue

            T_meas = np.array(result).reshape(4, 4)

            q_cmd_list.append(q_cmd)
            T_meas_list.append(T_meas)

            print(f"Captured sample {len(q_cmd_list)}")
            print("q =", np.round(q_full[RIGHT_ARM_IDX],3))
            T_print = np.array(result).reshape(4, 4)
            print("marker =", np.round(T_print,3))
        elif key == 'q':
            break

        time.sleep(0.05)

    return np.array(q_cmd_list), np.array(T_meas_list)


# ============================================================
# Gauss Newton (기존 알고리즘 그대로)
# ============================================================


def generate_sim_measurements(robot, dyn_model,
                              q_cmd_list, RIGHT_ARM_IDX,
                              q_nominal, ndof):
    q_offset_true = np.deg2rad([3, -2, 1, 4, -3, 2, 1])
    xi_cam_true = np.array([0.02, -0.01, 0.015, 0.01, 0.02, -0.01])
    
    T_list = []

    for q_cmd in q_cmd_list:
        q_full = q_nominal.copy()
        q_full[RIGHT_ARM_IDX] = q_cmd + q_offset_true

        state = dyn_model.make_state(
            ["link_torso_5", "ee_right"],
            robot.model().robot_joint_names
        )
        state.set_q(q_full)
        dyn_model.compute_forward_kinematics(state)

        T_fk = dyn_model.compute_transformation(state, 0, 1)

        # noise = np.array([-0.100956, 0.024401, 0.009073,
        #                   -0.002027, 0.028192, 0.022449])

        # T_meas = T_fk @ se3_exp(noise)
        if ndof == 13:
            T_meas = T_fk @ se3_exp(xi_cam_true)         
        else :
            T_meas = T_fk          
        T_list.append(T_meas)

    return T_list

def update_optimization(q_cmd_list, T_meas_list):
     # 7자유도 최적화 해 역대입
    q_offset_deg = np.array([-0.3833566 ,  0.15210911, -0.08483475 , 0.2933563 , -2.17410442 , 1.20850996  ,0.42705511])
    q_offset_rad = np.deg2rad(q_offset_deg)
    q_cmd_list = q_cmd_list + q_offset_rad        
    # 6자유도 최적화 해 역대입
    T_noise = se3_exp((np.array([-0.18633638 , 0.00041163 , 0.00745352 , 0.00289723 , 0.00718141 , 0.03564564])))
    T_meas_list = np.array([
        T @ np.linalg.inv(T_noise)
        for T in T_meas_list
    ])
    return q_cmd_list, T_meas_list

def optimize(robot, dyn_model,
             q_cmd_list, T_meas_list,
             RIGHT_ARM_IDX, ndof):

    q_nominal = robot.get_state().position.copy()

    q_offset = np.zeros(7)
    xi_cam = np.zeros(6)

    optimize_all = (ndof == 13)
    optimize_camera = (ndof == 6)
    print("optimize_all ==",optimize_all)
    print("optimize_camera ==",optimize_camera)
    
    max_iter = 500
    eps = 1e-6

    for it in range(max_iter):

        if optimize_all:
            H = np.zeros((13, 13))
            g = np.zeros(13)
        elif optimize_camera:
            H = np.zeros((6, 6))
            g = np.zeros(6)
        else:
            H = np.zeros((7, 7))
            g = np.zeros(7)
            

        total_err = 0

        for q_cmd, T_meas in zip(q_cmd_list, T_meas_list):

            q_full = q_nominal.copy()
            if optimize_camera:
                q_full[RIGHT_ARM_IDX[:7]] = q_cmd 
            else:
                q_full[RIGHT_ARM_IDX[:7]] = q_cmd + q_offset
            
            state = dyn_model.make_state(
                ["link_torso_5", "ee_right"],
                robot.model().robot_joint_names
            )

            state.set_q(q_full)
            dyn_model.compute_forward_kinematics(state)
            dyn_model.compute_diff_forward_kinematics(state)

            T_fk = dyn_model.compute_transformation(state, 0, 1)

            if optimize_all:
                T_model = T_fk @ se3_exp(xi_cam)
            elif optimize_camera:
                T_model = T_fk @ se3_exp(xi_cam)
            else:
                T_model = T_fk

            T_err = np.linalg.inv(T_model) @ T_meas
            xi = se3_log(T_err)

            Jb = dyn_model.compute_body_jacobian(state, 0, 1)

            if optimize_all:
                J = np.zeros((6, 13))
                J[:, :7] = Jb[:, RIGHT_ARM_IDX[:7]]
                J[:, 7:] = np.eye(6)
            elif optimize_camera:
                # J = np.zeros((6, 6))
                # J[:, :7] = Jb[:, RIGHT_ARM_IDX[:7]]
                J = np.eye(6)
            else:
                J = Jb[:, RIGHT_ARM_IDX[:7]]

            H += J.T @ J
            g += J.T @ xi
            total_err += np.linalg.norm(xi)

        dx = np.linalg.pinv(H) @ g


        if optimize_all:    
            q_offset += dx[:7]
            xi_cam += dx[7:]
        elif optimize_camera:
            xi_cam += dx[:6]
        else :
            q_offset += dx[:7]
            
        print(f"[{it}] |dx|={np.linalg.norm(dx):.3e}, |err|={total_err:.3e}")

        if np.linalg.norm(dx) < eps:
            print("Converged.")
            break

    return q_offset, xi_cam


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ndof", type=int, default=7,
                        choices=[6, 7, 13])
    parser.add_argument("--ip", type=str,
                        default="192.168.30.1:50051")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["live","npz", "sim"]
                        )
    parser.add_argument("--path", type=str, default="captured_dataset.npz",
        help="Path to npz dataset (default: captured_dataset.npz)")

    

    args = parser.parse_args()

    marker_transform = None
    
    robot = create_robot(args.ip)
    dyn_model = robot.get_dynamics()
    model = robot.model()
    RIGHT_ARM_IDX = model.right_arm_idx


    if args.mode == "live":
        # marker_transform는 기존 코드 그대로 사용
        marker_transform = Marker_Transform(Stereo=True)
        marker_transform.camera.monitoring(Flag=True)

        q_cmd_list, T_meas_list = capture_dataset(
            robot, dyn_model,
            RIGHT_ARM_IDX,
            marker_transform
        )
        np.savez_compressed(
            "captured_dataset.npz",
            q=q_cmd_list,
            marker=T_meas_list
        )

    elif args.mode == "npz":
        q_cmd_list, T_meas_list = load_npz_dataset(args.path)
        print("size=", np.size(q_cmd_list))

    else :
        q_cmd_list = np.random.uniform(-1, 1, (10, 7))
        q_nominal = robot.get_state().position.copy()
        print("q_nominal", q_nominal)
        T_meas_list = generate_sim_measurements(
            robot, dyn_model,
            q_cmd_list,
            RIGHT_ARM_IDX,
            q_nominal,
            args.ndof
        )
        

    print("Dataset saved.")
    
    
    q_cmd_list, T_meas_list = update_optimization(q_cmd_list, T_meas_list)    
    
    q_offset, xi_cam = optimize(
        robot, dyn_model,
        q_cmd_list,
        T_meas_list,
        RIGHT_ARM_IDX,
        args.ndof
    )

    
    # q_offset, xi_cam = optimize(
    #     robot, dyn_model,
    #     q_cmd_list,
    #     T_meas_list,
    #     RIGHT_ARM_IDX,
    #     args.ndof
    # )

    print("\n===== RESULT =====")
    print("Joint offset (deg):")
    print(np.rad2deg(q_offset))
    print("Camera xi:")
    print(xi_cam)

    # ✅ JSON 저장
    result_dict = {
        "joint_offset_deg": np.rad2deg(q_offset).tolist(),
        "xi_cam": np.array(xi_cam).tolist()
    }

    with open("calibration_result.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    print("Result saved to calibration_result.json")

    if marker_transform is not None:
        marker_transform.camera.monitoring(Flag=True)
    
if __name__ == "__main__":
    main()
