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

recorded_data = []   # (q, marker) 같이 저장

def check_key_press():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip()
    return None

# ===============================
# Camere Function
# ===============================


###
# 해당 클래스는 마커인식을 위한 기능들을 realsense 카메라를 기준으로 정의한 클래스 
# 다른 카메라로 기능을 사용할 경우 나머지 클래스와의 연동을 위해 함수의 양식은 일치시켜주어야함
# 필수 구현함수 : capture_image(), get 이라 들어간 모든 함수

###
# 데이터를 텍스트 파일로 저장하는 클래스
class File_Logger:
    def __init__(self, filepath="log.txt"):
        self.filepath = filepath

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")


class RealSenseCamera:
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
        self.depth_resolution = depth_scale*1000

        #나머지 카메라 구동을 위한 파라미터 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.spatial = rs.spatial_filter()       # 공간적 평활화 (노이즈 제거)
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.temporal = rs.temporal_filter()     # 시간적 평활화 (깜빡임 방지)
        self.hole_filling = rs.hole_filling_filter() # 빈 공간 채우기

        #적외선카메라 사용여부확인
        self.Infrared = Stereo
        self.camera_running = False

        #이미지 저장 변수
        self.color_image = None
        self.depth_image = None
        self.left_ir_image = None
        self.right_ir_image = None

        # 기본해상도
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30

        #카메라 내부 파라미터
        self.fx = 0.0
        self.fy = 0.0
        self.principal_point = [0.0, 0.0]
        self.intrinsics = None
        self.profile = None
        self.baseline = 0.065

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

            

            # 구동 후 카메라 안정화를 위해 10프레임 정도는 무시하고 사용
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
        if Flag:
            self.thread = threading.Thread(target=self.stream_on)
            self.thread.start()
        else:
            self.stream_off()
            if self.thread is not None:
                self.thread.join()

    def stream_on(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
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
                time.sleep(0.01)
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
                # 파이썬 API의 한계로 인해 필터를 align 이후에 적용합니다. 
                # 대신 뒤에서 SE(3) 필터로 노이즈를 제어합니다.
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


class Marker_Detection:
    def __init__(self):

        # self.logger = File_Logger(filepath="marker_pixel.txt")


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
        self.parameters.perspectiveRemovePixelPerCell = 10 # 정밀한 비트 추출

        # 계산에 쓰일 카메라 내부 파라미터
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.depth_resolution = 1
        self.rpy = [0, 0, 0]
        
        self.lpf_alpha = 0.3
        self.prev_pts_dict = {}

    def apply_point_lpf(self, marker_id, pts_3d):
        pts_arr = np.array(pts_3d, dtype=np.float32)
        if marker_id not in self.prev_pts_dict:
            self.prev_pts_dict[marker_id] = pts_arr
            return pts_3d
            
        new_pts = self.lpf_alpha * pts_arr + (1.0 - self.lpf_alpha) * self.prev_pts_dict[marker_id]
        self.prev_pts_dict[marker_id] = new_pts
        
        return new_pts.tolist()

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
    
    def validate_and_correct_marker_shape(self, pts_3d, length_tol=0.2, angle_tol_deg=2.0):
        """
        추출된 마커의 3D 모서리 좌표들이 정사각형에 가까운지 확인 (직각, 길이 동일 여부)
        유사하지 않다고 판단될 경우 오차를 보정한 완벽한 정사각형 좌표를 반환합니다.
        """
        pts = np.array(pts_3d)
        if len(pts) != 4:
            return False, pts_3d
            
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[1]
        v2 = pts[3] - pts[2]
        v3 = pts[0] - pts[3]
        
        edges = [v0, v1, v2, v3]
        lengths = [np.linalg.norm(v) for v in edges]
        if any(l == 0 for l in lengths):
            return False, pts_3d
            
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
                return False, pts_3d
                
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
            return False, [list(p) for p in corrected_pts]
            
        return True, pts_3d

    def get_rotation_matrix(self, corners_3d):
        # corners_3d: list of [x_mm, y_mm, z_mm]
        # 입력으로 이미 3D mm 좌표계를 바로 받도록 변경 (LPF 필터링 이후 좌표 사용 위함)
        pts_3d = corners_3d
        
        # 코너들이 정사각형에 가까운지 검증하고 필요시 보정------------------------------------------------------------------
        # is_valid, pts_3d = self.validate_and_correct_marker_shape(pts_3d)

        # Vector logic from C++
        # x_axis = (tr + br) - (tl + bl) -> (pt1 + pt2) - (pt0 + pt3)
        # y_axis = (bl + br) - (tl + tr) -> (pt3 + pt2) - (pt0 + pt1) -> Y Down
        # corners order in aruco: TL, TR, BR, BL (0, 1, 2, 3)
        
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
        
        # Recompute Y to be orthogonal
        y_axis = self.cross(z_axis, x_axis)
        y_axis = self.normalize(y_axis)
        
        # Rotation Matrix (3x3)
        # Columns correspond to basis vectors
        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = x_axis[0]; R[0][1] = y_axis[0]; R[0][2] = z_axis[0]
        R[1][0] = x_axis[1]; R[1][1] = y_axis[1]; R[1][2] = z_axis[1]
        R[2][0] = x_axis[2]; R[2][1] = y_axis[2]; R[2][2] = z_axis[2]
        
        return R

    # 회전행렬을 오일러각으로 변환(검증용)
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

    def get_depth_average(self, target, img, radius=3):
        x, y = target
        # 이미지 범위를 벗어나지 않도록 슬라이싱 범위 설정 (Clamping)
        y_min = max(0, y - radius)
        y_max = min(img.shape[0], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(img.shape[1], x + radius + 1)

        # 관심 영역(ROI) 추출
        roi = img[y_min:y_max, x_min:x_max]

        # 0(유효하지 않은 값)을 제외하고 평균 계산
        valid_values = roi[roi > 0]
        
        if valid_values.size > 0:
            return np.median(valid_values)
        else:
            return 0
        

    def get_marker_plane_equation(self, corners, depth_img):
        # corners: list of [x, y]
        # Find bounding box of the marker
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # # print(self.depth_resolution)
        # for i in range(4):
        #     i_next = (i + 1) % 4
        #     # 픽셀 좌표와 raw depth 값을 이용해 실제 3D 좌표(mm)로 변환 후 거리 계산
        #     pt1 = self.convert_pixel2mm([xs[i], ys[i], float(depth_img[int(ys[i]), int(xs[i])])])
        #     pt2 = self.convert_pixel2mm([xs[i_next], ys[i_next], float(depth_img[int(ys[i_next]), int(xs[i_next])])])
            
        #     x_diff = pt2[0] - pt1[0]
        #     y_diff = pt2[1] - pt1[1]
        #     z_diff = pt2[2] - pt1[2]
        #     norm = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        #     print(f"norm {i} : {norm}")
        
        # Shrink ROI to center 50% to avoid border noise
        w = max_x - min_x
        h = max_y - min_y
        
        start_x = int(min_x + w * 0.1)
        end_x = int(max_x - w * 0.1)
        start_y = int(min_y + h * 0.1)
        end_y = int(max_y - h * 0.1)
        
        # Ensure ROI is within image bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(depth_img.shape[1], end_x)
        end_y = min(depth_img.shape[0], end_y)
        
        if end_x <= start_x or end_y <= start_y:
            return None

        # Extract depth values in ROI 
        # (중요: 이후 convert_pixel2mm에서 self.depth_resolution이 곱해지므로 평면 방정식은 Raw Depth 단위로 추출해야 함)
        roi_depth = depth_img[start_y:end_y, start_x:end_x]
        
        # Create coordinate grids
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
        
        if np.sum(valid_mask) < 50: # Need enough points for robust fit
            return None, 0, 0
            
        z_valid = z_flat[valid_mask]
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        
        # Robust Plane Fitting: 1/Z vs (u, v) normalized
        # A plane in 3D (AX + BY + CZ + D = 0) corresponds to 
        # 1/Z = -(A/D)*X/Z - (B/D)*Y/Z - (C/D)
        # 1/Z = a * u_norm + b * v_norm + c
        # where u_norm = (u - cx)/fx, v_norm = (v - cy)/fy
        
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
            
            return X, len(z_valid), rmse # [a, b, c], count, rmse
        except np.linalg.LinAlgError:
            return None, 0, 0

    # 마커들의 중심좌표(4*4행렬)
    def detect(self, color_image, depth_image):
        depth_filtered = depth_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        # cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        # cv2.imshow("Main", color_image)
        # cv2.waitKey(1)
        
        marker_centers_result = []
        
        if ids is not None and len(ids) > 0:
            
            for i in range(len(ids)):

                # for j in range(4):
                #     print(f"norm {j} : {corners[i][0][j][0]}, {corners[i][0][j][1]}")

                c = corners[i][0] # c has 4 points
                
                # C++ Logic: Center is average of min/max (AABB center), not centroid
                xs = [pt[0] for pt in c]
                ys = [pt[1] for pt in c]
                xs.sort()
                ys.sort()
                
                x_center = (xs[0] + xs[3]) / 2.0
                y_center = (ys[0] + ys[3]) / 2.0
                
                # Plane Fitting for Rotation
                plane, valid_count, rmse = self.get_marker_plane_equation(c, depth_filtered)
                
                # 만약 노이즈가 너무 심해 평면 방정식을 구하지 못했다면 신뢰할 수 없는 데이터이므로 무시합니다.
                if plane is None:
                    continue
                    
                A, B, C_val = plane
                
                # Recalculate z for corners based on plane
                c_list = []
                for pt in c:
                    u_n = (pt[0] - self.principal_point[0]) / self.fx
                    v_n = (pt[1] - self.principal_point[1]) / self.fy
                    inv_z_est = A*u_n + B*v_n + C_val
                    z_est = 1.0 / inv_z_est if abs(inv_z_est) > 1e-6 else 0
                    
                    # 미리 mm 단위 3D 좌표로 변환해서 저장
                    pt_3d_mm = self.convert_pixel2mm([pt[0], pt[1], float(z_est)])
                    c_list.append(pt_3d_mm)

                # string = f"{c[0][0]},{c[0][1]},{c[1][0]},{c[1][1]},{c[2][0]},{c[2][1]},{c[3][0]},{c[3][1]}"
                # self.logger.save(string)
                    
                #코너점들의 형태를 정사각형으로 보정
                is_valid, c_list_corrected = self.validate_and_correct_marker_shape(c_list)

                # 보정된 코너점들에 Point LPF 적용 (마커 ID 별로 추적)
                marker_id = ids[i][0]
                c_list_filtered = self.apply_point_lpf(marker_id, c_list_corrected)

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

    def detect_stereo(self, main_img, ref_img):
        main_corners, main_ids, main_rejected = cv2.aruco.detectMarkers(main_img, self.dictionary, parameters=self.parameters)
        ref_corners, ref_ids, ref_rejected = cv2.aruco.detectMarkers(ref_img, self.dictionary, parameters=self.parameters)
        
        # cv2.aruco.drawDetectedMarkers(main_img, main_corners, main_ids)
        # cv2.imshow("Main", main_img)
        # cv2.waitKey(1)

        marker_centers_result = []
        
        if main_ids is not None and ref_ids is not None:
            # Group reference markers by ID to handle duplicates
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
                is_valid, c_list_corrected = self.validate_and_correct_marker_shape(corners_3d_mm)

                # 보정된 코너점들에 Point LPF 적용 (마커 ID 별로 추적)
                c_list_filtered = self.apply_point_lpf(mid, c_list_corrected)

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
        

    def stereo_cal_corners_3d_mm(self, main_corner, ref_corner):
        corners_3d_mm = []
        for i in range(4):
            disparity = main_corner[i][0] - ref_corner[i][0]
            if disparity <= 0:
                continue
            depth = (self.baseline * self.fx) / disparity * 1000 # m -> mm
            x_mm = (main_corner[i][0] - self.principal_point[0]) * depth / self.fx
            y_mm = (main_corner[i][1] - self.principal_point[1]) * depth / self.fy
            corners_3d_mm.append([x_mm, y_mm, depth])
        return corners_3d_mm

class Marker_Transform:
    def __init__(self, Stereo=False):
        self.Stereo = Stereo
        
        # Setup Transforms
        T5_to_marker_data = [0.022, 0.0, 0.18, 180, 0.0, -90.0]
        # T5_to_marker_data = [0,0,0,0,0,0]
        # tool_to_cam = [0,0,0,0,0,0]
        tool_to_cam = [0.009,-0.09,-0.085,144,0,180]
        self.T5_to_marker_tf = self.make_transform(T5_to_marker_data)
        self.tool_to_cam_tf = self.make_transform(tool_to_cam)
        
        # Initialize
        self.camera = RealSenseCamera(Stereo=Stereo)
        self.marker_detection = Marker_Detection()
        
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

        depth_resolution = self.camera.get_depth_resolution()
        self.marker_detection.set_depth_resolution(depth_resolution)
        if self.Stereo:
            baseline = self.camera.get_baseline()
            self.marker_detection.set_baseline(baseline)

    def make_transform(self, data):
        # data: [x, y, z, roll, pitch, yaw] (x,y,z in meters, r,p,y in degrees)
        # The C++ code multiplies x,y,z by 1000 inside make_Transform
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

    def get_marker_transform(self, sampling_time=0):
        T5_to_tool_vec = None
        
        # Collection list for sampling
        collected_transforms = []
        start_time = time.time()
        
        # Loop condition:
        # If sampling_time == 0, run once (current behavior).
        # If sampling_time > 0, run loop until time expires.
        
        while True:
            # Check timeout if sampling
            if sampling_time > 0 and (time.time() - start_time > sampling_time):
                break
                
            try:
                # self.camera.capture_image()
                if self.Stereo:
                    left_ir_img = self.camera.get_left_ir_image()
                    right_ir_img = self.camera.get_right_ir_image()
                    if left_ir_img is None or right_ir_img is None:
                        if sampling_time == 0: return None
                        continue
                    marker_transforms = self.marker_detection.detect_stereo(left_ir_img, right_ir_img)
                else:
                    color_img = self.camera.get_color_image()
                    depth_img = self.camera.get_depth_image()
                    if color_img is None or depth_img is None:
                        if sampling_time == 0: return None
                        continue
                    marker_transforms = self.marker_detection.detect(color_img, depth_img)
                
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

def create_robot(ip):
    robot = rby.create_robot_a(ip)
    robot.connect()
    robot.power_on(".*")
    robot.servo_on(".*")
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

            result = marker_transform.get_marker_transform(sampling_time=2)

            if result is None:
                print("Marker not detected.")
                continue

            T_meas = np.array(result).reshape(4, 4)

            q_cmd_list.append(q_cmd)
            T_meas_list.append(T_meas)

            print(f"Captured sample {len(q_cmd_list)}")
            print("q =", np.round(q_current[RIGHT_ARM_IDX],3))
            T_print = np.array(result).reshape(4, 4)
            print("marker =", np.round(T_print,3))
        elif key == 'q':
            break

        time.sleep(0.05)

    return np.array(q_cmd_list), np.array(T_meas_list)


# ============================================================
# Gauss Newton (기존 알고리즘 그대로)
# ============================================================

def optimize(robot, dyn_model,
             q_cmd_list, T_meas_list,
             RIGHT_ARM_IDX, ndof):

    q_nominal = robot.get_state().position.copy()

    q_offset = np.zeros(7)
    xi_cam = np.zeros(6)

    optimize_camera = (ndof == 13)

    max_iter = 500
    eps = 1e-6

    for it in range(max_iter):

        if optimize_camera:
            H = np.zeros((13, 13))
            g = np.zeros(13)
        else:
            H = np.zeros((7, 7))
            g = np.zeros(7)

        total_err = 0

        for q_cmd, T_meas in zip(q_cmd_list, T_meas_list):

            q_full = q_nominal.copy()
            q_full[RIGHT_ARM_IDX[:7]] = q_cmd + q_offset

            state = dyn_model.make_state(
                ["link_torso_5", "ee_right"],
                robot.model().robot_joint_names
            )

            state.set_q(q_full)
            dyn_model.compute_forward_kinematics(state)
            dyn_model.compute_diff_forward_kinematics(state)

            T_fk = dyn_model.compute_transformation(state, 0, 1)

            if optimize_camera:
                T_model = T_fk @ se3_exp(xi_cam)
            else:
                T_model = T_fk

            T_err = np.linalg.inv(T_model) @ T_meas
            xi = se3_log(T_err)

            Jb = dyn_model.compute_body_jacobian(state, 0, 1)

            if optimize_camera:
                J = np.zeros((6, 13))
                J[:, :7] = Jb[:, RIGHT_ARM_IDX[:7]]
                J[:, 7:] = np.eye(6)
            else:
                J = Jb[:, RIGHT_ARM_IDX[:7]]

            H += J.T @ J
            g += J.T @ xi
            total_err += np.linalg.norm(xi)

        dx = np.linalg.pinv(H) @ g

        q_offset += dx[:7]

        if optimize_camera:
            xi_cam += dx[7:]

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
                        choices=[7, 13])
    parser.add_argument("--ip", type=str,
                        default="192.168.30.1:50051")

    args = parser.parse_args()

    robot = create_robot(args.ip)
    dyn_model = robot.get_dynamics()
    model = robot.model()
    RIGHT_ARM_IDX = model.right_arm_idx

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

    print("Dataset saved.")

    q_offset, xi_cam = optimize(
        robot, dyn_model,
        q_cmd_list,
        T_meas_list,
        RIGHT_ARM_IDX,
        args.ndof
    )

    print("\n===== RESULT =====")
    print("Joint offset (deg):")
    print(np.rad2deg(q_offset))
    print("Camera xi:")
    print(xi_cam)

    marker_transform.camera.monitoring(Flag=False)


if __name__ == "__main__":
    main()