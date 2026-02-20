
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading

###
# 데이터를 텍스트 파일로 저장하는 클래스
class File_Logger:
    def __init__(self, filepath="log.txt"):
        self.filepath = filepath

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")

###
# 해당 클래스는 마커인식을 위한 기능들을 realsense 카메라를 기준으로 정의한 클래스 
# 다른 카메라로 기능을 사용할 경우 나머지 클래스와의 연동을 위해 함수의 양식은 일치시켜주어야함
# 필수 구현함수 : capture_image(), get 이라 들어간 모든 함수

###
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
        depth_scale = depth_sensor.get_depth_scale()
        print("depth scale : ", depth_scale)
        self.depth_resolution = depth_scale*1000

        #나머지 카메라 구동을 위한 파라미터 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()

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
                        min_dist = float(np.min(self.depth_image[self.depth_image > 0]))
                        max_dist = float(np.max(self.depth_image))
                        alpha = (0.0 - 200.0) / (max_dist - min_dist)
                        beta = 200.0 - (min_dist * alpha)
                    else:
                        min_dist = 70.0
                        max_dist = 2000.0
                        alpha = (0.0 - 200.0) / (max_dist - min_dist)
                        beta = 200.0 - (min_dist * alpha)
                    depth_re_img = self.depth_image.astype(np.float32)
                    depth_re_img = depth_re_img * alpha + beta
                    depth_re_img = np.clip(depth_re_img, 0, 255).astype(np.uint8)
                    depth_re_img[self.depth_image == 0] = 0
                    depth_re_img_bgr = cv2.cvtColor(depth_re_img, cv2.COLOR_GRAY2BGR)
                    result_list.append(depth_re_img_bgr)
                if self.Infrared == True and self.left_ir_image is not None and self.right_ir_image is not None:
                    result_list.append(self.left_ir_image)
                    result_list.append(self.right_ir_image)
                
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
        align_to = rs.stream.color
        align = rs.align(align_to)
        if self.camera_running:
            if self.thread is not None and self.thread.is_alive() and threading.current_thread() != self.thread:
                time.sleep(1.0/self.fps)
                return

            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            if self.Infrared:
                left_ir_frame = frames.get_infrared_frame(1)
                right_ir_frame = frames.get_infrared_frame(2)
                
                if not left_ir_frame or not right_ir_frame:
                    print("no frame")
                    return
                left_ir_data = np.asanyarray(left_ir_frame.get_data())
                right_ir_data = np.asanyarray(right_ir_frame.get_data())
            else:
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    print("no frame")
                    return
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
            
            with self.lock:
                if self.Infrared:
                    self.left_ir_image = left_ir_data
                    self.right_ir_image = right_ir_data
                else:
                    self.color_image = color_data
                    self.depth_image = depth_data

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
        # 어떤 마커를 인식시킬건지 정의
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        # 마커 인식 정밀도 향상을 위한 파라미터 튜닝
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 서브픽셀 활성화
        self.parameters.adaptiveThreshWinSizeMin = 3  # 적응형 임계값 최소 윈도우 크기
        self.parameters.adaptiveThreshWinSizeMax = 23 # 적응형 임계값 최대 윈도우 크기
        #self.parameters.adaptiveThreshWinSizeStep = 10 # 윈도우 크기 증가 단계
        self.parameters.adaptiveThreshConstant = 9 # 임계값 상수 (조명 변화에 강하게)
        self.parameters.polygonalApproxAccuracyRate = 0.02 # 다각형 근사 정확도 (낮을수록 정밀)
        self.parameters.minDistanceToBorder = 5 # 마커와 이미지 경계 사이의 최소 거리
        self.parameters.useAruco3Detection = True # 아루코 3.0 디텍터 사용

        # 계산에 쓰일 카메라 내부 파라미터
        self.principal_point = [0, 0]
        self.fx = 0
        self.fy = 0
        self.depth_resolution = 1

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
    
    # 마커의 회전행렬 계산
    def get_rotation_matrix(self, corners):
        # corners: list of [x, y]
        pts_3d = []
        for i in range(4):
            cx, cy = corners[i][0], corners[i][1]
            d = corners[i][2]
            pts_3d.append(self.convert_pixel2mm([cx, cy, float(d)]))

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
                c = corners[i][0] # c has 4 points
                
                # C++ Logic: Center is average of min/max (AABB center), not centroid
                xs = [pt[0] for pt in c]
                ys = [pt[1] for pt in c]
                xs.sort()
                ys.sort()
                
                x_center = (xs[0] + xs[3]) / 2.0
                y_center = (ys[0] + ys[3]) / 2.0
                
                # Get depth
                z = 0
                # C++ uses simple cast (truncation), not rounding
                iy, ix = int(y_center), int(x_center)
                if 0 <= iy < depth_filtered.shape[0] and 0 <= ix < depth_filtered.shape[1]:
                    z = self.get_depth_average([ix, iy], depth_filtered, 5)
                if z == 0:
                    continue
                # Pixel to MM
                # center_pos logic kept same but verify usage of z
                # However, z from center is used for translation. Can we improve translation too?
                # For now keep center logic as requested, just fix Rotation
                
                # Plane Fitting for Rotation
                plane, valid_count, rmse = self.get_marker_plane_equation(c, depth_filtered)
                
                # print(f"Plane Fit: Count={valid_count}, RMSE={rmse:.6f}")
                
                c_list = []
                if plane is not None:
                    # Recalculate z for corners based on plane
                    # Plane is 1/Z = A*u_norm + B*v_norm + C
                    A, B, C_val = plane
                    for pt in c:
                        # pt is [x, y]
                        u_n = (pt[0] - self.principal_point[0]) / self.fx
                        v_n = (pt[1] - self.principal_point[1]) / self.fy
                        inv_z_est = A*u_n + B*v_n + C_val
                        if abs(inv_z_est) > 1e-6:
                            z_est = 1.0 / inv_z_est
                        else:
                            z_est = 0
                        c_list.append([pt[0], pt[1], z_est])
                else:
                    # Fallback to old method (corner sampling)
                    c_list = [[pt[0], pt[1], depth_filtered[int(pt[1])][int(pt[0])]] for pt in c]

                # Recalculate center Z based on plane (Optional but better)
                # If plane is good, center Z from plane is better than median of pixels
                if plane is not None:
                    A, B, C_val = plane
                    u_center_n = (x_center - self.principal_point[0]) / self.fx
                    v_center_n = (y_center - self.principal_point[1]) / self.fy
                    inv_z_center = A*u_center_n + B*v_center_n + C_val
                    if abs(inv_z_center) > 1e-6:
                        z = 1.0 / inv_z_center
                
                center_pos = self.convert_pixel2mm([x_center, y_center, float(z)])
                
                # Rotation Matrix
                rot_matrix = self.get_rotation_matrix(c_list)
                
                # RPY(디버깅용)
                rpy = self.get_rpy_from_matrix(rot_matrix)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                # print(f"Center [{transform[3]}, {transform[7]}, {transform[11]}]")
                # print(f"rpy    [{rpy[0]*180/math.pi}, {rpy[1]*180/math.pi}, {rpy[2]*180/math.pi}]")
                
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
                
                # Find best matching candidate based on Y-coordinate similarity (Epipolar constraint)
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
                ref_corner = candidates.pop(best_idx) # Remove matched candidate to prevent double counting
                if len(candidates) == 0:
                    del ref_dict[mid]

                # Get depth
                corners_3d_mm = self.stereo_cal_corners_3d_mm(main_corner, ref_corner)

                c = corners_3d_mm # c has 4 points
                
                # C++ Logic: Center is average of min/max (AABB center), not centroid
                xs = [pt[0] for pt in c]
                ys = [pt[1] for pt in c]
                zs = [pt[2] for pt in c]
                xs.sort()
                ys.sort()
                zs.sort()
                
                x_center = (xs[0] + xs[3]) / 2.0
                y_center = (ys[0] + ys[3]) / 2.0
                z_center = (zs[0] + zs[3]) / 2.0
                
                # Rotation Matrix
                c_list = [[pt[0], pt[1], pt[2]] for pt in c]
                rot_matrix = self.get_rotation_matrix(c_list) # 사용중인 함수의 양식을 맞추기 위함
                
                # RPY
                rpy = self.get_rpy_from_matrix(rot_matrix)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], x_center,
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], y_center,
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], z_center,
                    0.0, 0.0, 0.0, 1.0
                ]
                print(f"Center [{transform[3]}, {transform[7]}, {transform[11]}]")
                print(f"rpy    [{rpy[0]*180/math.pi}, {rpy[1]*180/math.pi}, {rpy[2]*180/math.pi}]")
                
                marker_centers_result.append(transform)
                
        return marker_centers_result
        

    def stereo_cal_corners_3d_mm(self, main_corner, ref_corner):
        corners_3d_mm = []
        for i in range(4):
            disparity = main_corner[i][0] - ref_corner[i][0]
            if disparity == 0:
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
        tool_to_cam = [0,0,0,0,0,0]
        # tool_to_cam = [0.009,-0.09,-0.085,144,0,180]
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
        
        return m

    def get_marker_transform(self, sampling_time=0):
        T5_to_tool_tf = None
        T5_to_tool_vec = None
        
        # Collection list for sampling
        collected_vectors = []
        start_time = time.time()
        
        # Loop condition:
        # If sampling_time == 0, run once (current behavior).
        # If sampling_time > 0, run loop until time expires.
        
        while True:
            # Check timeout if sampling
            if sampling_time > 0 and (time.time() - start_time > sampling_time):
                break
                
            try:
                self.camera.capture_image()
                if self.Stereo:
                    left_ir_img = self.camera.get_left_ir_image()
                    right_ir_img = self.camera.get_right_ir_image()
                else:
                    color_img = self.camera.get_color_image()
                    depth_img = self.camera.get_depth_image()
                
                if self.Stereo:
                    if left_ir_img is None or right_ir_img is None:
                        if sampling_time == 0: return None
                        continue
                    marker_transforms = self.marker_detection.detect_stereo(left_ir_img, right_ir_img)
                else:
                    if color_img is None or depth_img is None:
                        if sampling_time == 0: return None
                        continue
                    marker_transforms = self.marker_detection.detect(color_img, depth_img)
                
                current_vec = None
                for tf_list in marker_transforms:
                    # Convert flattened list to 4x4 matrix
                    camera_to_marker_tf = np.array(tf_list, dtype=np.float32).reshape(4, 4)
                    
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
                    except np.linalg.LinAlgError:
                        print("Singular matrix, cannot invert")
                        continue
                
                # If valid vector found
                if T5_to_tool_vec is not None:
                    if sampling_time == 0:
                        return T5_to_tool_vec
                    else:
                        collected_vectors.append(T5_to_tool_vec)
                        
            except KeyboardInterrupt:
                raise
            
            # If not sampling, break after one attempt (handled by return above)
            # If sampling, continue loop
            if sampling_time == 0: 
                break
                
        # Post-processing for sampling
        if sampling_time > 0:
            if not collected_vectors:
                return None
            
            data = np.array(collected_vectors) # Shape (N, 16)
            
            # Separate translation and rotation
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
            
            # Reconstruct the 4x4 flattened vector
            final_vec = np.zeros(16, dtype=np.float32)
            final_vec[0:3] = final_R[0, :]
            final_vec[4:7] = final_R[1, :]
            final_vec[8:11] = final_R[2, :]
            
            final_vec[3] = final_translation[0]
            final_vec[7] = final_translation[1]
            final_vec[11] = final_translation[2]
            
            final_vec[15] = 1.0
            
            return final_vec
            
        return None



def main():
    logger = File_Logger()
    marker_transform = None
    try:
        marker_transform = Marker_Transform(Stereo=False)
        marker_transform.camera.monitoring()
        while True:
            result = marker_transform.get_marker_transform(sampling_time=3)
            if result is None:
                continue
            result = [round(n, 4) for n in result]
            print(result[0],result[1],result[2],result[3])
            print(result[4],result[5],result[6],result[7])
            print(result[8],result[9],result[10],result[11])
            print(result[12],result[13],result[14],result[15])
            string = f"{result[3]},{result[7]},{result[11]}"
            logger.save(string)
            time.sleep(0.01) # Removed sleep for better responsiveness
    except RuntimeError as e:
        print(f"Initialization Error: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        if marker_transform is not None:
            marker_transform.camera.monitoring(Flag=False)
            print("Camera Stopped.")

if __name__ == "__main__":
    main()
