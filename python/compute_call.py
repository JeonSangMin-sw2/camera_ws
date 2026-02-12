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


class RealSenseCamera:
    def __init__(self, serial_number=None, Stereo=False):
        # 별도로 설정한 시리얼 번호가 없으면 처음 인식되는 카메라 사용
        self.device_number = 0
        # 연결되어있던 카메라 검색
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices found!")
            raise RuntimeError("No RealSense connected")
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} (Serial: {dev.get_info(rs.camera_info.serial_number)})")
            if serial_number == dev.get_info(rs.camera_info.serial_number):
                self.device_number = i
                break
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
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            if self.Infrared:
                print("Infrared camera is used")
                self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
                self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
            else:
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # 파이프라인 시작
            self.profile = self.pipeline.start(self.config)

            # 카메라 안정화를 위해 10프레임 정도는 무시하고 사용
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
                # Monitoring thread handles capture, just wait or check cached data
                # We can sleep briefly to mimic frame rate
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
        # 카메라 내부 파라미터
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

    # 마커들의 중심좌표(4*4행렬)
    def detect(self, color_image, depth_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
        # cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        # cv2.imshow("Main", color_image)
        # cv2.waitKey(1)
        
        marker_centers_result = []
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
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
                if 0 <= iy < depth_image.shape[0] and 0 <= ix < depth_image.shape[1]:
                    z = depth_image[iy, ix]
                
                # Draw center
                cv2.circle(color_image, (ix, iy), 5, (0, 0, 255), -1)
                
                # Pixel to MM
                center_pos = self.convert_pixel2mm([x_center, y_center, float(z)])
                
                # Rotation Matrix
                c_list = [[pt[0], pt[1],depth_image[int(pt[1])][int(pt[0])]] for pt in c]
                rot_matrix = self.get_rotation_matrix(c_list)
                
                # RPY
                rpy = self.get_rpy_from_matrix(rot_matrix)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                print(f"id : {ids[i][0]}")
                print(f"Center [{transform[3]}, {transform[7]}, {transform[11]}]")
                print(f"rpy    [{rpy[0]*180/math.pi}, {rpy[1]*180/math.pi}, {rpy[2]*180/math.pi}]")
                
                marker_centers_result.append(transform)
                
        return marker_centers_result

    def detect_stereo(self, main_img, ref_img):
        main_corners, main_ids, main_rejected = cv2.aruco.detectMarkers(main_img, self.dictionary, parameters=self.parameters)
        ref_corners, ref_ids, ref_rejected = cv2.aruco.detectMarkers(ref_img, self.dictionary, parameters=self.parameters)
        
        # cv2.aruco.drawDetectedMarkers(main_img, main_corners, main_ids)
        # cv2.imshow("Main", main_img)
        # cv2.waitKey(1)

        marker_centers_result = []
        
        if main_ids is not None and ref_ids is not None and len(main_ids) == len(ref_ids):
            # main_ids와 ref_ids의 개수가 서로 다를 경우에 관한 로직 추후게 구현해야함
            
            for i in range(len(main_ids)):
                if main_ids[i] != ref_ids[i]:
                    continue
                # Get depth
                corners_3d_mm = self.stereo_cal_corners_3d_mm(main_corners[i][0], ref_corners[i][0])

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
                
                print(f"id : {main_ids[i][0]}")
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
        
        self.T5_to_marker_tf = self.make_transform(T5_to_marker_data)
        
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

    def get_marker_transform(self):
        T5_to_cam_tf = None
        T5_to_cam_vec = None
        # print("RealSense Camera Started. Press 'ESC' to exit.")
        # time.sleep(1) # Warmup - Moved or removed for loop performance
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
                    return None
                marker_transforms = self.marker_detection.detect_stereo(left_ir_img, right_ir_img)
            else:
                if color_img is None or depth_img is None:
                    return None
                marker_transforms = self.marker_detection.detect(color_img, depth_img)
            
            for tf_list in marker_transforms:
                # Convert flattened list to 4x4 matrix
                camera_to_marker_tf = np.array(tf_list, dtype=np.float32).reshape(4, 4)
                
                try:
                    camera_to_marker_inv = np.linalg.inv(camera_to_marker_tf)
                    # base_to_tool = base_to_marker * camera_to_marker^-1 * camera_to_tool
                    T5_to_cam_tf = self.T5_to_marker_tf @ camera_to_marker_inv
                    T5_to_cam_vec = T5_to_cam_tf.flatten()
                    if abs(T5_to_cam_vec[3]) > 4 :
                        T5_to_cam_vec[3] = T5_to_cam_vec[3]/1000
                        T5_to_cam_vec[7] = T5_to_cam_vec[7]/1000
                        T5_to_cam_vec[11] = T5_to_cam_vec[11]/1000
                except np.linalg.LinAlgError:
                    print("Singular matrix, cannot invert")

        except KeyboardInterrupt:
            raise

        return T5_to_cam_vec

# ===============================
# Lie algebra utils
# ===============================
def adjoint(T):
    R = T[:3,:3]
    p = T[:3,3]
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = p_hat @ R
    return Ad

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

    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )


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

        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = V @ v
    return T

def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([
        w_hat[2,1],
        w_hat[0,2],
        w_hat[1,0]
    ])


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = t
    else:
        w_hat = np.array([
            [    0, -w[2],  w[1]],
            [ w[2],     0, -w[0]],
            [-w[1],  w[0],     0]
        ]) / theta

        A = (
            np.eye(3)
            - 0.5 * w_hat
            + (1/theta**2) * (1 - theta/(2*np.tan(theta/2)))
            * (w_hat @ w_hat)
        )
        v = A @ t

    return np.hstack([w, v])   # (6,)


# ===============================
# Robot setup
# ===============================
robot = rby.create_robot_a("192.168.30.1:50051")
model = robot.model()

robot.connect()
robot.power_on(".*")
robot.servo_on(".*")
robot.reset_fault_control_manager()
robot.enable_control_manager(False)

dyn_model = robot.get_dynamics()

RIGHT_ARM_IDX = model.right_arm_idx[:7]
ndof = len(RIGHT_ARM_IDX)

BASE, EE = 0, 1
# def compute_T_fk(q):
#     model = robot.model()
#     dyn_robot = robot.get_dynamics()
#     dyn_state = dyn_robot.make_state(["link_torso_5", "ee_right"], model.robot_joint_names)
#     q_ = q
#     dyn_state.set_q(q_)
#     dyn_robot.compute_forward_kinematics(dyn_state)
#     T_fk = dyn_robot.compute_transformation(dyn_state, BASE, EE)
#     return np.round(T_fk[:3, 3],2)

marker_transform = Marker_Transform(Stereo=True)
marker_transform.camera.monitoring(Flag=True)


print("\nPress 'e' + Enter to capture (q + marker)")
print("Press 'q' + Enter to quit\n")

q_cmd_list = []
T_meas_list = []
while True:
    key = check_key_press()

    if key == 'e':
        # 1️⃣ 현재 로봇 상태 읽기
        state = robot.get_state()
        q_current = state.position.copy()
        q_full = state.position.copy()       # 또는 현재 사용중인 q 읽는 함수
        q_cmd = q_full[RIGHT_ARM_IDX].copy()
        
        # 2️⃣ 마커 pose 읽기
        result = marker_transform.get_marker_transform()
        if result is None:
            print("Marker not detected.")
            continue
        
        T_meas = np.array(result).reshape(4, 4)
        
        # 3️⃣ 같이 저장
        q_cmd_list.append(q_cmd)
        T_meas_list.append(T_meas)
        
        # recorded_data.append((q_current, result))

        print("Captured!")
        print("q =", q_current[RIGHT_ARM_IDX])
        print("marker =", result)

    elif key == 'q':
        print("Exiting capture loop.")
        break

    time.sleep(0.05)

# 저장
if len(q_cmd_list) > 0:

    np.savez_compressed(
        "captured_dataset.npz",
        q=np.array(q_cmd_list),
        marker=np.array(T_meas_list)
    )

    print(f"\nSaved {len(q_cmd_list)} samples to captured_dataset.npz")

else:
    print("No data captured.")
    
# ===============================
# Ground truth offset (simulation)
# ===============================
# q_offset_true = np.deg2rad([0.5, -1.0, 1.0, 0.5, -5.0, 0.5, 0.2])
# q_offset_true = np.deg2rad([5, -5, 2, 5, -5, 5, 2])
# xi_cam_true = np.array([0.02, -0.04, 0.03,   # rotation
#                         0.01, 0.02, -0.015]) # translation

# expected position about camera braket
xi_cam_pose = np.array([0, 0, 0,   # rotation
                        0, 0, 0]) # translation
T_cam_pose = se3_exp(xi_cam_pose)
T_cam_true = se3_exp(xi_cam_true)

# ===============================
# Command poses
# ===============================
joint_limits = np.array([
    [-2.0,  2.0],
    [-2.5,  0.0],
    [-1.5,  1.5],
    [-2.5,  0.0],
    [-3.141592654,  3.141592654],
    [-1.570796327,  1.570796327],
    [-1.570796327,  1.570796327]
])

# def generate_random_q_list(n_samples=10, margin_ratio=0.15, seed=42):
#     rng = np.random.default_rng(seed)
#     q_list = []
#     for _ in range(n_samples):
#         q = []
#         for lo, hi in joint_limits:
#             span = hi - lo
#             q.append(rng.uniform(lo + margin_ratio*span,
#                                  hi - margin_ratio*span))
#         q_list.append(np.array(q))
#     return q_list

# q_cmd_list = generate_random_q_list(n_samples=100)


# ===============================
# Nominal configuration
# ===============================
q_nominal = robot.get_state().position.copy()


# ===============================
# Fake camera measurements
# ===============================
T_cam_list = []

# for q_cmd in q_cmd_list:
#     q_full = q_nominal.copy()
#     q_full[RIGHT_ARM_IDX] = q_cmd + q_offset_true

#     dyn_state = dyn_model.make_state(
#         ["link_torso_5", "ee_right"],
#         model.robot_joint_names
#     )
#     dyn_state.set_q(q_full)
#     dyn_model.compute_forward_kinematics(dyn_state)

#     T_cam = dyn_model.compute_transformation(dyn_state, BASE, EE)
#     # T_cam_list.append(T_cam)
     
#     T_meas = T_cam @ T_cam_pose @ T_cam_true 
#     T_cam_list.append(T_meas)
# ===============================
# Gauss–Newton Offset Calibration
# ===============================
max_iter = 500
eps = 1e-3

q_offset = np.zeros(ndof)
xi_cam = np.zeros(6)

for it in range(max_iter):

    H = np.zeros((ndof+6, ndof+6))
    g = np.zeros(ndof+6)

    total_err = 0.0

    for q_cmd, T_meas in zip(q_cmd_list, T_meas_list):

        # 🔁 재선형화 지점
        q_full = q_nominal.copy()
        q_full[RIGHT_ARM_IDX] = q_cmd + q_offset

        dyn_state = dyn_model.make_state(
            ["link_torso_5", "ee_right"],
            model.robot_joint_names
        )
        dyn_state.set_q(q_full)
        dyn_model.compute_forward_kinematics(dyn_state)
        dyn_model.compute_diff_forward_kinematics(dyn_state)

        T_fk = dyn_model.compute_transformation(dyn_state, BASE, EE)

        # ---- Camera extrinsic ----
        T_extrinsic = se3_exp(xi_cam)
        # ---- Full model ----
        T_model = T_fk @ T_cam_pose @ T_extrinsic
        
        
        # T_err = T_model @ np.linalg.inv(T_meas)
        # xi = se3_log(T_err)   # space error
        T_err =  T_meas @ np.linalg.inv(T_model) 
        xi = se3_log(T_err)   # space error

        Jb = dyn_model.compute_space_jacobian(dyn_state, BASE, EE) 
        
        xi[3:] *= 0.1
        # J_joint[3:, :] *= 0.1
        # Jb[:, 7:][3:, :] *= 0.1
        # J_joint[:, 7:][3:, :] *= 0.1
        
        # Camera Jacobian = Identity
        J = np.zeros((6,13))
        J[:, :7] = Jb[:, RIGHT_ARM_IDX]
        # J[:, 7:] = np.eye(6)
        J[:, 7:] = adjoint(T_fk @ T_cam_pose)
        
        H += J.T @ J
        g += J.T @ xi
        total_err += np.linalg.norm(xi)

    
    dx = np.linalg.pinv(H) @ g
    q_offset += dx[:7]
    xi_cam += dx[7:]

    print(f"[Iter {it:02d}] |dq| = {np.linalg.norm(dx):.3e}, "
          f"|xi| = {total_err:.3e}")

    if np.linalg.norm(dx) < eps:
        print("Converged.")
        break


# ===============================
# Result
# ===============================
print("\n===== Joint Offset =====")
# print("True offset [deg]:")
# print(np.round(np.rad2deg(q_offset_true), 4))

print("Estimated offset [deg]:")
print(np.round(np.rad2deg(q_offset), 4))


print("\n===== Camera Extrinsic (xi) =====")
# print("True xi_cam:")
# print(np.round(xi_cam_true, 6))

print("Estimated xi_cam:")
print(np.round(xi_cam, 6))

marker_transform.camera.monitoring(Flag=False)