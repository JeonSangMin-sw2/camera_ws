
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading


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
                
                left_ir_profile = self.profile.get_stream(rs.stream.infrared, 1)
                extrinsics = depth_stream.get_extrinsics_to(left_ir_profile)
                self.baseline = abs(extrinsics.translation[0])
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
        
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        cv2.imshow("Main", color_image)
        cv2.waitKey(1)
        
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
        cv2.aruco.drawDetectedMarkers(main_img, main_corners, main_ids)
        
        cv2.imshow("Main", main_img)
        cv2.waitKey(1)

        marker_centers_result = []
        
        if main_ids is not None and ref_ids is not None and len(main_ids) == len(ref_ids):
            # main_ids와 ref_ids의 개수가 서로 다를 경우에 관한 로직 추후게 구현해야함
            print(main_ids)
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
        T5_to_marker_data = [0.022, 0.0, 0.25, 180, 0.0, -90.0]
        
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



def main():
    marker_transform = None
    try:
        marker_transform = Marker_Transform(Stereo=True)
        #marker_transform.camera.monitoring()
        while True:
            result = marker_transform.get_marker_transform()
            if result is None:
                continue
            print(result[0],result[1],result[2],result[3])
            print(result[4],result[5],result[6],result[7])
            print(result[8],result[9],result[10],result[11])
            print(result[12],result[13],result[14],result[15])
            time.sleep(0.01) # Removed sleep for better responsiveness
    except RuntimeError as e:
        print(f"Initialization Error: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        if marker_transform is not None:
            #marker_transform.camera.monitoring(Flag=False)
            print("Camera Stopped.")

if __name__ == "__main__":
    main()
