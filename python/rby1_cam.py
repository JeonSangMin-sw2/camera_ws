
import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import math
import time
import threading


class RealSenseCamera:
    def __init__(self, serial_number=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial_number = serial_number
        self.camera_running = True
        self.color_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        
        self.width = 640
        self.height = 480
        self.fps = 30
        self.focal_length = 0.0
        self.principal_point = [0.0, 0.0]
        self.intrinsics = None
        self.profile = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        
        if self.serial_number:
            self.config.enable_device(self.serial_number)
            
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        self.profile = self.pipeline.start(self.config)
        
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.depth_intrinsics = depth_stream.get_intrinsics()
        
        # Calculate focal length: sqrt(fx^2 + fy^2) as in C++ code
        self.focal_length = math.sqrt(self.depth_intrinsics.fx**2 + self.depth_intrinsics.fy**2) # pixel
        self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy] #pixel
        
        print(f"Focal Length: {self.focal_length}")
        print(f"Principal Point: {self.principal_point[0]}, {self.principal_point[1]}")

    def start(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            while self.camera_running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                with self.lock:
                    self.color_image = color_data
                    self.depth_image = depth_data
                    
        except RuntimeError as e:
            print(f"RealSense Error: {e}")
        except Exception as e:
            print(f"Standard Error: {e}")
        finally:
            self.pipeline.stop()

    def stop(self):
        self.camera_running = False

    def capture_image(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        if self.camera_running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("no frame")
            
            # Convert to numpy arrays
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            with self.lock:
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

    def get_principal_point_and_focal_length(self):
        return [self.principal_point[0], self.principal_point[1], self.focal_length]


class Marker_Detection:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.principal_point = [0, 0]
        self.focal_length = 0

    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.focal_length = param[2]

    def convert_pixel2mm(self, center):
        # center: [x, y, z] (z is depth value in mm)
        if not center:
            return center
        
        if self.focal_length == 0:
            return center 

        x = (center[0] - self.principal_point[0]) * center[2] / self.focal_length
        y = (center[1] - self.principal_point[1]) * center[2] / self.focal_length
        z = center[2]
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
    
    def get_rotation_matrix(self, corners, depth_data):
        # corners: list of [x, y]
        pts_3d = []
        for i in range(4):
            cx, cy = corners[i]
            d = 0
            # C++ behavior: truncation
            iy, ix = int(cy), int(cx)
            if 0 <= iy < depth_data.shape[0] and 0 <= ix < depth_data.shape[1]:
                d = depth_data[iy, ix]
            
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

    def detect(self, color_image, depth_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
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
                c_list = [[pt[0], pt[1]] for pt in c]
                rot_matrix = self.get_rotation_matrix(c_list, depth_image)
                
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

class Marker_Transform:
    def __init__(self, serial_number=None):
        # Setup Transforms
        T5_to_marker_data = [0.022, 0.0, 0.25, 180, 0.0, -90.0]
        
        self.T5_to_marker_tf = self.make_transform(T5_to_marker_data)
        
        # Initialize
        self.camera = RealSenseCamera(serial_number)
        self.marker_detection = Marker_Detection()
        
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

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

    def get_marker_transform(self,Visualization=False):
        T5_to_cam_vec = None
        # print("RealSense Camera Started. Press 'ESC' to exit.")
        # time.sleep(1) # Warmup - Moved or removed for loop performance
        try:
            self.camera.capture_image()
            color_img = self.camera.get_color_image()
            depth_img = self.camera.get_depth_image()
            
            if color_img is None or depth_img is None:
                return None
            
            marker_transforms = self.marker_detection.detect(color_img, depth_img)
            
            for tf_list in marker_transforms: # 마커 여러개일 때 처리할 기능도 추가해야함
                # Convert flattened list to 4x4 matrix
                camera_to_marker_tf = np.array(tf_list, dtype=np.float32).reshape(4, 4)
                
                try:
                    camera_to_marker_inv = np.linalg.inv(camera_to_marker_tf)
                    # base_to_tool = base_to_marker * camera_to_marker^-1 * camera_to_tool
                    T5_to_cam_tf = self.T5_to_marker_tf @ camera_to_marker_inv
                    T5_to_cam_vec = T5_to_cam_tf.flatten()
                    if T5_to_cam_vec[3] > 4 :
                        T5_to_cam_vec[3] = T5_to_cam_vec[3]/1000
                        T5_to_cam_vec[7] = T5_to_cam_vec[7]/1000
                        T5_to_cam_vec[11] = T5_to_cam_vec[11]/1000
                except np.linalg.LinAlgError:
                    print("Singular matrix, cannot invert")

        except KeyboardInterrupt:
            raise
        # finally:
            # self.camera.stop()
            # cv2.destroyAllWindows() # Do not destroy windows every frame if looping
        # print("Camera Stopped.")

        return T5_to_cam_vec

    def stop(self):
        self.camera.stop()
        cv2.destroyAllWindows()

def main():
    marker_transform = Marker_Transform()
    try:
        while True:
            result = marker_transform.get_marker_transform(Visualization=True)
            if result is None:
                continue
            print(result[0],result[1],result[2],result[3])
            print(result[4],result[5],result[6],result[7])
            print(result[8],result[9],result[10],result[11])
            print(result[12],result[13],result[14],result[15])
            # time.sleep(0.01) # Removed sleep for better responsiveness
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        marker_transform.stop()
        print("Camera Stopped.")

if __name__ == "__main__":
    main()
