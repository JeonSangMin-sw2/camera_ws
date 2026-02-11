
import math
import numpy as np

# ==========================================
# C++ Logic Simulation (Ported from main.cpp)
# ==========================================

class CPP_Logic:
    def __init__(self):
        self.principal_point = [320.0, 240.0] # Example
        self.focal_length = 600.0 # Example

    def convert_pixel2mm_vec3(self, center):
        # center: [x, y, z]
        # logic: float x = (center[0] - pp[0]) * center[2] / focal
        result = [0.0]*3
        result[0] = (center[0] - self.principal_point[0]) * center[2] / self.focal_length
        result[1] = (center[1] - self.principal_point[1]) * center[2] / self.focal_length
        result[2] = center[2]
        return result

    def normalize(self, v):
        m = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if m > 1e-9:
             v[0]/=m; v[1]/=m; v[2]/=m
        return v

    def cross(self, a, b):
        return [ a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ]

    def get_rotation_matrix(self, corners, depth_val):
        # corners: 4 points [x, y]
        # depth_val: simulated uniform depth
        
        # In C++, corners are converted to 3D first
        corners_3d = []
        for i in range(4):
            # C++ uses depth_data.at<ushort>(y, x). Here we simulate constant depth.
            pos = [corners[i][0], corners[i][1], float(depth_val)]
            corners_3d.append(self.convert_pixel2mm_vec3(pos))

        x_axis = [0.0]*3
        y_axis = [0.0]*3
        z_axis = [0.0]*3

        # x_axis = (corners[1] + corners[2]) - (corners[0] + corners[3]);
        for i in range(3):
            x_axis[i] = (corners_3d[1][i] + corners_3d[2][i]) - (corners_3d[0][i] + corners_3d[3][i])
            y_axis[i] = (corners_3d[3][i] + corners_3d[2][i]) - (corners_3d[0][i] + corners_3d[1][i])

        x_axis = self.normalize(x_axis)
        y_axis = self.normalize(y_axis)
        
        z_axis = self.cross(x_axis, y_axis)
        z_axis = self.normalize(z_axis)

        y_axis = self.cross(z_axis, x_axis)
        y_axis = self.normalize(y_axis)

        # Rotation Matrix
        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = x_axis[0]; R[0][1] = y_axis[0]; R[0][2] = z_axis[0]
        R[1][0] = x_axis[1]; R[1][1] = y_axis[1]; R[1][2] = z_axis[1]
        R[2][0] = x_axis[2]; R[2][1] = y_axis[2]; R[2][2] = z_axis[2]
        
        return R

    def get_marker_center(self, corners, depth_val):
        # C++ logic:
        # Sort X coords, Sort Y coords
        # center_x = (min_x + max_x) / 2
        # center_y = (min_y + max_y) / 2
        
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        xs.sort()
        ys.sort()
        
        cx = (xs[0] + xs[3]) / 2.0
        cy = (ys[0] + ys[3]) / 2.0
        
        # In C++, it gets depth at (cy, cx) (implied integer casting)
        # We assume constant depth here
        
        return [cx, cy, float(depth_val)]

    def make_Transform(self, data):
        # data: x, y, z, r, p, y (degrees)
        roll = data[3] * math.pi / 180
        pitch = data[4] * math.pi / 180
        yaw = data[5] * math.pi / 180
        
        m = np.zeros((4,4))
        
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)

        m[0, 0] = cy*cp
        m[0, 1] = sr*sp*cy - cr*sy
        m[0, 2] = cp*sr*cy + sr*sy # WAIT check C++ code again
        # C++: cos(yaw) * sin(pitch)*cos(roll) + sin(yaw)*sin(roll)
        # My transcription above: cp*cr*cy ?? No.
        # R02 = cy*sp*cr + sy*sr.
        
        # Let's verify carefully against C++ snippet provided
        # coordinate.at<float>(0, 2) = cos(yaw) * sin(pitch)*cos(roll) + sin(yaw)*sin(roll);
        m[0, 2] = cy * sp * math.cos(roll) + sy * sr
        
        m[0, 3] = data[0]*1000
        
        # coordinate.at<float>(1, 0) = sin(yaw) * cos(pitch);
        m[1, 0] = sy * cp
        
        # coordinate.at<float>(1, 1) = sin(roll) * sin(pitch)*sin(yaw) + cos(roll)*cos(yaw);
        m[1, 1] = sr * sp * sy + math.cos(roll) * cy
        
        # coordinate.at<float>(1, 2) = sin(yaw) * sin(pitch)*cos(roll) - cos(yaw)*sin(roll);
        m[1, 2] = sy * sp * math.cos(roll) - cy * sr
        
        m[1, 3] = data[1]*1000
        
        # coordinate.at<float>(2, 0) = -sin(pitch);
        m[2, 0] = -sp
        
        # coordinate.at<float>(2, 1) = cos(pitch) * sin(roll);
        m[2, 1] = cp * sr
        
        # coordinate.at<float>(2, 2) = cos(pitch) * cos(roll);
        m[2, 2] = cp * math.cos(roll)
        
        m[2, 3] = data[2]*1000
        
        m[3, 3] = 1.0
        
        return m

# ==========================================
# Python Logic (from main.py)
# ==========================================

class Python_Logic:
    def __init__(self):
        self.principal_point = [320.0, 240.0]
        self.focal_length = 600.0

    def convert_pixel2mm(self, center):
        if not center: return center
        if self.focal_length == 0: return center
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
        
    def get_rotation_matrix(self, corners, depth_val):
        pts_3d = []
        # corners simulates list of [x, y]
        for i in range(4):
            cx, cy = corners[i]
            # Python uses depth directly (simulated)
            pts_3d.append(self.convert_pixel2mm([cx, cy, float(depth_val)]))
            
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
        
        y_axis = self.cross(z_axis, x_axis)
        y_axis = self.normalize(y_axis)
        
        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = x_axis[0]; R[0][1] = y_axis[0]; R[0][2] = z_axis[0]
        R[1][0] = x_axis[1]; R[1][1] = y_axis[1]; R[1][2] = z_axis[1]
        R[2][0] = x_axis[2]; R[2][1] = y_axis[2]; R[2][2] = z_axis[2]
        
        return R

    def make_transform(self, data):
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

# ==========================================
# Runs
# ==========================================

def run_comparison():
    print("Running Verification...")
    
    # 1. Inputs
    # -------------------------------------
    # Simulated Marker Corners (TL, TR, BR, BL) in pixels
    # Slightly rotated to test axis logic
    # Assume 100px square centered at 320, 240
    corners = [
        [270, 190], # TL
        [370, 190], # TR
        [370, 290], # BR
        [270, 290]  # BL
    ]
    depth_val = 1000.0 # 1 meter (1000mm)
    
    # Transforms
    base_to_marker_data = [0.2, 0.0, 1.0, 180.0, 0.0, -90.0]
    camera_to_tool_data = [0.0, 0.0, -0.1, 0.0, 180.0, 90.0]
    
    # 2. C++ Calculation
    # -------------------------------------
    cpp = CPP_Logic()
    # Poses
    base_to_marker_tf_cpp = cpp.make_Transform(base_to_marker_data)
    camera_to_tool_tf_cpp = cpp.make_Transform(camera_to_tool_data)
    
    # Detection
    R_cpp = cpp.get_rotation_matrix(corners, depth_val)
    center_cpp_pix = cpp.get_marker_center(corners, depth_val) # [cx, cy, depth]
    center_cpp_mm = cpp.convert_pixel2mm_vec3(center_cpp_pix)
    
    # Construct Matrix (Row Major)
    transform_cpp_list = [
        R_cpp[0][0], R_cpp[0][1], R_cpp[0][2], center_cpp_mm[0],
        R_cpp[1][0], R_cpp[1][1], R_cpp[1][2], center_cpp_mm[1],
        R_cpp[2][0], R_cpp[2][1], R_cpp[2][2], center_cpp_mm[2],
        0, 0, 0, 1
    ]
    camera_to_marker_tf_cpp = np.array(transform_cpp_list).reshape(4,4)
    # C++ Inverse (using numpy to simulate cv::inv)
    camera_to_marker_inv_cpp = np.linalg.inv(camera_to_marker_tf_cpp)
    
    # Final Chain
    # base_to_tool = base_to_marker * camera_to_marker.inv() * camera_to_tool
    base_to_tool_tf_cpp = base_to_marker_tf_cpp @ camera_to_marker_inv_cpp @ camera_to_tool_tf_cpp
    
    
    # 3. Python Calculation
    # -------------------------------------
    py = Python_Logic()
    base_to_marker_tf_py = py.make_transform(base_to_marker_data)
    camera_to_tool_tf_py = py.make_transform(camera_to_tool_data)
    
    R_py = py.get_rotation_matrix(corners, depth_val)
    
    # Python Centroid Logic
    cx = sum([c[0] for c in corners])/4.0
    cy = sum([c[1] for c in corners])/4.0
    center_py_pix = [cx, cy, depth_val]
    center_py_mm = py.convert_pixel2mm(center_py_pix)
    
    transform_py_list = [
        R_py[0][0], R_py[0][1], R_py[0][2], center_py_mm[0],
        R_py[1][0], R_py[1][1], R_py[1][2], center_py_mm[1],
        R_py[2][0], R_py[2][1], R_py[2][2], center_py_mm[2],
        0, 0, 0, 1
    ]
    camera_to_marker_tf_py = np.array(transform_py_list).reshape(4,4)
    camera_to_marker_inv_py = np.linalg.inv(camera_to_marker_tf_py)
    
    base_to_tool_tf_py = base_to_marker_tf_py @ camera_to_marker_inv_py @ camera_to_tool_tf_py
    
    
    # 4. Compare
    # -------------------------------------
    print("--- Comparison ---")
    print(f"Base To Marker TF Match: {np.allclose(base_to_marker_tf_cpp, base_to_marker_tf_py)}")
    print(f"Camera To Tool TF Match: {np.allclose(camera_to_tool_tf_cpp, camera_to_tool_tf_py)}")
    
    print("\n[Rotation Matrix]")
    print(f"Match: {np.allclose(np.array(R_cpp), np.array(R_py))}")
    if not np.allclose(np.array(R_cpp), np.array(R_py)):
        print("Diff:")
        print(np.array(R_cpp) - np.array(R_py))

    print("\n[Marker Center MM]")
    print(f"CPP: {center_cpp_mm}")
    print(f"PY:  {center_py_mm}")
    print(f"Match: {np.allclose(center_cpp_mm, center_py_mm)}")
    
    print("\n[Camera To Marker TF]")
    print(f"Match: {np.allclose(camera_to_marker_tf_cpp, camera_to_marker_tf_py)}")
    
    print("\n[Final Result: Base To Tool TF]")
    print("CPP Result:")
    print(base_to_tool_tf_cpp)
    print("PY Result:")
    print(base_to_tool_tf_py)
    print(f"Match: {np.allclose(base_to_tool_tf_cpp, base_to_tool_tf_py)}")

if __name__ == "__main__":
    run_comparison()
