import cv2
import numpy as np
import time
import yaml
import os
import argparse
from scipy.optimize import least_squares

from marker_detection import RealSenseCamera

class CubeCalibrator:
    def __init__(self, cube_size_mm=60.0, marker_size_mm=36.0, is_left_cube=True):
        self.cube_size_mm = cube_size_mm
        self.marker_size_mm = marker_size_mm
        self.is_left_cube = is_left_cube
        self.marker_ids = [10, 11, 12, 13, 14] if is_left_cube else [30, 31, 32, 33, 34]
        # ID 10 / 30 (+Y face) is typically used as the locked base
        self.base_marker_id = self.marker_ids[0] 
        
        self.ideal_corners = self._generate_ideal_corners()
        self.images_data = [] # List of {m_id: array(4,2)}
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01

        self.optimized_corners = None

    def _generate_ideal_corners(self):
        ideal = {}
        R_dict = {
            10: np.array([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]]),
            11: np.array([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]]),
            12: np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]),
            13: np.array([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]]),
            14: np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]])
        }
        half_c = self.cube_size_mm / 2.0
        t_dict = {
            10: np.array([0, half_c, 0]),
            11: np.array([-half_c, 0, 0]),
            12: np.array([0, 0, -half_c]),
            13: np.array([half_c, 0, 0]),
            14: np.array([0, 0, half_c])
        }
        
        half_m = self.marker_size_mm / 2.0
        local_corners = np.array([
            [-half_m, -half_m, 0],
            [ half_m, -half_m, 0],
            [ half_m,  half_m, 0],
            [-half_m,  half_m, 0]
        ])
        
        for i, m_id in enumerate(self.marker_ids):
            base_id = 10 + i
            R = R_dict[base_id]
            t = t_dict[base_id]
            global_corners = (R @ local_corners.T).T + t
            ideal[m_id] = global_corners
            
        return ideal
        
    def add_frame_data(self, corners, ids):
        if ids is None: return False
        
        frame_dict = {}
        for i in range(len(ids)):
            m_id = ids[i][0]
            if m_id in self.marker_ids:
                frame_dict[m_id] = corners[i][0]
                
        if len(frame_dict) >= 2:
            self.images_data.append(frame_dict)
            return True
        return False

    def optimize(self, camera_matrix, dist_coeffs=np.zeros(4)):
        if len(self.images_data) < 2:
            print("Not enough frames. Please capture at least 2 frames with multiple markers.")
            return False
            
        print(f"\nStarting Bundle Adjustment with {len(self.images_data)} frames...")
        
        N_frames = len(self.images_data)
        unknown_marker_ids = [m for m in self.marker_ids if m != self.base_marker_id]
        
        x0 = []
        
        # 1. Init camera poses using SolvePnP on ideal points
        for frame_dict in self.images_data:
            obj_pts = []
            img_pts = []
            for m_id, c2d in frame_dict.items():
                obj_pts.append(self.ideal_corners[m_id])
                img_pts.append(c2d)
            
            obj_pts = np.vstack(obj_pts).astype(np.float32)
            img_pts = np.vstack(img_pts).astype(np.float32)
            
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
            if success:
                x0.extend(rvec.flatten())
                x0.extend(tvec.flatten())
            else:
                x0.extend([0, 0, 0,  0, 0, 1000]) # Fallback
                
        # 2. Init unknown marker points (from ideal definitions)
        for m_id in unknown_marker_ids:
            x0.extend(self.ideal_corners[m_id].flatten())
            
        x0 = np.array(x0, dtype=np.float64)
        
        def residuals(params):
            res = []
            idx = 0
            
            cam_rvecs = []
            cam_tvecs = []
            for _ in range(N_frames):
                cam_rvecs.append(params[idx:idx+3])
                cam_tvecs.append(params[idx+3:idx+6])
                idx += 6
                
            opt_points = {self.base_marker_id: self.ideal_corners[self.base_marker_id]}
            for m_id in unknown_marker_ids:
                pts = params[idx:idx+12].reshape(4,3)
                opt_points[m_id] = pts
                idx += 12
                
            for i, frame_dict in enumerate(self.images_data):
                rvec = cam_rvecs[i]
                tvec = cam_tvecs[i]
                
                for m_id, c2d in frame_dict.items():
                    pts_3d = opt_points[m_id]
                    proj_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
                    proj_2d = proj_2d.reshape(4, 2)
                    res.extend((proj_2d - c2d).flatten())
                    
            return np.array(res)

        print("Running Least Squares Optimization...")
        result = least_squares(residuals, x0, loss='huber', f_scale=1.0, verbose=2, 
                               ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=10000)
        print("Optimization completed.")
        
        self.optimized_corners = {self.base_marker_id: self.ideal_corners[self.base_marker_id]}
        idx = N_frames * 6
        for m_id in unknown_marker_ids:
            pts = result.x[idx:idx+12].reshape(4,3)
            self.optimized_corners[m_id] = pts
            idx += 12
            
        initial_res = residuals(x0)
        final_res = residuals(result.x)
        print(f"Initial Mean Reprojection Error: {np.mean(np.abs(initial_res)):.3f} pixels")
        print(f"Final Mean Reprojection Error:   {np.mean(np.abs(final_res)):.3f} pixels")
        
        return True

    def save_calibration(self, filename):
        if not self.optimized_corners:
            print("No optimized corners to save.")
            return
            
        out_dict = {}
        for k, v in self.optimized_corners.items():
            out_dict[int(k)] = v.tolist()
            
        with open(filename, 'w') as f:
            yaml.dump(out_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Calibrated 3D map saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"], help="Calibrate left or right arm cube")
    args = parser.parse_args()
    
    is_left = (args.side == "left")
    
    config_path = os.path.join(os.path.dirname(__file__), "config", "markers.yaml")
    try:
        with open(config_path, "r") as f:
            markers_config = yaml.safe_load(f)
            cube_size = markers_config.get("cube", {}).get("cube_size_mm", 60.0)
    except Exception as e:
        print(f"Failed to load config, defaulting to 60.0: {e}")
        cube_size = 60.0
        
    marker_size = cube_size * 0.8
    calibrator = CubeCalibrator(cube_size_mm=cube_size, marker_size_mm=marker_size, is_left_cube=is_left)
    
    print("Connecting to RealSense Camera...")
    cam = RealSenseCamera()
    cam.initialize_camera(1280, 720, 30)
    
    intrinsics = cam.get_principal_point_and_focal_length()
    camera_matrix = np.array([
        [intrinsics[2], 0, intrinsics[0]],
        [0, intrinsics[3], intrinsics[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4)
    
    print("\n==================================")
    print("      Phase 1 Cube Calibration    ")
    print("==================================")
    print(f"Targeting {'LEFT' if is_left else 'RIGHT'} Cube (IDs: {calibrator.marker_ids})")
    print("\nInstructions:")
    print("  1) Show the cube to the camera.")
    print("  2) Press 'SPACE' or 'c' to capture a static frame.")
    print("     (At least 2 target markers must be visible simultaneously)")
    print("  3) Rotate the cube / move camera and repeat step 2 (approx. 15~30 frames).")
    print("  4) Press 'ENTER' to compute Bundle Adjustment.")
    print("  5) Press 'ESC' or 'q' to quit.")
    print("----------------------------------\n")
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            cam.capture_image()
            img = cam.get_color_image()
            
            if img is None:
                time.sleep(0.01)
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, calibrator.dictionary, parameters=calibrator.parameters)
            
            display_img = img.copy()
            if ids is not None:
                # Filter only target marker IDs for display
                target_corners = []
                target_ids = []
                for i in range(len(ids)):
                    if ids[i][0] in calibrator.marker_ids:
                        target_corners.append(corners[i])
                        target_ids.append(ids[i])
                
                if len(target_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display_img, target_corners, np.array(target_ids))
                
            cv2.putText(display_img, f"Captured Frames: {len(calibrator.images_data)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            cv2.imshow("Calibration", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 or key == ord('c'): # Space or 'c'
                if calibrator.add_frame_data(corners, ids):
                    print(f"[{len(calibrator.images_data)}] Frame added successfully.")
                else:
                    print("Frame rejected (needs at least 2 target cube markers visible).")
                    
            elif key == 13 or key == 10: # Enter
                if calibrator.optimize(camera_matrix, dist_coeffs):
                    out_name = os.path.join("config", "calibrated_cube_right.yaml" if args.right else "calibrated_cube_left.yaml")
                    calibrator.save_calibration(out_name)
                    print("Calibration Done. You can press 'ESC' to exit.")
                    
            elif key == 27 or key == ord('q'): # ESC or q
                print("Exiting...")
                break
                
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        cv2.destroyAllWindows()
