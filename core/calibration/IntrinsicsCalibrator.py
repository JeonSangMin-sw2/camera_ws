import cv2
import numpy as np
import os
import glob
import yaml

class IntrinsicsCalibrator:
    class BoardPattern:
        NONE = 0
        CHESSBOARD = 1
        CHARUCOBOARD = 2

    CALIB_GUIDELINES = [
        # Center Poses
        {"name": "Center (Flat)", "pts": np.array([[0.30, 0.25], [0.70, 0.25], [0.70, 0.70], [0.30, 0.70]])},
        {"name": "Center (Tilt Up)", "pts": np.array([[0.30, 0.25], [0.70, 0.25], [0.66, 0.70], [0.34, 0.70]])},
        {"name": "Center (Tilt Down)", "pts": np.array([[0.34, 0.25], [0.66, 0.25], [0.70, 0.70], [0.30, 0.70]])},
        {"name": "Center (Tilt Left)", "pts": np.array([[0.30, 0.25], [0.70, 0.30], [0.70, 0.65], [0.30, 0.70]])},
        {"name": "Center (Tilt Right)", "pts": np.array([[0.30, 0.30], [0.70, 0.25], [0.70, 0.70], [0.30, 0.65]])},
        # Four Corners Flat
        {"name": "Top-Left Corner (Flat)", "pts": np.array([[0.05, 0.05], [0.43, 0.05], [0.43, 0.45], [0.05, 0.45]])},
        {"name": "Top-Right Corner (Flat)", "pts": np.array([[0.57, 0.05], [0.95, 0.05], [0.95, 0.45], [0.57, 0.45]])},
        {"name": "Bottom-Left Corner (Flat)", "pts": np.array([[0.05, 0.50], [0.43, 0.50], [0.43, 0.90], [0.05, 0.90]])},
        {"name": "Bottom-Right Corner (Flat)", "pts": np.array([[0.57, 0.50], [0.95, 0.50], [0.95, 0.90], [0.57, 0.90]])},
        # Four Corners Tilted
        {"name": "Top-Left Corner (Tilt Up-Left)", "pts": np.array([[0.05, 0.05], [0.43, 0.08], [0.43, 0.42], [0.05, 0.45]])},
        {"name": "Top-Right Corner (Tilt Up-Right)", "pts": np.array([[0.57, 0.08], [0.95, 0.05], [0.95, 0.45], [0.57, 0.42]])},
        {"name": "Bottom-Left Corner (Tilt Down-Left)", "pts": np.array([[0.05, 0.50], [0.43, 0.53], [0.43, 0.87], [0.05, 0.90]])},
        {"name": "Bottom-Right Corner (Tilt Down-Right)", "pts": np.array([[0.57, 0.53], [0.95, 0.50], [0.95, 0.90], [0.57, 0.87]])},
        # Edge Midpoints Flat
        {"name": "Left Edge (Flat)", "pts": np.array([[0.05, 0.25], [0.43, 0.25], [0.43, 0.70], [0.05, 0.70]])},
        {"name": "Right Edge (Flat)", "pts": np.array([[0.57, 0.25], [0.95, 0.25], [0.95, 0.70], [0.57, 0.70]])},
        {"name": "Top Edge (Flat)", "pts": np.array([[0.30, 0.05], [0.70, 0.05], [0.70, 0.50], [0.30, 0.50]])},
    ]

    def __init__(self):
        self.cameraMatrix = np.eye(3, dtype=np.float64)
        self.distCoeffs = np.zeros((5, 1), dtype=np.float64)
        self.board_size = (0, 0)
        self.pattern = self.BoardPattern.NONE
        self.square_size = 0.0
        self.marker_size = 0.0
        self.aruco_dict = None
        self.charuco_board = None
        self.b_set_board = False
        
        # Results and verification data
        self.rvecs = None
        self.tvecs = None
        self.all_obj_points = []
        self.all_img_points = []
        self.all_ids = []
        self.rms_error = 0.0
        self.std_fx = 0.0
        self.std_fy = 0.0
        self.std_cx = 0.0
        self.std_cy = 0.0
        self.test_rmse = None

    def set_board(self, width, height, pattern, square_size, marker_size, aruco_dict_name):
        self.board_size = (width, height)
        self.pattern = pattern
        self.square_size = square_size
        self.marker_size = marker_size
        
        if pattern == self.BoardPattern.CHARUCOBOARD:
            # Map string dictionary name to cv2.aruco constants
            try:
                dict_attr = getattr(cv2.aruco, aruco_dict_name)
            except AttributeError:
                dict_attr = cv2.aruco.DICT_5X5_100
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
            self.charuco_board = cv2.aruco.CharucoBoard((width, height), square_size, marker_size, self.aruco_dict)
        
        self.b_set_board = True
        return True

    def run_calibration(self, image_dir, output_yaml):
        if not self.b_set_board:
            print("Board not set!")
            return False

        image_paths = sorted(glob.glob(os.path.join(image_dir, "calib_*.png")))
        if not image_paths:
            print(f"No images found in {image_dir}")
            return False

        all_obj_points = []
        all_img_points = []
        all_ids = []
        img_size = None

        for path in image_paths:
            img = cv2.imread(path)
            if img is None: continue
            
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.pattern == self.BoardPattern.CHESSBOARD:
                ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
                if ret:
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    all_obj_points.append(objp)
                    all_img_points.append(corners)
                    all_ids.append(np.arange(len(corners)))

            elif self.pattern == self.BoardPattern.CHARUCOBOARD:
                detector = cv2.aruco.CharucoDetector(self.charuco_board)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_ids is not None and len(charuco_ids) > 4:
                    all_img_points.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    all_obj_points.append(self.charuco_board.getChessboardCorners()[charuco_ids.flatten()])

        success = self._calibrate_and_validate(all_obj_points, all_img_points, img_size, all_ids)
        if success:
            print(f"Calibration successful! RMS error: {self.rms_error:.4f}")
            if output_yaml is not None:
                self._save_results(output_yaml, img_size[0], img_size[1])
        return success

    def run_calibration_with_images(self, images, output_yaml):
        """
        Runs calibration using a list of images (numpy arrays) instead of reading from disk.
        """
        if not self.b_set_board:
            print("Board not set!")
            return False

        if not images:
            print("No images provided for calibration.")
            return False

        all_obj_points = []
        all_img_points = []
        all_ids = []
        img_size = None

        for i, img in enumerate(images):
            if img is None: continue
            
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.pattern == self.BoardPattern.CHESSBOARD:
                ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
                if ret:
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
                    objp *= self.square_size
                    all_obj_points.append(objp)
                    all_img_points.append(corners)
                    all_ids.append(np.arange(len(corners)))

            elif self.pattern == self.BoardPattern.CHARUCOBOARD:
                detector = cv2.aruco.CharucoDetector(self.charuco_board)
                charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

                if charuco_ids is not None and len(charuco_ids) > 4:
                    all_img_points.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    all_obj_points.append(self.charuco_board.getChessboardCorners()[charuco_ids.flatten()])

        success = self._calibrate_and_validate(all_obj_points, all_img_points, img_size, all_ids)
        if success:
            print(f"Calibration successful! RMS error: {self.rms_error:.4f}")
            if output_yaml is not None:
                self._save_results(output_yaml, img_size[0], img_size[1])
        return success

    def _calibrate_and_validate(self, all_obj_points, all_img_points, img_size, all_ids):
        if len(all_obj_points) < 5:
            print(f"Not enough valid frames for calibration (detected {len(all_obj_points)} valid frames, minimum 5 required).")
            return False

        try:
            cameraMatrix = np.eye(3, dtype=np.float64)
            distCoeffs = np.zeros((5, 1), dtype=np.float64)
            ret, mtx, dist, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors = cv2.calibrateCameraExtended(
                all_obj_points, all_img_points, img_size, cameraMatrix, distCoeffs
            )
            self.cameraMatrix = mtx
            self.distCoeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.all_obj_points = all_obj_points
            self.all_img_points = all_img_points
            self.all_ids = all_ids
            self.rms_error = ret
            
            std_int = stdIntrinsics.flatten()
            self.std_fx = float(std_int[0])
            self.std_fy = float(std_int[1])
            self.std_cx = float(std_int[2])
            self.std_cy = float(std_int[3])
        except Exception as e:
            # Fallback to standard calibrateCamera if extended is unsupported or fails
            try:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    all_obj_points, all_img_points, img_size, None, None
                )
                self.cameraMatrix = mtx
                self.distCoeffs = dist
                self.rvecs = rvecs
                self.tvecs = tvecs
                self.all_obj_points = all_obj_points
                self.all_img_points = all_img_points
                self.all_ids = all_ids
                self.rms_error = ret
                
                self.std_fx = 0.0
                self.std_fy = 0.0
                self.std_cx = 0.0
                self.std_cy = 0.0
            except Exception as inner_e:
                print(f"Calibration calculation failed: {inner_e}")
                return False

        # Run cross-validation
        self.test_rmse = self.compute_cross_validation_rmse(all_obj_points, all_img_points, img_size)
        return True

    def compute_cross_validation_rmse(self, all_obj_points, all_img_points, img_size):
        if len(all_obj_points) < 6:
            return None
            
        # Determinisitic split: test set is every 4th frame (indices 3, 7, etc.)
        test_indices = list(range(3, len(all_obj_points), 4))
        train_indices = [i for i in range(len(all_obj_points)) if i not in test_indices]
        
        train_obj = [all_obj_points[i] for i in train_indices]
        train_img = [all_img_points[i] for i in train_indices]
        
        test_obj = [all_obj_points[i] for i in test_indices]
        test_img = [all_img_points[i] for i in test_indices]
        
        # Calibrate on train set
        mtx_init = np.eye(3, dtype=np.float64)
        dist_init = np.zeros((5, 1), dtype=np.float64)
        try:
            ret_t, mtx_t, dist_t, rvecs_t, tvecs_t = cv2.calibrateCamera(
                train_obj, train_img, img_size, mtx_init, dist_init
            )
            
            # Evaluate on test set
            test_errors = []
            for obj_pts, img_pts in zip(test_obj, test_img):
                # Solve PnP to find test view extrinsics
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx_t, dist_t)
                if ok:
                    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx_t, dist_t)
                    err = np.linalg.norm(img_pts - proj_pts, axis=2)
                    test_errors.extend(err.flatten())
            
            if test_errors:
                return float(np.sqrt(np.mean(np.array(test_errors)**2)))
        except Exception:
            pass
        return None

    def _save_results(self, output_yaml, width, height):
        data = {
            "width": int(width),
            "height": int(height),
            "camera_matrix": self.cameraMatrix.tolist(),
            "dist_coeffs": self.distCoeffs.tolist()
        }
        with open(output_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Results saved to {output_yaml} ({width}x{height})")

    def generate_verification_image(self, test_img, save_path):
        """
        Generates an undistorted side-by-side comparison with grids and saves it.
        """
        if test_img is None:
            return False
            
        h, w = test_img.shape[:2]
        
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(test_img, self.cameraMatrix, self.distCoeffs, None, new_mtx)
        
        combined_res = np.hstack((test_img, undistorted))
        h_res, w_res = combined_res.shape[:2]

        grid_size = 60
        for y in range(0, h_res, grid_size):
            cv2.line(combined_res, (0, y), (w_res, y), (0, 255, 0), 1)
        for x in range(0, w_res, grid_size):
            cv2.line(combined_res, (x, 0), (x, h_res), (0, 255, 0), 1)

        cv2.putText(combined_res, "Original", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, "Undistorted", (w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(combined_res, f"RMS Error: {self.rms_error:.4f}", (w + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, combined_res)
        return True
