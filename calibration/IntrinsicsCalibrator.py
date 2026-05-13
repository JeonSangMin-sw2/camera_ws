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

        if len(all_obj_points) < 5:
            print("Not enough valid frames for calibration (minimum 5 required).")
            return False

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, img_size, None, None)
        
        self.cameraMatrix = mtx
        self.distCoeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.all_obj_points = all_obj_points
        self.all_img_points = all_img_points
        self.all_ids = all_ids
        self.rms_error = ret

        print(f"Calibration successful! RMS error: {ret:.4f}")
        
        if output_yaml is not None:
            self._save_results(output_yaml, img_size[0], img_size[1])
        return True

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

        if len(all_obj_points) < 5:
            print(f"Not enough valid frames for calibration (detected {len(all_obj_points)} valid frames, minimum 5 required).")
            return False

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, img_size, None, None)
        
        self.cameraMatrix = mtx
        self.distCoeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.all_obj_points = all_obj_points
        self.all_img_points = all_img_points
        self.all_ids = all_ids
        self.rms_error = ret

        print(f"Calibration successful! RMS error: {ret:.4f}")
        
        if output_yaml is not None:
            self._save_results(output_yaml, img_size[0], img_size[1])
        return True

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
