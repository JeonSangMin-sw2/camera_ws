import sys
sys.path.append('/home/rainbow/camera_ws')
from core.calibration.CalibratorBase import BaseCalibrator
class DummyRobot:
    def model(self):
        class Model:
            left_arm_idx = [0, 1, 2, 3, 4, 5, 6]
            right_arm_idx = [7, 8, 9, 10, 11, 12, 13]
        return Model()
    def get_dynamics(self):
        return None
# We don't have dynamics, wait! The log says it dynamically calculated nominal axes using FK!
