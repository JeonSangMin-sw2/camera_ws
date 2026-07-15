import sys
import numpy as np

sys.path.append('.')
from core.calibration.JointCalibrator import JointCalibrator

jc = JointCalibrator(None)
# Wait, I can't instantiate it without a robot.
