import numpy as np
import os
import sys

sys.path.insert(0, '/home/rainbow/camera_ws')
from core.calibration_optimizer import QPCalibrationOptimizer, CalibrationOptimizer
import rby1_sdk as rby
from core.calibration_core import get_both_arm_config, get_head_config

def main():
    dataset_path = '/home/rainbow/camera_ws/result/result_real/result_step2/dataset_20260716_153458.npz'
    data = np.load(dataset_path, allow_pickle=True)
    q_arm_list = data['q_arm']
    q_head_list = data['q_head'] if 'q_head' in data else None
    marker_list = data['marker']
    
    print(f"Loaded dataset: {q_arm_list.shape} samples.")
    
    try:
        # Create a mock robot using the same logic main_ui.py does when not connecting
        # Actually, let's look at how QPCalibrationOptimizer uses robot.
        # It uses robot.get_state().position, and dyn_model.
        # To avoid connecting via grpc to the real robot (which might fail or interfere), 
        # let's look at the class definition of QPCalibrationOptimizer.
        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
