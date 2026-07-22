import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration_core import load_npz_dataset

def main():
    dataset_path = "/home/jsm/camera_ws/result/result_step2/dataset_20260722_175935.npz"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(dataset_path)
    print(f"Successfully loaded dataset with {len(q_arm_list)} poses.")

if __name__ == "__main__":
    main()
