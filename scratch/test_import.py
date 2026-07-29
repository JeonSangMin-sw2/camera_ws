import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

out_file = "/home/rainbow/camera_ws/scratch/output_import.txt"
with open(out_file, "w") as f:
    f.write("Import script started.\n")

try:
    with open(out_file, "a") as f:
        f.write("Importing numpy...\n")
    import numpy as np
    
    with open(out_file, "a") as f:
        f.write("Importing rby1_sdk...\n")
    import rby1_sdk as rby
    
    with open(out_file, "a") as f:
        f.write("Importing calibration_optimizer...\n")
    from core.calibration_optimizer import QPCalibrationOptimizer
    
    with open(out_file, "a") as f:
        f.write("Importing calibration_core...\n")
    from core.calibration_core import get_both_arm_config, get_head_config, load_npz_dataset

    with open(out_file, "a") as f:
        f.write("All imports successful!\n")

except Exception as e:
    with open(out_file, "a") as f:
        f.write(f"Crashed with: {e}\n")
        f.write(traceback.format_exc())
