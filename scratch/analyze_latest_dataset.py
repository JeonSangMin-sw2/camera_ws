import numpy as np
import yaml
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config

class DummyModel:
    def __init__(self):
        # We need a dummy model with left_arm_idx and right_arm_idx
        # Let's import it from rby1_sdk or use a real robot if we can initialize it,
        # but wait! We can just initialize the SDK robot in dummy/sim mode.
        pass

def main():
    dataset_path = "result/result_step2/dataset_20260812_021103.npz"
    baseline_path = "config/home_reset_baseline.json"
    
    # Load dataset
    data = np.load(dataset_path)
    q_arm = data["q_arm"]
    q_head = data["q_head"]
    T_meas = data["marker"]
    
    # Load baseline
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)
    base_r = np.array(baseline_data["right_arm_joint_offset_deg"])
    base_l = np.array(baseline_data["left_arm_joint_offset_deg"])
    
    # Let's initialize the robot using rby1_sdk
    import rby1_sdk as rby
    # Create simulated robot model (using "a" or the local urdf)
    # The codebase uses rby.create_robot(..., "a") or similar
    # Let's look at how create_robot is done in main_ui.py
    # Since we don't have connection to the real robot, we can use the local model description or a simulated robot object.
    # Wait, does rby1_sdk allow creating a model without connecting?
    # Yes, we can try to connect to localhost or use dummy.
    # Actually, let's see how main_ui.py initializes robot in simulation mode:
    # robot = rby.create_robot(ip, model_name)
    # If we run simulation, we can just use the model representation.
    # Let's write a script that connects to the SDK model:
    try:
        robot = rby.create_robot("127.0.0.1", "m") # m is the model name for v1.2/v1.3
        # Since 127.0.0.1 is not running a real robot, it will fail to connect.
        # But wait! The SDK has a model file or offline representation.
        # Let's check if we can initialize the robot in simulation or offline mode.
        # Let's check how main_ui.py does it for simulation.
    except Exception as e:
        print(f"Error creating robot: {e}")
        return

if __name__ == "__main__":
    main()
