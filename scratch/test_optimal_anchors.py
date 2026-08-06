import sys
import os
import numpy as np
import rby1_sdk as rby

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.calibration_optimizer import QPCalibrationOptimizer

# Load dataset
dataset_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260805_221431.npz"
data = np.load(dataset_path)
q_arm_list = data["q_arm"]
T_meas_list = data["marker"]
q_head_list = np.zeros((q_arm_list.shape[0], 2))

ip = "127.0.0.1:50051"
robot = rby.create_robot(ip, "m")
if not robot.connect():
    print("Error: Could not connect to robot simulator")
    sys.exit(1)

try:
    model = robot.model()
    arm_idx = np.concatenate([model.right_arm_idx[:7], model.left_arm_idx[:7]])
    ee_links = {"right": "ee_right", "left": "ee_left"}
    
    # Bracket values from Step 1
    ee_to_marker = {
        "left": [0.0, 0.05413, -0.00273, 91.71, -0.35, 0.0],
        "right": [0.0, -0.05416, -0.00237, 91.83, 0.0, 180.0]
    }
    
    head_base_to_cam_nom = [0.102, 0.009, 0.044, -90.0, 0.0, -90.0]
    
    joint_offsets = {
        "right": {"joint3": -2.7265, "joint5": -0.9283, "joint6": -0.2649},
        "left": {"joint3": -2.1863, "joint5": 0.0, "joint6": 0.0}
    }

    # Custom QPOptimizer that implements the custom shoulder anchor!
    class QPOptimizerWithShoulderAnchors(QPCalibrationOptimizer):
        def compute_step(self, q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam):
            # Call evaluate/build normal equation internally
            # We copy and paste the logic from the parent compute_step here:
            dim = self.total_dim()
            H = np.zeros((dim, dim), dtype=np.float64)
            g = np.zeros(dim, dtype=np.float64)
            residual_samples = []
            
            # (We simplify the evaluation for testing)
            # Actually, to make it 100% correct, we can just temporarily patch the H and g calculation!
            # Let's run a test where we patch super().compute_step's H and g!
            return super().compute_step(q_arm_list, q_head_list, T_meas_list, q_arm_offset, q_head_offset, xi_mount_cam)

    # Let's write the actual script that runs the modified calibration_optimizer.py code!
    # Wait, we can modify core/calibration_optimizer.py first, and then run scratch/test_optimal_anchors.py!
    
finally:
    robot.disconnect()
