import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.calibration_optimizer import QPCalibrationOptimizer
from core.calibration_core import get_both_arm_config, get_head_config, generate_sim_measurements
import rby1_sdk as rby

class MockRobot:
    def __init__(self):
        self._real_robot = rby.create_robot("127.0.0.1", "m")
        self.joint_offsets = {"right": {}, "left": {}}
    def model(self):
        return self._real_robot.model()
    def get_dynamics(self):
        return self._real_robot.get_dynamics()
    def get_state(self):
        class State:
            position = np.zeros(20)
        return State()

mock_robot = MockRobot()
dyn_model = mock_robot.get_dynamics()
model = mock_robot.model()

active_arms = ["right", "left"]
cfg = get_both_arm_config(model, version="1.2")
head_cfg = get_head_config(model)

# Generate dummy trajectory q_arm_list
np.random.seed(42)
N = 40
q_arm_list = np.random.uniform(-0.3, 0.3, size=(N, 14))
q_head_list = np.random.uniform(-0.2, 0.2, size=(N, 2))

# Generate GT sim measurements
T_meas_list = generate_sim_measurements(
    robot=mock_robot,
    dyn_model=dyn_model,
    q_arm_list=q_arm_list,
    q_head_list=q_head_list,
    arm_idx=cfg["arm_idx"],
    head_idx=head_cfg["head_idx"],
    q_nominal=np.zeros(20),
    optimize_arm=True,
    optimize_head=True,
    optimize_camera=True,
    active_arms=active_arms,
    ee_links=cfg["ee_links"],
    mount_to_cam_nom=cfg["mount_to_cam_nom"],
    head_base_to_cam_nom=cfg["head_base_to_cam_nom"],
    ee_to_marker_nom=cfg["ee_to_marker_nom"],
    camera_position_noise_std_m=0.0,
    camera_orientation_noise_std_deg=0.0
)

# Ground-truth values in generate_sim_measurements:
# q_offset_true (14 joints): [3, 0, 1, 2, -3, 2, 1, -2, -1, 3, 2, -4, 2, -2] (in deg)
# q_head_offset_true: [2.0, -1.5] (in deg)
gt_arm_deg = np.array([3, 0, 1, 2, -3, 2, 1, -2, -1, 3, 2, -4, 2, -2])

staged_offsets = {
    "right": {"joint3": -2.0, "joint5": -2.0, "joint6": -1.0},
    "left":  {"joint3": -2.0, "joint5": -2.0, "joint6": 2.0}
}

def run_test(label, lambda_cam_rot, bound_margin_deg):
    sys.stdout.write(f"\n========================================================\n")
    sys.stdout.write(f" TEST: {label}\n")
    sys.stdout.write(f" (lambda_cam_rot={lambda_cam_rot}, bound_margin={bound_margin_deg}deg)\n")
    sys.stdout.write(f"========================================================\n")
    sys.stdout.flush()

    class CustomOptimizer(QPCalibrationOptimizer):
        def get_joint_offset_limits(self):
            q_lower, q_upper = super().get_joint_offset_limits()
            if getattr(self, 'apply_joint_offset_limits', False) and getattr(self, 'joint_offsets_to_apply', None) is not None:
                jo = self.joint_offsets_to_apply
                D2R = np.pi / 180.0
                r_j3 = jo.get("right", {}).get("joint3", 0.0)
                l_j3 = jo.get("left", {}).get("joint3", 0.0)
                q_lower[3] = min((-r_j3 - bound_margin_deg) * D2R, (-r_j3 + bound_margin_deg) * D2R)
                q_upper[3] = max((-r_j3 - bound_margin_deg) * D2R, (-r_j3 + bound_margin_deg) * D2R)

                q_lower[10] = min((-l_j3 - bound_margin_deg) * D2R, (-l_j3 + bound_margin_deg) * D2R)
                q_upper[10] = max((-l_j3 - bound_margin_deg) * D2R, (-l_j3 + bound_margin_deg) * D2R)

                r_j6 = jo.get("right", {}).get("joint6", 0.0)
                l_j6 = jo.get("left", {}).get("joint6", 0.0)
                q_lower[6] = min((-r_j6 - bound_margin_deg) * D2R, (-r_j6 + bound_margin_deg) * D2R)
                q_upper[6] = max((-r_j6 - bound_margin_deg) * D2R, (-r_j6 + bound_margin_deg) * D2R)

                q_lower[13] = min((-l_j6 - bound_margin_deg) * D2R, (-l_j6 + bound_margin_deg) * D2R)
                q_upper[13] = max((-l_j6 - bound_margin_deg) * D2R, (-l_j6 + bound_margin_deg) * D2R)
            return q_lower, q_upper

    opt3 = CustomOptimizer(
        robot=mock_robot,
        arm_idx=cfg["arm_idx"],
        ee_links=cfg["ee_links"],
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-9,
        lambda_cam_pos=1.0,
        lambda_cam_rot=lambda_cam_rot,
        optimize_head=True,
        optimize_camera=True,
        active_arms=active_arms,
        estimate_measurement_noise=False,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets
    )
    q_arm_offset, q_head_offset, xi_cam, mount_to_cam_new, _ = opt3.optimize(
        q_arm_list, q_head_list, T_meas_list
    )

    arm_deg = np.rad2deg(q_arm_offset)
    head_deg = np.rad2deg(q_head_offset)
    cam_rot_deg = np.rad2deg(xi_cam[:3])

    diff = np.abs(arm_deg - gt_arm_deg)
    sys.stdout.write(f"Right Arm Calc (deg): {np.round(arm_deg[:7], 4)}\n")
    sys.stdout.write(f"Right Arm Diff (deg): {np.round(diff[:7], 4)}\n")
    sys.stdout.write(f"Left Arm Calc (deg) : {np.round(arm_deg[7:], 4)}\n")
    sys.stdout.write(f"Left Arm Diff (deg) : {np.round(diff[7:], 4)}\n")
    sys.stdout.write(f"Head Offsets (deg)  : Pan={head_deg[0]:.4f} (diff={abs(head_deg[0]-2.0):.4f}), Tilt={head_deg[1]:.4f} (diff={abs(head_deg[1]-(-1.5)):.4f})\n")
    sys.stdout.write(f"Camera xi (rot deg) : {np.round(cam_rot_deg, 4)}\n")
    sys.stdout.write(f"Max Arm Joint Error : {np.max(diff):.4f}deg\n")
    sys.stdout.flush()

run_test("1) Current Behavior (lambda_cam_rot=1e6, bound=0.1deg)", lambda_cam_rot=1e6, bound_margin_deg=0.1)
run_test("2) Reduced Cam Penalty (lambda_cam_rot=1000.0, bound=0.1deg)", lambda_cam_rot=1000.0, bound_margin_deg=0.1)
run_test("3) Tight Bound (lambda_cam_rot=1e6, bound=0.01deg)", lambda_cam_rot=1e6, bound_margin_deg=0.01)
run_test("4) Tight Bound + Balanced Cam (lambda_cam_rot=1000.0, bound=0.01deg)", lambda_cam_rot=1000.0, bound_margin_deg=0.01)
