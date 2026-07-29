import sys
import os
import traceback
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

out_file = "/home/rainbow/camera_ws/scratch/output_stage1.txt"
with open(out_file, "w") as f:
    f.write("Script started.\n")

try:
    from core.calibration_optimizer import QPCalibrationOptimizer
    from core.calibration_core import get_both_arm_config, get_head_config, load_npz_dataset
    import rby1_sdk as rby

    class SingleRobotHolder:
        _instance = None
        @classmethod
        def get_robot(cls):
            if cls._instance is None:
                cls._instance = rby.create_robot("127.0.0.1", "m")
            return cls._instance

    class MockRobot:
        def __init__(self):
            self._real_robot = SingleRobotHolder.get_robot()
        def model(self):
            return self._real_robot.model()
        def get_dynamics(self):
            return self._real_robot.get_dynamics()
        def get_state(self):
            return self._real_robot.get_state()

    mock_robot = MockRobot()
    model = mock_robot.model()

    npz_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260729_192805.npz"
    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(npz_path)

    head_cfg = get_head_config(model)
    both_cfg = get_both_arm_config(model, version="1.2")

    staged_offsets = {
        "right": {"joint3": -0.4955, "joint5": -5.4697, "joint6": -2.3879},
        "left": {"joint3": -0.6914, "joint5": 2.9833, "joint6": -3.5370}
    }

    opt = QPCalibrationOptimizer(
        robot=mock_robot,
        arm_idx=both_cfg["arm_idx"],
        ee_links=both_cfg["ee_links"],
        mount_to_cam_nom=both_cfg["mount_to_cam_nom"],
        head_base_to_cam_nom=both_cfg.get("head_base_to_cam_nom"),
        ee_to_marker_nom=both_cfg["ee_to_marker_nom"],
        head_idx=head_cfg["head_idx"],
        eps=1e-6,
        lambda_cam_pos=1.0,
        lambda_cam_rot=1e6,
        use_sag=False,
        optimize_head=True,
        optimize_camera=True,
        active_arms=["right", "left"],
        estimate_measurement_noise=True,
        apply_joint_offset_limits=True,
        joint_offsets_to_apply=staged_offsets,
    )

    q_arm_offset = np.zeros(len(opt.arm_idx))
    q_head_offset = np.zeros(len(opt.head_idx))
    xi_mount_cam = np.zeros(6)

    with open(out_file, "a") as f:
        f.write("Starting manual optimize loop...\n")

    for it in range(opt.max_iter):
        with open(out_file, "a") as f:
            f.write(f"Iteration {it} compute_step started...\n")
        dx, total_err = opt.compute_step(
            q_arm_list,
            q_head_list,
            T_meas_list,
            q_arm_offset,
            q_head_offset,
            xi_mount_cam,
        )
        with open(out_file, "a") as f:
            f.write(f"Iteration {it} compute_step finished. |err|={total_err:.3e}\n")

        q_arm_offset, q_head_offset, xi_mount_cam = opt.apply_update(
            q_arm_offset,
            q_head_offset,
            xi_mount_cam,
            dx,
        )
        with open(out_file, "a") as f:
            f.write(f"Iteration {it} update applied. |dx|={np.linalg.norm(dx):.3e}\n")

        if np.linalg.norm(dx) < opt.eps:
            with open(out_file, "a") as f:
                f.write("Converged.\n")
            break

    calc_r = np.rad2deg(q_arm_offset)
    calc_l = calc_r[7:]
    calc_r = calc_r[:7]
    calc_h = np.rad2deg(q_head_offset)

    gt_r = np.array([0.5, 2.5, 1.2, 0.5, -1.5, 5.4, 2.3])
    gt_l = np.array([-0.4, -1.6, -1.0, 0.7, 1.1, -3.0, 3.5])
    gt_h = np.array([0.8, -1.6])

    with open(out_file, "a") as f:
        f.write("\n======================================\n")
        f.write("OPTIMIZATION RESULT:\n")
        f.write("======================================\n")
        f.write("[RIGHT ARM ERROR]\n")
        for i in range(7):
            f.write(f"J{i}: GT={gt_r[i]:+6.2f} | Calc={calc_r[i]:+7.4f} | Err={np.abs(calc_r[i]-gt_r[i]):6.4f}\n")

        f.write("\n[LEFT ARM ERROR]\n")
        for i in range(7):
            f.write(f"J{i}: GT={gt_l[i]:+6.2f} | Calc={calc_l[i]:+7.4f} | Err={np.abs(calc_l[i]-gt_l[i]):6.4f}\n")

        f.write("\n[HEAD ERROR]\n")
        f.write(f"Pan : GT={gt_h[0]:+6.2f} | Calc={calc_h[0]:+7.4f} | Err={np.abs(calc_h[0]-gt_h[0]):6.4f}\n")
        f.write(f"Tilt: GT={gt_h[1]:+6.2f} | Calc={calc_h[1]:+7.4f} | Err={np.abs(calc_h[1]-gt_h[1]):6.4f}\n")

except Exception as e:
    with open(out_file, "a") as f:
        f.write(f"CRASHED WITH ERROR: {e}\n")
        f.write(traceback.format_exc())
