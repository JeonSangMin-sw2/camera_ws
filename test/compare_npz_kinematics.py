import argparse
import numpy as np
import os
import sys

# Add the parent directory to PYTHONPATH so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration_core import create_robot, load_npz_dataset, get_arm_config, get_head_config
from core.calibration_optimizer import QPCalibrationOptimizer, rot_to_euler_zyx

def print_pose(name, T):
    trans = T[:3, 3]
    euler = np.rad2deg(rot_to_euler_zyx(T[:3, :3]))
    print(f"  {name:10s}: Pos(x={trans[0]:.4f}, y={trans[1]:.4f}, z={trans[2]:.4f}) [m]  |  Rot(r={euler[0]:.2f}, p={euler[1]:.2f}, y={euler[2]:.2f}) [deg]")

def main():
    parser = argparse.ArgumentParser(description="Compare NPZ data with Robot Kinematics")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    parser.add_argument("--npz", type=str, required=True, help="Path to NPZ dataset")
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"], help="Arm to check")
    parser.add_argument("--head", action="store_true", help="Use head kinematics (cal_with_head=True)")
    args = parser.parse_args()

    print(f"Connecting to robot at {args.ip}...")
    try:
        robot = create_robot(args.ip)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print(f"Loading dataset from {args.npz}...")
    q_arm_list, q_head_list, T_meas_list = load_npz_dataset(args.npz)
    
    # Auto-slice if the dataset has both arms but we only want one
    if q_arm_list.shape[1] == 14:
        if args.arm == "right":
            q_arm_list = q_arm_list[:, :7]
        else:
            q_arm_list = q_arm_list[:, 7:]
            
    if T_meas_list.ndim == 4 and T_meas_list.shape[1] == 2:
        if args.arm == "right":
            T_meas_list = T_meas_list[:, 0]
        else:
            T_meas_list = T_meas_list[:, 1]

    model = robot.model()
    cfg = get_arm_config(model, args.arm)
    head_cfg = get_head_config(model)

    # Initialize optimizer just to use its evaluate_sample logic
    optimizer = QPCalibrationOptimizer(
        robot=robot,
        arm_idx=cfg["arm_idx"],
        ee_links={args.arm: cfg["ee_link"]},
        mount_to_cam_nom=cfg["mount_to_cam_nom"],
        t5_to_cam_nom=cfg.get("t5_to_cam_nom"),
        ee_to_marker_nom={args.arm: cfg["ee_to_marker_nom"]},
        ndof=7,
        head_idx=head_cfg["head_idx"],
        optimize_head=args.head,
        active_arms=[args.arm]
    )

    print("\n" + "="*80)
    print(f"Comparing kinematics for {args.arm.upper()} arm (Optimize Head: {args.head})")
    print(f"Base Link used: {optimizer.base_link}")
    print("="*80)

    # We will check the first 5 samples
    num_samples_to_check = min(5, len(q_arm_list))
    
    q_arm_off = np.zeros(7)
    q_head_off = np.zeros(2)
    xi_cam = np.zeros(6)

    for i in range(num_samples_to_check):
        q_arm = q_arm_list[i]
        q_head = q_head_list[i] if q_head_list is not None else None
        T_meas = T_meas_list[i]

        _, _, _, T_model = optimizer.evaluate_sample(
            q_arm=q_arm,
            q_head=q_head,
            arm_side=args.arm,
            q_arm_offset=q_arm_off,
            q_head_offset=q_head_off,
            xi_mount_cam=xi_cam
        )

        print(f"\n[Sample {i}]")
        print_pose("T_model", T_model)
        print_pose("T_meas", T_meas)
        
        # Calculate Difference
        diff_trans = T_model[:3, 3] - T_meas[:3, 3]
        R_err = T_model[:3, :3].T @ T_meas[:3, :3]
        diff_euler = np.rad2deg(rot_to_euler_zyx(R_err))
        
        print(f"  Difference: Pos(x={diff_trans[0]:.4f}, y={diff_trans[1]:.4f}, z={diff_trans[2]:.4f})      |  Rot(r={diff_euler[0]:.2f}, p={diff_euler[1]:.2f}, y={diff_euler[2]:.2f})")

if __name__ == "__main__":
    main()
