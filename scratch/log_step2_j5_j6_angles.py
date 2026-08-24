import sys
import os
import time
import numpy as np
import rby1_sdk as rby

# Add project root to sys.path
sys.path.insert(0, '/home/rainbow/camera_ws')

from core.robot_motion import (
    AutoCollectionConfig,
    move_to_auto_ready_pose,
    build_incremental_motion_plan,
    execute_auto_motion_step,
    reset_motion_state
)

def run_step2_motion_logging():
    robot_address = "127.0.0.1:50051"
    model_name = "m"
    active_arms = ["right", "left"]
    
    print(f"[INFO] Connecting to robot/simulator at {robot_address} (model: {model_name})...")
    robot = rby.create_robot(robot_address, model_name)
    if not robot.connect():
        print(f"[ERROR] Failed to connect to robot at {robot_address}")
        return

    # Check power and servo status
    if not robot.is_power_on(".*"):
        robot.power_on(".*")
        time.sleep(0.5)
    if not robot.is_servo_on(".*"):
        robot.servo_on(".*")
        time.sleep(0.5)

    model = robot.model()
    dyn_model = robot.get_dynamics()
    
    right_arm_idx = model.right_arm_idx
    left_arm_idx = model.left_arm_idx
    
    print("[INFO] Step 1 & 2: Moving to Auto Ready Pose (v1.3)...")
    move_to_auto_ready_pose(robot, active_arms, minimum_time=5.0, priority=10, include_head_motion=True, robot_version="1.3")
    time.sleep(1.0)
    
    config = AutoCollectionConfig(
        angle_step_deg=5.0,
        position_step_m=0.03,
        step_x_m=0.03,
        max_loops=1, # 1 loop for validation
        move_time=1.5,
        settle_time=0.3,
        hold_time=0.3,
        priority=10
    )
    
    print("[INFO] Building incremental motion plan...")
    plan = build_incremental_motion_plan(robot, dyn_model, config, active_arms, include_head_motion=True)
    print(f"[INFO] Generated {len(plan)} motion steps in plan.")
    
    results = []
    
    print("\n" + "="*85)
    print(f"{'Step':<5} | {'Description':<32} | {'R_J5 (Pitch)':<12} {'R_J6 (Roll)':<12} | {'L_J5 (Pitch)':<12} {'L_J6 (Roll)':<12}")
    print("="*85)
    
    # Measure initial state
    state0 = robot.get_state()
    q0 = np.array(state0.position)
    r_j5_0 = np.degrees(q0[right_arm_idx[5]])
    r_j6_0 = np.degrees(q0[right_arm_idx[6]])
    l_j5_0 = np.degrees(q0[left_arm_idx[5]])
    l_j6_0 = np.degrees(q0[left_arm_idx[6]])
    
    print(f"{0:<5} | {'[Initial Ready Pose]':<32} | {r_j5_0:>10.2f}°  {r_j6_0:>10.2f}°  | {l_j5_0:>10.2f}°  {l_j6_0:>10.2f}°")
    results.append({
        "step": 0,
        "desc": "Initial Ready Pose",
        "right_j5_deg": r_j5_0,
        "right_j6_deg": r_j6_0,
        "left_j5_deg": l_j5_0,
        "left_j6_deg": l_j6_0
    })
    
    for idx, step_plan in enumerate(plan, start=1):
        # Execute motion
        execute_auto_motion_step(robot, config, step_plan, active_arms, include_head_motion=True)
        time.sleep(config.settle_time)
        
        # Read joint states
        state = robot.get_state()
        q = np.array(state.position)
        
        r_j5 = np.degrees(q[right_arm_idx[5]])
        r_j6 = np.degrees(q[right_arm_idx[6]])
        l_j5 = np.degrees(q[left_arm_idx[5]])
        l_j6 = np.degrees(q[left_arm_idx[6]])
        
        desc = step_plan.get("desc", f"Step {idx}")
        print(f"{idx:<5} | {desc:<32} | {r_j5:>10.2f}°  {r_j6:>10.2f}°  | {l_j5:>10.2f}°  {l_j6:>10.2f}°")
        
        results.append({
            "step": idx,
            "desc": desc,
            "right_j5_deg": r_j5,
            "right_j6_deg": r_j6,
            "left_j5_deg": l_j5,
            "left_j6_deg": l_j6
        })
        
    print("="*85)
    
    # Save to CSV
    csv_path = "/home/rainbow/camera_ws/scratch/step2_j5_j6_log.csv"
    with open(csv_path, "w") as f:
        f.write("step,description,right_j5_deg,right_j6_deg,left_j5_deg,left_j6_deg\n")
        for r in results:
            f.write(f"{r['step']},\"{r['desc']}\",{r['right_j5_deg']:.4f},{r['right_j6_deg']:.4f},{r['left_j5_deg']:.4f},{r['left_j6_deg']:.4f}\n")
            
    print(f"\n[SUCCESS] Completed Step 2 motion logging! Saved {len(results)} rows to {csv_path}")

if __name__ == "__main__":
    run_step2_motion_logging()
