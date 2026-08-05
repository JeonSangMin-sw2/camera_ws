import sys
import numpy as np
import rby1_sdk as rby

def main():
    ip = "127.0.0.1:50051"
    for model_name in ["a", "m"]:
        print(f"Trying to connect with model '{model_name}' to {ip}...")
        try:
            robot = rby.create_robot(ip, model_name)
            if robot.connect():
                print(f"Connected successfully with model {model_name}!")
                dyn = robot.get_dynamics()
                model = robot.model()
                state = dyn.make_state(["ee_right", "ee_left"], model.robot_joint_names)
                q_lower = np.array(dyn.get_limit_q_lower(state))
                q_upper = np.array(dyn.get_limit_q_upper(state))
                
                print("\nRight Arm Joint Limits:")
                for i, joint_idx in enumerate(model.right_arm_idx):
                    low = np.degrees(q_lower[joint_idx])
                    upp = np.degrees(q_upper[joint_idx])
                    print(f"  Joint {i} ({model.robot_joint_names[joint_idx]}): [{low:.1f}, {upp:.1f}]")
                    
                print("\nLeft Arm Joint Limits:")
                for i, joint_idx in enumerate(model.left_arm_idx):
                    low = np.degrees(q_lower[joint_idx])
                    upp = np.degrees(q_upper[joint_idx])
                    print(f"  Joint {i} ({model.robot_joint_names[joint_idx]}): [{low:.1f}, {upp:.1f}]")
                    
                robot.disconnect()
                return
        except Exception as e:
            print(f"Failed with model {model_name}: {e}")

if __name__ == "__main__":
    main()
