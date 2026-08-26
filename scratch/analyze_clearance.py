import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import rby1_sdk as rby

# Load real captured dataset
data_path = "/home/rainbow/camera_ws/result/result_step2/dataset_20260825_120413.npz"
data = np.load(data_path)
q = data['q']
q_head = data.get('q_head')

robot = rby.create_robot('127.0.0.1', 'a')
model = robot.model()
dyn = robot.get_dynamics()

links = ['ee_right', 'ee_left', 'link_right_arm_3', 'link_left_arm_3', 'link_right_arm_5', 'link_left_arm_5', 'link_torso_5']

report_path = "/home/rainbow/camera_ws/scratch/clearance_report.txt"
with open(report_path, "w") as f:
    f.write(f"Total samples: {len(q)}\n")
    f.write("--- Proximity / Clearance Analysis ---\n")
    
    close_samples = []
    for i, q_sample in enumerate(q):
        q_full = np.zeros(len(model.robot_joint_names))
        q_full[model.right_arm_idx[:7]] = q_sample[:7]
        q_full[model.left_arm_idx[:7]] = q_sample[7:14]
        
        state = dyn.make_state(links, model.robot_joint_names)
        state.set_q(q_full)
        dyn.compute_forward_kinematics(state)
        
        p_ee_r = dyn.compute_transformation(state, 0, links.index('ee_right'))[:3, 3]
        p_ee_l = dyn.compute_transformation(state, 0, links.index('ee_left'))[:3, 3]
        p_elb_r = dyn.compute_transformation(state, 0, links.index('link_right_arm_3'))[:3, 3]
        p_elb_l = dyn.compute_transformation(state, 0, links.index('link_left_arm_3'))[:3, 3]
        p_wri_r = dyn.compute_transformation(state, 0, links.index('link_right_arm_5'))[:3, 3]
        p_wri_l = dyn.compute_transformation(state, 0, links.index('link_left_arm_5'))[:3, 3]
        p_torso = dyn.compute_transformation(state, 0, links.index('link_torso_5'))[:3, 3]
        
        d_ee = np.linalg.norm(p_ee_r - p_ee_l) * 1000.0 # mm
        d_elb = np.linalg.norm(p_elb_r - p_elb_l) * 1000.0
        d_wri = np.linalg.norm(p_wri_r - p_wri_l) * 1000.0
        
        min_d = min(d_ee, d_elb, d_wri)
        if min_d < 220.0:
            close_samples.append((i+1, min_d, d_ee, d_wri, d_elb))
            f.write(f"Sample {i+1:02d}: Min Distance = {min_d:.1f} mm | End-Effectors: {d_ee:.1f} mm, Wrists: {d_wri:.1f} mm, Elbows: {d_elb:.1f} mm\n")
            
    f.write(f"\nFound {len(close_samples)} samples with clearance < 220mm\n")

print("Report generated successfully.")
