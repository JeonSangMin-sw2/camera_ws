import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        jtype = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        if origin is not None:
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
        else:
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
        axis_elem = joint.find('axis')
        if axis_elem is not None:
            axis = [float(x) for x in axis_elem.get('xyz', '0 0 0').split()]
        else:
            axis = [0, 0, 0]
        joints[name] = {
            'type': jtype,
            'parent': parent,
            'child': child,
            'xyz': np.array(xyz),
            'rpy': np.array(rpy),
            'axis': np.array(axis)
        }
    return joints

joints = parse_urdf('/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf')

def compute_arm_fk(arm_side, q_arm):
    # Returns T_torso5_to_link for each link and ee
    # q_arm is 7 elements (Joint 0 to 6)
    T = np.eye(4)
    transforms = {'link_torso_5': np.eye(4)}
    
    for i in range(7):
        jname = f"{arm_side}_arm_{i}"
        j = joints[jname]
        # Fixed origin transform
        R_orig = R_scipy.from_euler('xyz', j['rpy']).as_matrix()
        T_orig = np.eye(4)
        T_orig[:3, :3] = R_orig
        T_orig[:3, 3] = j['xyz']
        
        # Joint rotation
        axis = j['axis'] / np.linalg.norm(j['axis'])
        rotvec = axis * q_arm[i]
        R_j = R_scipy.from_rotvec(rotvec).as_matrix()
        T_j = np.eye(4)
        T_j[:3, :3] = R_j
        
        T = T @ T_orig @ T_j
        transforms[j['child']] = T.copy()
        
    # EE transform
    ee_jname = f"END_{arm_side}"
    if ee_jname in joints:
        j = joints[ee_jname]
        R_orig = R_scipy.from_euler('xyz', j['rpy']).as_matrix()
        T_orig = np.eye(4)
        T_orig[:3, :3] = R_orig
        T_orig[:3, 3] = j['xyz']
        T_ee = T @ T_orig
        transforms[f"ee_{arm_side}"] = T_ee
    return transforms

# Let's test at ready pose
ready_poses = {
    'right': np.radians([-55.0, -45.0, 25.0, -117.0, 0.0, 0.0, 0.0]),
    'left': np.radians([-55.0, 45.0, -25.0, -117.0, 0.0, 0.0, 0.0])
}

# Head transform to camera:
# Pan-tilt head:
# head_0: torso_5 to head_0 (pan, z-axis)
# head_1: head_0 to head_1 (tilt, y-axis)
# mount_to_cam: [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]

def compute_cam_transform(q_head=[0, 0]):
    # In model_v1.3:
    # head_0: xyz="0.088 0 0.205", axis="0 0 1" (pan)
    # head_1: xyz="0 0 0.040", axis="0 1 0" (tilt)
    T = np.eye(4)
    # head 0
    j0 = joints['head_0']
    T0_orig = np.eye(4)
    T0_orig[:3, :3] = R_scipy.from_euler('xyz', j0['rpy']).as_matrix()
    T0_orig[:3, 3] = j0['xyz']
    T0_j = np.eye(4)
    T0_j[:3, :3] = R_scipy.from_rotvec(j0['axis'] * q_head[0]).as_matrix()
    T_head0 = T0_orig @ T0_j
    
    # head 1
    j1 = joints['head_1']
    T1_orig = np.eye(4)
    T1_orig[:3, :3] = R_scipy.from_euler('xyz', j1['rpy']).as_matrix()
    T1_orig[:3, 3] = j1['xyz']
    T1_j = np.eye(4)
    T1_j[:3, :3] = R_scipy.from_rotvec(j1['axis'] * q_head[1]).as_matrix()
    T_head1 = T_head0 @ T1_orig @ T1_j
    
    # mount to cam
    mount_to_cam = [0.047, 0.009, 0.057, -90.0, 0.0, -90.0]
    R_mc = R_scipy.from_euler('ZYX', [mount_to_cam[5], mount_to_cam[4], mount_to_cam[3]], degrees=True).as_matrix()
    T_mc = np.eye(4)
    T_mc[:3, :3] = R_mc
    T_mc[:3, 3] = mount_to_cam[:3]
    
    T_torso_to_cam = T_head1 @ T_mc
    return T_torso_to_cam

for arm in ['right', 'left']:
    print(f"\n=================== {arm.upper()} ARM FK AT READY POSE ===================")
    q = ready_poses[arm]
    tf = compute_arm_fk(arm, q)
    T_cam = compute_cam_transform([0, 0])
    T_cam_to_torso = np.linalg.inv(T_cam)
    
    # Check joint axes in Torso and Camera frame
    # J4 (Yaw): axis is Z of link 3
    # J5 (Pitch): axis is Y of link 4
    # J6 (Roll): axis is X of link 5
    T_l3 = tf[f"link_{arm}_arm_3"]
    T_l4 = tf[f"link_{arm}_arm_4"]
    T_l5 = tf[f"link_{arm}_arm_5"]
    T_l6 = tf[f"link_{arm}_arm_6"]
    T_ee = tf[f"ee_{arm}"]
    
    axis_4_t5 = T_l3[:3, :3] @ np.array([0, 0, 1])
    axis_5_t5 = T_l4[:3, :3] @ np.array([0, 1, 0])
    axis_6_t5 = T_l5[:3, :3] @ np.array([1, 0, 0])
    
    axis_4_cam = T_cam_to_torso[:3, :3] @ axis_4_t5
    axis_5_cam = T_cam_to_torso[:3, :3] @ axis_5_t5
    axis_6_cam = T_cam_to_torso[:3, :3] @ axis_6_t5
    
    print(f"J4 axis in torso: {axis_4_t5.round(4)}, in cam: {axis_4_cam.round(4)}")
    print(f"J5 axis in torso: {axis_5_t5.round(4)}, in cam: {axis_5_cam.round(4)}")
    print(f"J6 axis in torso: {axis_6_t5.round(4)}, in cam: {axis_6_cam.round(4)}")
    
    ang_45 = np.degrees(np.arccos(np.clip(abs(np.dot(axis_4_t5, axis_5_t5)), -1, 1)))
    ang_56 = np.degrees(np.arccos(np.clip(abs(np.dot(axis_5_t5, axis_6_t5)), -1, 1)))
    ang_46 = np.degrees(np.arccos(np.clip(abs(np.dot(axis_4_t5, axis_6_t5)), -1, 1)))
    print(f"Angle between J4 and J5: {ang_45:.4f}°")
    print(f"Angle between J5 and J6: {ang_56:.4f}°")
    print(f"Angle between J4 and J6: {ang_46:.4f}°")
    
    # Marker position in EE frame
    # NOMINAL_BRACKET_TEMPLATES['1.3']: [0.067, 0.0, 0.0, 90.0, 0.0, -90.0]
    p_m_ee = np.array([0.067, 0.0, 0.0, 1.0])
    p_m_t5 = T_ee @ p_m_ee
    p_m_cam = T_cam_to_torso @ p_m_t5
    print(f"Nominal marker pos in torso: {(p_m_t5[:3]*1000).round(1)} mm")
    print(f"Nominal marker pos in cam:   {(p_m_cam[:3]*1000).round(1)} mm")
    
    # Radius of rotation around each axis:
    # J6 (Roll): distance from J6 axis (origin of link 5, direction axis_6_t5) to marker
    p_l5_t5 = T_l5[:3, 3]
    v6 = p_m_t5[:3] - p_l5_t5
    r6 = np.linalg.norm(v6 - np.dot(v6, axis_6_t5) * axis_6_t5)
    
    # J5 (Pitch): distance from J5 axis (origin of link 4, direction axis_5_t5) to marker
    p_l4_t5 = T_l4[:3, 3]
    v5 = p_m_t5[:3] - p_l4_t5
    r5 = np.linalg.norm(v5 - np.dot(v5, axis_5_t5) * axis_5_t5)
    
    # J4 (Yaw): distance from J4 axis (origin of link 3, direction axis_4_t5) to marker
    p_l3_t5 = T_l3[:3, 3]
    v4 = p_m_t5[:3] - p_l3_t5
    r4 = np.linalg.norm(v4 - np.dot(v4, axis_4_t5) * axis_4_t5)
    
    print(f"Theoretical sweep radius around J6: {r6*1000:.2f} mm")
    print(f"Theoretical sweep radius around J5: {r5*1000:.2f} mm")
    print(f"Theoretical sweep radius around J4: {r4*1000:.2f} mm")
