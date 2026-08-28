import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def parse_urdf_transforms(urdf_path):
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

urdf_v13 = '/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf'
joints = parse_urdf_transforms(urdf_v13)

for arm in ['right', 'left']:
    print(f"=== {arm.upper()} ARM JOINTS ===")
    for j_idx in range(7):
        jname = f"{arm}_arm_{j_idx}"
        if jname in joints:
            j = joints[jname]
            print(f"  {jname}: parent={j['parent']}, child={j['child']}, xyz={j['xyz']}, rpy={j['rpy']}, axis={j['axis']}")
