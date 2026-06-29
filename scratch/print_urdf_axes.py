import xml.etree.ElementTree as ET
import os

urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model.urdf"
if not os.path.exists(urdf_path):
    print("URDF not found at", urdf_path)
else:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall('joint'):
        name = joint.get('name')
        axis = joint.find('axis')
        axis_str = axis.get('xyz') if axis is not None else "None"
        if "arm" in name:
            print(f"Joint: {name:30} | Axis: {axis_str}")
