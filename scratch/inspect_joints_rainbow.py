import xml.etree.ElementTree as ET
import os

version = "v1.0"
urdf_path = f"/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_{version}.urdf"
if not os.path.exists(urdf_path):
    print("URDF not found:", urdf_path)
else:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        if "cam" in child or "cam" in parent or "camera" in child or "camera" in parent:
            print(f"Joint: {joint.get('name')} | Parent: {parent} | Child: {child}")
            origin = joint.find('origin')
            if origin is not None:
                print(f"  XYZ: {origin.get('xyz')} | RPY: {origin.get('rpy')}")
