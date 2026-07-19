import xml.etree.ElementTree as ET
import os

def main():
    urdf_path = "/home/jsm/sdk/rby1-sdk/models/rby1m/urdf/model_v1.2.urdf"
    if not os.path.exists(urdf_path):
        print("URDF not found.")
        return
        
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    print("=== Joints in URDF ===")
    for joint in root.findall('joint'):
        name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        origin_str = "None"
        if origin is not None:
            xyz = origin.get('xyz', '0 0 0')
            rpy = origin.get('rpy', '0 0 0')
            origin_str = f"xyz: {xyz} | rpy: {rpy}"
            
        if "right" in name or "ee_" in name:
            print(f"Joint: {name:30} | Parent: {parent:25} | Child: {child:25} | Origin: {origin_str}")

if __name__ == "__main__":
    main()
