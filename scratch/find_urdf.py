import os

def find_urdfs(start_dir):
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.urdf'):
                print(os.path.join(root, file))

print("Searching for URDF files under /home/jsm...")
find_urdfs("/home/jsm")
print("Finished search.")
