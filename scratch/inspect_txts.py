import os
import glob

txt_dir = '/home/rainbow/camera_ws/result_1_3/result_txt'

for f in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
    with open(f, 'r') as fp:
        lines = fp.readlines()
        print(f"=== {os.path.basename(f)} ({len(lines)} lines) ===")
        # Print first 5 lines and last 5 lines
        for l in lines[:3]:
            print("  ", l.strip())
        if len(lines) > 6:
            print("   ...")
            for l in lines[-3:]:
                print("  ", l.strip())
