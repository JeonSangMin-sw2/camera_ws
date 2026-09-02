import os
import sys
import shutil

if getattr(sys, 'frozen', False):
    # PyInstaller 실행 파일 위치
    current_dir = os.path.dirname(sys.executable)
else:
    # 소스 코드 위치
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 설정 파일 자동 복사 기능 (실행 파일 기준 경로에 config가 없을 시 자동 생성)
if getattr(sys, 'frozen', False):
    ext_config_dir = os.path.abspath(os.path.join(current_dir, "config"))
    os.makedirs(ext_config_dir, exist_ok=True)
    bundled_config_dir = os.path.join(sys._MEIPASS, "config")
    
    if os.path.exists(bundled_config_dir):
        for filename in os.listdir(bundled_config_dir):
            bundled_file = os.path.join(bundled_config_dir, filename)
            ext_file = os.path.join(ext_config_dir, filename)
            if os.path.isfile(bundled_file) and not os.path.exists(ext_file):
                try:
                    shutil.copy2(bundled_file, ext_file)
                    print(f"[Paths] Copied default template {filename} to {ext_file}")
                except Exception as e:
                    print(f"[Paths] Failed to copy default template {filename}: {e}")

CONFIG_PATHS = {
    "setting_yaml": os.path.abspath(os.path.join(current_dir, "config", "setting.yaml")),
    "camera_info": os.path.abspath(os.path.join(current_dir, "config", "camera_info.yaml")),
    "ready_poses_yaml": os.path.abspath(os.path.join(current_dir, "config", "ready_poses.yaml")),
    "camera_intrinsics": os.path.abspath(os.path.join(current_dir, "config", "camera_intrinsics.yaml")),
    "result_dir": os.path.abspath(os.path.join(current_dir, "result", "result_step2")),
    "plot_dir": os.path.abspath(os.path.join(current_dir, "result", "result_img")),
    "txt_dir": os.path.abspath(os.path.join(current_dir, "result", "result_txt")),
}
