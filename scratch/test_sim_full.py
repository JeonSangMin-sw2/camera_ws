import sys
from PyQt5.QtWidgets import QApplication
from main_ui import CalibrationApp
import time
import threading

def run_test():
    app = QApplication(sys.argv)
    window = CalibrationApp()
    
    # Enable mock run
    window.marker_tab.mock_run_checkbox.setChecked(True)
    
    def test_thread():
        # Wait for GUI to load
        time.sleep(1)
        # Start Full Auto Calibration
        print("Starting Full Auto Calibration...")
        window.full_auto_btn.click()
        
        # Wait for completion (check every 2 seconds)
        for _ in range(30):
            time.sleep(2)
            if not window.full_auto_worker or not window.full_auto_worker.isRunning():
                break
        
        print("\n--- Final Results ---")
        print("Right Arm J6 Offset:", window.joint_offsets_store["right"]["joint6"])
        print("Left Arm J6 Offset:", window.joint_offsets_store["left"]["joint6"])
        
        # We expect right J6 to be around +2.30 and left J6 to be around +3.50
        app.quit()
        
    t = threading.Thread(target=test_thread)
    t.start()
    
    app.exec_()

if __name__ == "__main__":
    run_test()
