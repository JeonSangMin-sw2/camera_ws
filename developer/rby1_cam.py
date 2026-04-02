from marker_detection import Marker_Transform, File_Logger
import math
import time
import numpy as np

# 디버깅용 유틸리티: 회전행렬에서 RPY(Roll, Pitch, Yaw) 추출 (단위: degree)
def get_rpy_from_matrix(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


def main():
    # logger = File_Logger()
    marker_transform = None
    fps = 1/30
    check_marker_side = "left"
    try:
        marker_transform = Marker_Transform(serial_number= None, monitoring = False)
        marker_transform.marker_detection.set_marker_type("cube")
        while True:
            raw_results = marker_transform.get_marker_transform(sampling_time=2, side=check_marker_side)
            if raw_results is None or len(raw_results) == 0:
                continue
            
            temp = marker_transform.current_smoothed_temp
            if temp is not None: temp = round(temp, 4)
            # 출력용
            print(f"Camera Temperature: {temp}")
            for raw_result in raw_results:
                result = [round(n, 4) for n in raw_result]
                print(result[0],result[1],result[2],result[3])
                print(result[4],result[5],result[6],result[7])
                print(result[8],result[9],result[10],result[11])
                print(result[12],result[13],result[14],result[15])

                # 디버깅용: RPY 출력
                R = np.array([
                    [result[0], result[1], result[2]],
                    [result[4], result[5], result[6]],
                    [result[8], result[9], result[10]]
                ])
                rpy = get_rpy_from_matrix(R)
                string = f"{check_marker_side},{temp},{result[3]*1000},{result[7]*1000},{result[11]*1000},{rpy[0]},{rpy[1]},{rpy[2]}"
                # logger.save(string)
                print(string)
            time.sleep(fps) # Removed sleep for better responsiveness
            
    except RuntimeError as e:
        print(f"Initialization Error: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        if marker_transform is not None:
            marker_transform.camera.monitoring(Flag=False)
            print("Camera Stopped.")

if __name__ == "__main__":
    main()
