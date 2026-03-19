from marker_detection import Marker_Transform, File_Logger
import math
import time

# 인식할 마커 타입 설정
mrker_type = "cube" # "cube" or "plate"
# 큐브 마커 id 설정
cube_id = [10,11,12,13,14]
# 플레이트 마커 id 설정
plate_id = [7]

def main():
    logger = File_Logger()
    marker_transform = None
    fps = 1/30
    try:
        marker_transform = Marker_Transform(Stereo=False, serial_number= None, monitoring = False)
        marker_transform.set_marker_type("cube", [10,11,12,13,14])
        while True:
            raw_results = marker_transform.get_marker_transform(sampling_time=0)
            if raw_results is None or len(raw_results) == 0:
                continue
            
            temp = marker_transform.camera.get_camera_temperature()
            
            for marker_id, raw_result in raw_results.items():
                # 필터링된 행렬로 RPY 재계산
                rot_matrix = [
                    [raw_result[0], raw_result[1], raw_result[2]],
                    [raw_result[4], raw_result[5], raw_result[6]],
                    [raw_result[8], raw_result[9], raw_result[10]]
                ]
                rpy = marker_transform.marker_detection.get_rpy_from_matrix(rot_matrix)
                
                # 출력용
                # print(f"--- Marker ID: {marker_id} ---")
                # print(f"Camera Temperature: {temp}")
                # result = [round(n, 4) for n in raw_result]
                # print(result[0],result[1],result[2],result[3])
                # print(result[4],result[5],result[6],result[7])
                # print(result[8],result[9],result[10],result[11])
                # print(result[12],result[13],result[14],result[15])
                
                # string = f"{marker_id},{temp},{result[3]*1000},{result[7]*1000},{result[11]*1000},{rpy[0]*180/math.pi},{rpy[1]*180/math.pi},{rpy[2]*180/math.pi}"
                # logger.save(string)
            # print("===========================")
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
