#include <librealsense2/rs.hpp> // 리얼센스 SDK 헤더
#include <opencv2/opencv.hpp>   // OpenCV 헤더
#include <opencv2/aruco.hpp>
#include <thread>               // 스레드
#include <mutex>                // 뮤텍스
#include <iostream>
#include <cmath>
#include <cstring>
#include <sys/socket.h> // Linux Socket
#include <arpa/inet.h>
#include <unistd.h>

class TCPClient {
    private:
        int sock;
        struct sockaddr_in serv_addr;
        bool connected = false;

    public:
        TCPClient(const char* ip, int port) {
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                std::cerr << "Socket creation error" << std::endl;
                return;
            }

            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);

            if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
                std::cerr << "Invalid address" << std::endl;
                return;
            }

            if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
                std::cerr << "Connection Failed. (Is Python Server running?)" << std::endl;
            } else {
                std::cout << "Connected to Python Server!" << std::endl;
                connected = true;
            }
        }

        ~TCPClient() {
            if (connected) close(sock);
        }

        // 데이터 전송 (Translation 3개 + Rotation 9개 = 12 floats)
        void sendPose(const std::vector<float>& T) {
            if (!connected) return;

            float buffer[16];
            for(int i = 0; i < 16; i++) {
                buffer[i] = T[i];
            }
            // 바이너리 전송 (4bytes * 16 = 64bytes)
            send(sock, buffer, sizeof(buffer), 0);
        }
};

class RealSenseCamera {
    private:
        std::mutex frame_mutex;
        rs2::pipeline pipe;
        rs2::config cfg;
        rs2::pipeline_profile profile;
        bool camera_running = true;
        rs2::frame color_frame;
        rs2::frame depth_frame;
        cv::Mat color_image;
        cv::Mat depth_image;

        int width = 640;
        int height = 480;
        int fps = 30;
        float focal_length;
        std::vector<float> principal_point = {0, 0};
    public:
        RealSenseCamera() {
            pipe = rs2::pipeline();
            cfg = rs2::config();
        }
        ~RealSenseCamera() {
            pipe.stop();
        }
        bool start() {
            try {
                rs2::align align_to_color(RS2_STREAM_COLOR);
                while (camera_running) {
                    // 프레임 수신까지 대기 (Blocking Call)
                    rs2::frameset frames = pipe.wait_for_frames();
                    // Depth -> Color 좌표계로 변환 및 픽셀 위치 매칭
                    frames = align_to_color.process(frames);

                    // 프레임 추출
                    color_frame = frames.get_color_frame();
                    //color_frame = frames.get_infrared_frame();
                    depth_frame = frames.get_depth_frame();

                    // 프레임 유효성 검사
                    if (!color_frame || !depth_frame) {
                        continue;
                    }else
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex);
                        // Color: 8비트 3채널 (BGR)
                        color_image = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP); 
                        //color_image = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP); 
                        // Depth: 16비트 1채널 (밀리미터 단위 거리 정보)
                        depth_image = cv::Mat(cv::Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
                    }
                }
                pipe.stop();
            } catch (const rs2::error & e) {
                std::cerr << "RealSense Error: " << e.what() << std::endl;
                return EXIT_FAILURE;
            } catch (const std::exception& e) {
                std::cerr << "Standard Error: " << e.what() << std::endl;
                return EXIT_FAILURE;
            }
            return true;
        }
        void stop() {
            camera_running = false;
        }
        void initialize_camera(int set_width, int set_height, int set_fps) {
            //인자 : 스트리밍종류, 가로,세로, 포맷, fps
            cfg.enable_stream(RS2_STREAM_COLOR, set_width, set_height, RS2_FORMAT_BGR8, set_fps);
            //cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
            //cfg.enable_stream(RS2_STREAM_COLOR, 1, width, height, RS2_FORMAT_Y8, fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, set_width, set_height, RS2_FORMAT_Z16, set_fps);
            width = set_width;
            height = set_height;
            fps = set_fps;
            // 카메라 스트리밍 준비
            profile = pipe.start(cfg);
            auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            rs2_intrinsics intr = color_stream.get_intrinsics();
            focal_length = sqrt(intr.fx * intr.fx + intr.fy * intr.fy);
            principal_point = {intr.ppx, intr.ppy};
            std::cout << "Focal Length: " << focal_length << std::endl;
            std::cout << "Principal Point: " << principal_point[0] << ", " << principal_point[1] << std::endl;
            // depth 이미지를 color 에 맞춰 정렬
        }
        cv::Mat getColorImage() {
            return color_image;
        }
        cv::Mat getDepthImage() {
            return depth_image;
        }
        std::vector<float> get_principal_point_and_focal_length(){
            return{principal_point[0],principal_point[1],focal_length};
        }
        std::vector<std::vector<float>> convert_pixel2mm(std::vector<std::vector<float>> center){
            std::vector<std::vector<float>> result = center;
            if(center.empty()){
                return result;
            }
            for(int i = 0; i < (int)center.size(); i++){
                if((int)center[i].size() != 16){
                    std::cerr << "Invalid center size" << std::endl;
                    continue;
                }
                float x = (center[i][3] - principal_point[0]) * center[i][11] / focal_length;
                float y = (center[i][7] - principal_point[1]) * center[i][11] / focal_length;
                float z = center[i][11];
                result[i][3] = x;
                result[i][7] = y;
                result[i][11] = z;
                std::cout << "Center [" << x/1000 << " , " << y/1000 << " , " << z/1000 << "]" << std::endl;
            }
            return result;
        }
        
    
};

class Marker_Detection{
    private:
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        cv::Ptr<cv::aruco::DetectorParameters> parameters;
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<std::vector<cv::Point2f>> rejectedCandidates;
        std::vector<float> principal_point;
        float focal_length = 0;
    public:
        Marker_Detection(){
            // AprilTag 딕셔너리 생성 (가장 많이 쓰는 36h11 태그 기준)
            // 여기서 DICT_ARUCO_ORIGINAL 대신 DICT_APRILTAG_36h11 을 쓰는 게 핵심입니다.
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
            // 파라미터 설정 (기본값 사용)
            parameters = cv::aruco::DetectorParameters::create();
        }
        std::vector<std::vector<float>> detect(cv::Mat &color_image, cv::Mat &depth_image){
            std::vector<std::vector<float>> marker_centers;
            cv::Mat color2gray;
            cv::cvtColor(color_image, color2gray, cv::COLOR_BGR2GRAY);
            cv::aruco::detectMarkers(color2gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
            // 결과 확인
            if (markerIds.size() > 0) {
                cv::aruco::drawDetectedMarkers(color_image, markerCorners, markerIds);
                std::vector<std::vector<float>> marker_centers_pixel = get_marker_centers_pixel(markerCorners, depth_image);//해당 함수는 마커중심을 시각화하기 위한 
                for(int i = 0; i < (int)marker_centers_pixel.size(); i++){
                    //디버깅을 위한 중심표시
                    cv::circle(color_image, cv::Point2f((int)marker_centers_pixel[i][0],(int)marker_centers_pixel[i][1]), 5,cv::Scalar(0,0,255),cv::FILLED,8,0);
                    //좌표계연산(mm)
                    std::vector<float> center_position = convert_pixel2mm(marker_centers_pixel[i]);
                    std::vector<std::vector<float>> marker_rotation_matrix = get_rotation_matrix(markerCorners[i],depth_image);
                    std::vector<float> matrix_rpy = get_rpy_from_matrix(marker_rotation_matrix);
                    std::vector<float> cartesian_matrix = {
                        marker_rotation_matrix[0][0], marker_rotation_matrix[0][1],marker_rotation_matrix[0][2], center_position[0],
                        marker_rotation_matrix[1][0], marker_rotation_matrix[1][1],marker_rotation_matrix[1][2], center_position[1],
                        marker_rotation_matrix[2][0], marker_rotation_matrix[2][1],marker_rotation_matrix[2][2], center_position[2],
                        0,0,0,1
                    };
                    std::cout << "id : " << markerIds[i] << std::endl;
                    std::cout << "Center [" << cartesian_matrix[3] << " , " << cartesian_matrix[7] << " , " << cartesian_matrix[11] << "]" << std::endl;
                    std::cout << "rqy    [" << matrix_rpy[0]*180/M_PI << " , " << matrix_rpy[1]*180/M_PI << " , " << matrix_rpy[2]*180/M_PI << "]" << std::endl;

                    marker_centers.push_back(cartesian_matrix);
                }
                markerIds.clear();
                markerCorners.clear();
                return marker_centers;
            }
            return std::vector<std::vector<float>>();
        }
        void set_intrinsics_param(std::vector<float> param){
            principal_point = {param[0],param[1]};
            focal_length = param[2];
        }
    private:
        std::vector<std::vector<float>> get_marker_centers_pixel(td::vector<std::vector<cv::Point2f>> detected_markerCorners,cv::Mat depth_data){
            std::vector<std::vector<float>> centers_pixel;
            for(int i = 0; i < (int)detected_markerCorners.size(); i++){
                std::vector<float> center_position(3);
                //모서리는 좌상,우상,우하,좌하 순
                std::vector<float> x_center = {detected_markerCorners[i][0].x, detected_markerCorners[i][1].x, detected_markerCorners[i][2].x, detected_markerCorners[i][3].x};
                std::vector<float> y_center = {detected_markerCorners[i][0].y, detected_markerCorners[i][1].y, detected_markerCorners[i][2].y, detected_markerCorners[i][3].y};
                //내림차순으로 최대 최소값 구하기 편하게 변경
                std::sort(x_center.begin(), x_center.end());
                std::sort(y_center.begin(), y_center.end());
                
                center_position[0] = (x_center[0] + x_center[3]) / 2;
                center_position[1] = (y_center[0] + y_center[3]) / 2;
                center_position[2] = depth_data.at<ushort>(center_position[1], center_position[0]);
                //[id,x,y]
                centers_pixel.push_back(center_position);
                //std::cout << "center : " << center_position[0] << " , " << center_position[1] << " , " << center_position[2] << std::endl; 
            }

            return centers_pixel;
        }
    
        std::vector<std::vector<float>> get_rotation_matrix(std::vector<cv::Point2f> markerCorners, cv::Mat &depth_data) {
            //해당 함수에선 각 축에 대한 벡터를 구하고 이를 통해 회전행렬을 구함
            //모서리 영점기준 : 좌상, 우상, 우하, 좌하
            std::vector<std::vector<float>> corners(4, std::vector<float>(3));
            for(int i = 0; i < 4; i++){
                std::vector<float> position = {markerCorners[i].x,markerCorners[i].y,(float)depth_data.at<ushort>(markerCorners[i].y, markerCorners[i].x)};
                corners[i] = convert_pixel2mm(position);
            }
            //기본 축 벡터 생성
            std::vector<float> x_axis(3), y_axis(3),z_axis(3);

            // (우상+우하) - (좌상+좌하)
            for(int i=0; i<3; ++i) {
                x_axis[i] = (corners[1][i] + corners[2][i]) - (corners[0][i] + corners[3][i]);
                y_axis[i] = (corners[3][i] + corners[2][i]) - (corners[0][i] + corners[1][i]); // Y축 (Down)
            }
            // yaw를 알기위한 축 정규화. 현재단계에선 y가 x와 수직이 아닐 수 있음
            normalize(x_axis);
            normalize(y_axis);
            z_axis = cross(x_axis, y_axis);
            normalize(z_axis);

            // Y축 재계산해서 제대로 수직이 되도록 설정
            y_axis = cross(z_axis, x_axis);
            normalize(y_axis);

            // 회전 행렬 생성
            std::vector<std::vector<float>> rotation_matrix(3, std::vector<float>(3));
            rotation_matrix[0][0] = x_axis[0];
            rotation_matrix[1][0] = x_axis[1];
            rotation_matrix[2][0] = x_axis[2];

            rotation_matrix[0][1] = y_axis[0];
            rotation_matrix[1][1] = y_axis[1];
            rotation_matrix[2][1] = y_axis[2];

            rotation_matrix[0][2] = z_axis[0];
            rotation_matrix[1][2] = z_axis[1];
            rotation_matrix[2][2] = z_axis[2];

            return rotation_matrix;
        }

        void normalize(std::vector<float> &v) {
            float m = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (m > 1e-9) { 
                v[0]/=m; v[1]/=m; v[2]/=m;
            }
        }
        std::vector<float> cross(std::vector<float>& a, const std::vector<float>& b) {
            return { a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] };
        }

        std::vector<float> convert_pixel2mm(std::vector<float> center){
            std::vector<float> result = center;
            if(center.empty()){
                return result;
            }
            float x = (center[0] - (float)principal_point[0]) * center[2] / focal_length;
            float y = (center[1] - (float)principal_point[1]) * center[2] / focal_length;
            float z = center[2];
            result[0] = x;
            result[1] = y;
            result[2] = z;
            return result;
        }

        std::vector<float> get_rpy_from_matrix(std::vector<std::vector<float>>& R) {
            float roll, pitch, yaw;

            // Pitch의 코사인 성분(sy) 계산
            // sy = sqrt(r00^2 + r10^2)
            float sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);

            // 짐벌락 체크 (sy가 0에 가까우면 Pitch가 +/- 90도인 상태)
            bool singular = sy < 1e-6;

            if (!singular) {
                // [일반적인 경우]
                // Roll (X축 회전) = atan2(r21, r22)
                roll = atan2(R[2][1], R[2][2]);
                
                // Pitch (Y축 회전) = atan2(-r20, sy)
                pitch = atan2(-R[2][0], sy);
                
                // Yaw (Z축 회전) = atan2(r10, r00)
                yaw = atan2(R[1][0], R[0][0]);
            } else {
                // [짐벌락 발생: Pitch = +/- 90도]
                // Yaw를 0으로 고정하고 Roll을 계산합니다.
                roll = atan2(-R[1][2], R[1][1]);
                pitch = atan2(-R[2][0], sy);
                yaw = 0;
            }

            return {roll, pitch, yaw};
        }
    
};

// 원활한 행렬연산을 위해 cv::Mat 활용
cv::Mat make_Transform(std::vector<float> data){
    //해당 회전행렬은 오일러 회전행렬(ZYX)를 참고하여 작성
    double roll = data[3] * CV_PI / 180;
	double pitch = data[4] * CV_PI / 180;
	double yaw = data[5] * CV_PI / 180;
	cv::Mat coordinate = cv::Mat::zeros(4, 4, CV_32F);
	coordinate.at<float>(0, 0) = cos(yaw)* cos(pitch);
	coordinate.at<float>(0, 1) = sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw);
	coordinate.at<float>(0, 2) = cos(yaw) * sin(pitch)*cos(roll) + sin(yaw)*sin(roll);
	coordinate.at<float>(0, 3) = data[0]*1000;
	coordinate.at<float>(1, 0) = sin(yaw) * cos(pitch);
	coordinate.at<float>(1, 1) = sin(roll) * sin(pitch)*sin(yaw) + cos(roll)*cos(yaw);
	coordinate.at<float>(1, 2) = sin(yaw) * sin(pitch)*cos(roll) - cos(yaw)*sin(roll);
	coordinate.at<float>(1, 3) = data[1]*1000;
	coordinate.at<float>(2, 0) = -sin(pitch);
	coordinate.at<float>(2, 1) = cos(pitch) * sin(roll);
	coordinate.at<float>(2, 2) = cos(pitch) * cos(roll);
	coordinate.at<float>(2, 3) = data[2]*1000;
	coordinate.at<float>(3, 0) = 0;
	coordinate.at<float>(3, 1) = 0;
	coordinate.at<float>(3, 2) = 0;
	coordinate.at<float>(3, 3) = 1;
    return coordinate;
}

std::vector<float> Mat2vector(cv::Mat mat){
    std::vector<float> result;
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            result.push_back(mat.at<float>(i, j));
        }
    }
    return result;
}

cv::Mat vector2Mat(std::vector<float> vec){
    cv::Mat result = cv::Mat::zeros(4, 4, CV_32F);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            result.at<float>(i, j) = vec[i*4 + j];
        }
    }
    return result;
}




int main() {
    //아래의 좌표들을 설정 시 회전각도를 ZYX순으로 고려하여 설정 
    std::vector<float> base_to_marker = {0.2,0.0,1.0,180,0.0,-90};//임의로 설정된 마커의 고정위치
    std::vector<float> camera_to_tool = {0.0,0.0,-0.1,0.0,180,90};//툴플렌지로부터 카메라까지의 위치
    //std::vector<float> camera_to_marker;//구해지는 마커의 좌표
    std::vector<float> base_to_tool; // base_to_marker*camera_to_marker.inverse()*tool_to_camera.inverse()
    // 로봇이 zero상태이면서 카메라가 아래를 향하도록 부착했을 경우
    // std::vector<float> end_effector_to_camera = {0,0,-0.1,180,0,90};// {x,y,z,roll,pitch,yaw}(mm, degree) 
    cv::Mat base_to_marker_tf = make_Transform(base_to_marker);
    cv::Mat camera_to_tool_tf = make_Transform(camera_to_tool);
    //cv::Mat base_to_tool = base_to_marker * tool_to_camera.inv();


    TCPClient client("127.0.0.1", 5000);
    RealSenseCamera camera;
    Marker_Detection marker_detection;
    // 가능 해상도 : 848*480, 1280*720
    int width = 848;
    int height = 480;
    int fps = 30;
    camera.initialize_camera(width, height, fps);
    std::vector<float> intrinsics_param = camera.get_principal_point_and_focal_length();
    marker_detection.set_intrinsics_param(intrinsics_param);
    
    std::thread camera_thread = std::thread(&RealSenseCamera::start, &camera);
    cv::Mat color_img;
    cv::Mat depth_img;
    while(true){
        color_img = camera.getColorImage();
        depth_img = camera.getDepthImage();
        if(color_img.empty() || depth_img.empty()){
            continue;
        }

        
        std::vector<std::vector<float>> marker_transforms = marker_detection.detect(color_img, depth_img);
        for(int i = 0; i < marker_transforms.size(); i++) {
            cv::Mat camera_to_marker_tf = vector2Mat(marker_transforms[i]);
            cv::Mat base_to_tool_tf = base_to_marker_tf * camera_to_marker_tf.inv() * camera_to_tool_tf;
            std::vector<float> base_to_tool_vec = Mat2vector(base_to_tool_tf);
            client.sendPose(base_to_tool_vec);
        }
        // 디버깅용 단순 시각화
        cv::Mat depth_debug;
        cv::Mat depth_debug_bgr;

        // 거리 범위 설정 (280mm ~ 3000mm). 카메라의 문서상 최소 인식거리 280mm
        // 가까울수록 밝게(255), 멀수록 어둡게(0)
        double min_dist = 280.0;
        double max_dist = 3000.0;
        double alpha = (0.0 - 200.0) / (max_dist - min_dist);
        double beta = 200.0 - (min_dist * alpha);

        // 변환 수행 (Clamping + Mapping + Format Conversion)
        depth_img.convertTo(depth_debug, CV_8UC1, alpha, beta);

        // 유효하지 않은 0값(No Depth)이 255(흰색)가 된 것을 다시 0(검은색)으로 보정
        depth_debug.setTo(0, depth_img == 0);
        cv::cvtColor(depth_debug, depth_debug_bgr, cv::COLOR_GRAY2BGR);
        //cv::cvtColor(depth_debug, depth_debug_bgr, cv::COLOR_GRAY2BGR);


        // 두 이미지를 가로로 붙여서 보여줌
        cv::Mat concat_image;
        if((int)depth_debug_bgr.cols != 848 || (int)depth_debug_bgr.rows != 480){
            //cv::resize(depth_debug_bgr, depth_debug_bgr, cv::Size(640, 480));
            //cv::resize(color_img, color_img, cv::Size(640, 480));
            cv::resize(color_img, color_img, cv::Size(848, 480));
            cv::resize(depth_debug_bgr, depth_debug_bgr, cv::Size(848, 480));
        }
        cv::hconcat(color_img, depth_debug_bgr, concat_image);
        //cv::hconcat(color_img_bgr, depth_debug_bgr, concat_image);
        color_img.release();
        depth_debug_bgr.release();

        cv::imshow("Preview", concat_image);

        // ESC 키(27)를 누르면 종료
        if (cv::waitKey(1) == 27) {
            break;
        }
        usleep(1000);
    }
    camera.stop();
    camera_thread.join();
    std::cout << "RealSense Camera Started. Press 'ESC' to exit." << std::endl;

    return EXIT_SUCCESS;
}