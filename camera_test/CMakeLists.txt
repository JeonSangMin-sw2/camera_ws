cmake_minimum_required(VERSION 3.0)
project(camera_test)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(realsense2 REQUIRED realsense2)

add_executable(camera_test main.cpp)
target_include_directories(camera_test PUBLIC ${realsense2_INCLUDE_DIRS})
target_link_libraries(camera_test
    ${OpenCV_LIBS}
    ${realsense2_LIBRARIES}
)