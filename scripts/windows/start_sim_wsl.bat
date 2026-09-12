@echo off
chcp 65001 >nul
echo [INFO] WSL2 환경에서 RBY1 시뮬레이터 컨테이너를 구동합니다...

:: X11 접속 권한 허용
wsl -d Ubuntu-22.04 -e bash -c "DISPLAY=:0 xhost +" >nul 2>&1

:: 기존 실행 중인 컨테이너가 있으면 삭제
wsl -d Ubuntu-22.04 -e docker rm -f rby1_sim >nul 2>&1

:: 시뮬레이터 컨테이너 실행 (WSLg GUI 및 50051 포트 연동, 자동 재시작)
wsl -d Ubuntu-22.04 -e docker run -d --restart unless-stopped --name rby1_sim --ipc=host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /mnt/wslg:/mnt/wslg:rw -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -p 50051:50051 rainbowroboticsofficial/rby1-sim:0.10.6-m_v1.3

if %errorlevel% equ 0 (
    echo [SUCCESS] RBY1 시뮬레이터가 성공적으로 시작되었습니다!
    echo [INFO] 포트: 127.0.0.1:50051 (또는 localhost:50051)
    echo [INFO] 화면에 MuJoCo 3D 뷰어 창이 표시됩니다.
) else (
    echo [ERROR] 시뮬레이터 실행 실패. Docker 상태를 확인해주세요.
)
pause
