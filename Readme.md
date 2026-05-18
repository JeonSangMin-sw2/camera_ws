# Camera Calibration Manual

## Setup
- 보정에 있어 필요한 라이브러리가 많으므로, 가상공간을 만들어서 설치 후 사용하는것을 권장

```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# 종료 시 deactivate 입력
```

## 1. 하드웨어 스펙 및 마커 정보
- **사용 카메라**: realsense D405
    - 해상도: 1280 x 720
    - 프레임 레이트: 30 FPS
    - 장착 위치
        - Head 존재하는 모델일 경우 : 옵션으로 구매가능한 브라켓 장착
        - Head 없는 모델일 경우 : 별도의 브라켓 사용 
- **사용 마커**: apriltag
    - **Plate 마커**: 크기 80mm. ID: Left(7), Right(8)


## 2. 보정 UI (Calibration UI) 설명
- **User Tab**: 일반 사용자를 위한 단순화된 인터페이스. 연결, 초기화, 자동 데이터 취득, 연산 및 적용 버튼으로 구성.
- **Developer Tab**: 전문가용 상세 설정 포함. 최적화 파라미터(`lambda_cam`), 데이터 소스(Live/NPZ/Sim), Solver 선택 가능.
- **주요 기능**:
    - 실시간 로봇 연결 및 서보 제어
    - 자동 경로 생성 및 데이터 로깅
    - 보정 결과 시각화 및 JSON/NPZ 저장

## 3. 보정 과정 상세 설명
1. **Zero Pose 확인**: 로봇을 기구적 원점(Zero Pose)으로 이동시켜 물리적 정렬 상태 확인.
2. **Init Pose 이동**: 보정을 시작하기 위한 준비 자세(Ready Pose)로 이동.
3. **교시(Teaching)**: 로봇의 조인트를 미세 조정하여 카메라가 모든 마커를 명확히 인식할 수 있도록 설정.
4. **Auto 데이터 취득**: 설정된 각도 및 거리 스텝에 따라 로봇이 이동하며 조인트 각도와 마커 포즈 데이터를 자동 수집.
5. **QP 솔버 연산**: 수집된 데이터를 바탕으로 QP(Quadratic Programming) 최적화를 수행하여 조인트 오프셋 및 카메라 외부 파라미터 산출.
6. **오프셋 적용**: 연산된 오프셋을 로봇 컨트롤러에 적용하고 설정 파일에 저장.
7. **보정 결과 확인**: 보정 전후의 오차를 비교하고, 필요시 NPZ 샘플 데이터를 통해 재검증 수행.

## 4. 시스템 아키텍처
- **UI 프레임워크**: Python Tkinter (Main calibration UI), PySide6 (Additional calibration UI).
- **핵심 엔진**:
    - `calibration_core`: 로봇 모델링 및 데이터 처리 루틴.
    - `calibration_optimizer`: QP 및 Least Squares 기반 최적화 엔진.
    - `marker_detection`: OpenCV 기반 마커 인식 및 포즈 추정.

## 5. 최적화 QP 솔버 구현 방식
- **목적 함수**: Forward Kinematics 예측값과 마커 관측값 사이의 SE(3) Log Error(수렴성 향상) 최소화.
- **제약 조건**: 조인트 가동 범위 내 오프셋 제한(`enforce_joint_offset_limits`) 및 변화량(Step bound) 제한 적용.
- **정규화**: `lambda_cam_pos`, `lambda_cam_rot` 파라미터를 통해 카메라 외부 파라미터가 설계값에서 과도하게 벗어나지 않도록 규제.
- **특징**: 복수 팔(Dual Arm) 및 헤드 조인트를 동시 최적화 가능.

## 6. 추가 구현 사항
1. **camera_intrinsic_calib_ui**: 카메라 내부 파라미터(초점거리, 왜곡 계수) 보정 도구.
2. **marker_bracket_calib_ui**: 로봇 말단(EE)과 마커 사이의 기구적 오프셋 산출용 5/6축 스윕 도구.
3. **샘플 데이터 NPZ 저장**: 보정 시 사용된 모든 로우 데이터를 NPZ 형식으로 저장하여 사후 분석 지원.
4. **home_offset_reset**: 특정 시점의 오프셋으로 복구하거나 초기화하는 기능.
5. **check_calibration_result**: 양팔이 대칭인 자세로 보내 보정 결과의 유효성을 검증하고 오차 리포트 생성.

## 7. 현재 기능 정확도
- (항목 기입 필요)

## 8. 추가 기능 개선 필요사항
- (항목 기입 필요)