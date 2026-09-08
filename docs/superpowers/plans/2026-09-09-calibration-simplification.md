# 09/07 기준 비교와 마커 기하 기반 Step1 복구 Implementation Plan

> 승인된 수정안과 실행 기록. 사용자가 구현 및 127.0.0.1:50051 시뮬레이터 검증을 승인했다. 이전 문서의 “Step1 엔코더 기록 유지”, “encoder bracket fit 유지” 지침은 폐기한다. 실물 홈 쓰기·Git 커밋/push·사용자 설정 초기화는 승인 범위가 아니다.

**Goal:** Step1을 미보정 엔코더/FK에 의존하지 않는 마커 원 기반 보정으로 정리하고, 실물/가상 차이를 관측 생성과 초기 전원으로 제한한다.

**Architecture:** 마커 계층은 실제 또는 가상 4×4 관측을 반환한다. Base는 관측점으로 원을 추정하고, Joint/Marker는 원의 기하학으로 각자의 보정만 계산한다. SDK와 엔코더는 이동·상태 확인 및 가상 센서 내부에 머무르며 Step1 추정기의 입력이 아니다. Step2의 미지 오프셋을 포함한 모델 추정은 Step1과 별도 검토한다.

**Tech Stack:** 기존 Python, NumPy/SciPy, OpenCV/RealSense, rby1 SDK, PySide6, YAML, unittest.

**Spec:** 이번 대화의 사용자 원칙: 링크 형상은 동일하고 관절 영점은 미지이며, Step1은 관측한 3차원 원으로 보정한다. 내부파라미터는 항상 camera_intrinsics.yaml. sim 하나만 사용. 신규 보조 소스는 담당 파일에 통합한다.

## 1. 비교 기준과 정정

- 기준: d3a2022, 2026-09-07 19:49:28 +0900, test_for_head_camera.
- 비교 대상: 40ad8eb, 2026-09-08 20:23:41 +0900.
- 중간 커밋: f6c973f → dad11cc → f0eff75 → 40ad8eb.
- 차이: 32개 tracked 파일, 3432줄 추가/1861줄 삭제. 신규 core Python 소스 7개.
- 이는 로컬 Git 커밋 비교이며 push 시각이나 누가 어떤 대화로 작성했는지는 단정하지 않는다.

중요한 정정: 09/07 구현이 완전히 마커 전용이었다고 확인된 것은 아니다. 다음 의존성이 이미 존재했다.

| 항목 | 09/07 소스 근거 | 09/08 이후 |
|---|---|---|
| 엔코더 각도 기반 원 피팅 | JointCalibrator.py:897–904의 angles_A/B와 fit_circle_3d_and_6dof_misalignment | 유지. robust 설정만 실물/가상 공통 True로 변경 |
| FK 기반 J3/v1.2 J5 중심·축 | JointCalibrator.py:1061–1086의 true_a_cand_cam, p_cand_cam | 유지. 카메라 장착 모드 판정 변경 |
| J6 절대 엔코더값 가산 | JointCalibrator.py:980–989의 j7_current_pos_deg | 첫 프레임에서 중앙값 기반 helper로 변경. 의존성 자체는 기존부터 존재 |
| v1.3 관절 계산이 Marker에 위치 | main_ui.py:2149에서 compute_wrist_joints_from_3axis_sweeps 호출 | 유지 |
| v1.3 결과에 converged=True 지정 | main_ui.py:2175,2180 | 유지 |
| mock/ui_only 분기 | main_ui.py 및 Base | 기존부터 존재 |
| 내부파라미터 YAML 선택적 사용 | marker_detection.py의 use_calib_int=False | factory/calibrated selector로 변경 |
| 브래킷의 일부 FK 사용 | MarkerCalibrator.py의 pts_ee 생성과 기하 경로 내 활용 | 신규 encoder-relative fit으로 주 경로까지 교체 |

따라서 d3a2022는 “비교 기준”이지 “검증 없이 통째로 복원할 정답”이 아니다. 이전 답변에서 기존 문제와 09/08 신규 변경을 충분히 구분하지 못했다.

재확인 명령:
    git show d3a2022:core/calibration/JointCalibrator.py
    git show d3a2022:core/calibration/MarkerCalibrator.py
    git diff d3a2022..40ad8eb -- core/calibration main_ui.py
    git diff d3a2022..40ad8eb -- core/calibration_optimizer.py core/marker_detection.py config/setting.yaml

## 2. 09/08 변경의 처리 방침

| 변경 | 판단 | 처리 |
|---|---|---|
| f6c973f: bracket_fitting.py와 fit_encoder_bracket 주 경로 추가 | 원 기반 보정에서 encoder/FK 모델 맞춤으로 방식 변경 | Step1에서 제거. 함수만 Marker로 옮겨 유지하지 않음 |
| f6c973f: Step1.5 평면/축 기반 계산을 헤드 엔코더 최적화로 교체 | 계산 의미가 달라지는 큰 변경 | Step1 복구와 섞지 않고 Task 7에서 별도 비교·검증 |
| f6c973f: Step2 SE(3), 정규화/anchor, bounds, head gauge, 실패 처리 변경 | 수학 수정·정책 변경·보호 로직 혼재 | 일괄 rollback 금지. 수치식/정책/안전을 분리해 Task 7에서 판정 |
| f6c973f: factory/calibrated 선택 UI·모듈 | 최신 고정 YAML 요구와 충돌 | 선택 분기 제거, 마커 클래스가 고정 파일 적용 |
| f6c973f: 고정 simulation model, 공통 robust fitting | 센서 정답 독립성과 실물/가상 동일 처리 취지는 유효 | 마커 내부로 통합. Step1의 encoder 기반 피팅은 별도로 제거 |
| f6c973f: YAML 원자적 저장·UI 정밀도 | 계산 원칙을 바꾸지 않는 개선 | 유지. 숫자 함수는 main_ui로 이동 |
| dad11cc: logger 수정 | 잘못된 self.logger 참조 수정 | 유지 |
| dad11cc 등: setting.yaml 보정 수치 변경 | 코드 이외 초기 조건도 달라짐 | 자동 복원/초기화 금지. 비교 시험에서는 설정을 고정하고 출처 기록 |
| f0eff75: J6 절대값에 damping을 곱하던 갱신 수정 | 기존 갱신 오류 수정 | “보정 증분에 damping 적용” 원칙 유지. 새 기하 residual 계약에 맞춰 재작성 |
| f0eff75/40ad8eb: 실패한 선행 관절 뒤 단계 차단 | 실패 은폐 방지 | 유지, 수동/v1.3에도 일관되게 적용 |
| 40ad8eb: J6 전체 프레임 회전 일관성 검사 | 마커 기반 품질 진단으로 사용 가능 | Joint에 통합하되 절대 엔코더 입력/가산 제거. 임계값 검증은 별도 |
| 40ad8eb: 스윕 기본 시간 J6 20초/J5 15초 | 측정 조건 변경, 효과 미확정 | 09/07 조건과 분리 비교. 시간으로 수학 문제를 덮지 않음 |
| 40ad8eb: 모든 debug iteration 기록 | 실패 재현에 유용 | 마커 원자료·명령 이력 중심으로 유지, 새 source 생성 불필요 |

09/07의 Joint 함수 기본은 12초, 개별 UI worker 기본은 15초였다. “과거 시간”도 하나가 아니므로 호출 경로까지 비교해야 한다.

## 3. 고정할 설계 규칙

1. Step1 계산은 마커 pose/3차원점과 원 기하학만 입력받는다. 엔코더, FK 결과, live robot 객체, 추정 중인 시뮬레이션 정답을 전달하지 않는다.
2. 원 추정의 초기값·방향 선택·품질 기준에도 미보정 FK 축을 쓰지 않는다. 엔코더 각도 차이를 사용하는 원 피팅 역시 이번 Step1 요구에서는 제거한다.
3. J3/J5는 사용자가 정의한 원 일치 조건을 목적 조건으로 한다. J6는 관측 원의 축·마커 좌표계에서의 상대 방향을 사용한다.
4. 데이터로부터 산출하는 기하 오차와, 명령 측에서 누적하는 적용 보정량은 구분한다. 현재 적용량을 보관하는 것은 측정 엔코더 절대각을 보정식에 넣는 것과 다르다.
5. 기하학만으로 결정되지 않는 값은 실패/미결정으로 표시한다. 몰래 FK, 가상 정답, nominal 축으로 메워 성공 처리하지 않는다.
6. 원 추정은 SDK 없이 저장된 마커 데이터만으로 재생 가능해야 한다.
7. 이동/안전 상태 확인에는 SDK 사용 가능. 가상 마커 생성에는 SDK 엔코더와 고정 가상 오프셋 사용 가능. 두 경우 모두 계산기 외부의 책임이다.
8. sim의 단일 소유자는 Marker_Transform이다. --ui 또는 카메라 부재 시 True. 준비 자세/스윕/검사/재시도/수렴/적용에는 모드별 알고리즘이 없다.
9. 초기 power 패턴은 연결 대상에 따라 실물 48v, 시뮬레이터 .*이다. sim=True인 실물 SDK 연결도 가능하므로 마커 sim으로 전원 대상을 판단하지 않는다. servo는 공통 정책을 따른다.
10. Step2는 미지 관절 오프셋을 포함한 모델 추정 단계이므로 Step1의 입력 금지를 무조건 확장하지 않는다. 미보정 엔코더를 정답으로 고정하는 것과 미지 오프셋을 함께 추정하는 것은 구분한다.
11. 기존 설정/로그/실험 파일·물리 홈 오프셋은 승인 없이 삭제·덮어쓰기하지 않는다.

## 4. 승인된 구현 계획 원문

아래 체크리스트는 승인 당시 계획을 보존한다. 실제 구현 범위와 검증 결과는 7절에서 별도로 판정한다.

### Task 1 — 비교 재현 기준과 독립성 테스트를 먼저 고정

**Files:** 기존 tests/test_calibration_regression.py, tests/test_j6_reference.py, tests/test_joint_retry.py, tests/test_calibration_workflow.py.

- [ ] 09/07/현재 버전, 동일 설정, 동일 마커 입력으로 비교한다. 설정 수치 변경과 알고리즘 변경을 섞지 않는다.
- [ ] 같은 물리 관측을 유지하며 상위 관절 엔코더 표시 기준만 바꾸는 시험을 기존 회귀 테스트로 만든다. 새 Step1 추정기에는 애초에 엔코더 인수가 없어야 한다.
- [ ] 원 자료 전체에 공통 강체변환을 적용해도 보정 residual이 유지되는지 확인한다. 기준 축을 사용하는 경우 그 기준도 정의된 좌표계대로 함께 변환한다.
- [ ] 가상 센서에서 관절별 영점 오차, 준비 자세, 좌우 팔, v1.2/v1.3을 바꿔 원 기반 추정의 성립 범위를 확인한다. 정답은 assertion에만 사용한다.
- [ ] 불량 원/짧은 호/회전 불연속/관측 누락은 실패로 처리하는 테스트를 작성한다.

직전 오프라인 단일 계산 시험에서 동일 마커 데이터에 J0 엔코더 +10/-10도를 넣자 v1.2 J5 결과는 -5.2618 → -4.2491/-6.8161도로 변했다. J3도 변했으며 해당 J6 사례는 일정했다. 이는 현재 코드의 불필요한 의존성 재현이지, 최신 실물 J6 실패의 단일 원인 확정은 아니다.

**통과 조건:** 기존 실패가 재현되고 새 계산의 독립성 판정 기준이 고정된다. 물리 로봇 시험은 이 단계에 포함하지 않는다.

### Task 2 — Base의 원 추정을 마커 전용으로 변경

**Files:** core/calibration/CalibratorBase.py, JointCalibrator.py, MarkerCalibrator.py, 기존 regression 테스트.

**Interface:** BaseCalibrator.fit_observed_circle(poses) -> dict. 입력은 (N,4,4) 마커 pose. 결과는 center_m, radius_m, axis, residual_rms_m, accepted, failure_reason이다. axis는 부호 모호성이 있는 관측 평면 법선이며 절대 로봇 축으로 해석하지 않는다.

- [ ] 기존 fit_circle_3d(points, robust=True)의 점 기반 경로를 출발점으로 사용한다. 결과 단위는 경계에서 m로 정규화한다.
- [ ] 엔코더 각도와 axis_prior가 없는 API를 추가하고 두 캘리브레이터를 이 API로 연결한다.
- [ ] 원 중심/법선/반지름 추정 및 outlier 처리가 입력 점만 사용하는지 확인한다.
- [ ] 기존 angle-indexed fitter의 운영 호출이 없어지면 제거한다. 진단용으로도 같은 원을 다시 다른 방식으로 추정하지 않는다.
- [ ] 축 부호는 관측 순서/마커 회전의 일관된 규약으로 정하고, 정보가 부족하면 미결정 처리한다.

테스트 예시는 기존 regression 파일에 다음 형태로 추가한다:
    theta = np.linspace(-0.5, 0.5, 41)
    poses = np.repeat(np.eye(4)[None], len(theta), axis=0)
    poses[:, :3, 3] = np.c_[0.08*np.cos(theta), 0.08*np.sin(theta), np.full(len(theta), 0.4)]
    result = BaseCalibrator.fit_observed_circle(poses)
    assert result['accepted']
    np.testing.assert_allclose(result['center_m'], [0, 0, 0.4], atol=1e-5)
    assert abs(result['radius_m'] - 0.08) < 1e-5

**통과 조건:** SDK 없이 원을 계산하며, 로봇 상태/URDF/엔코더/가상 정답을 변경해도 동일 pose 입력의 결과가 같다.

### Task 3 — J3/J5/J6 추정을 기하 residual과 보정 갱신으로 분리

**Files:** core/calibration/JointCalibrator.py, core/calibration/joint_reference.py, main_ui.py, 기존 J6/retry/workflow 테스트.

- [ ] J3/v1.2 J5의 true_a_cand_cam, p_cand_cam 및 FK 기준 투영을 제거한다.
- [ ] 원 일치 조건의 중심·평면·반지름 잔차를 정리한다. 어느 잔차가 보정 크기와 방향을 결정하는지 모델별로 유도하고 Task 1 데이터로 검증한다.
- [ ] 원 일치의 스칼라 오차만으로 방향이 결정되지 않으면, 명령 측의 제한된 ±보정 시험과 관측 오차 증감을 사용하는 공통 탐색을 설계한다. 로봇 없이 가상 관측 시험으로 먼저 검증한다. 일치 가능한 기하 조건 자체가 없으면 강제로 보정값을 만들지 않는다.
- [ ] J6는 마커 좌표계의 원 축 관계와 명시적인 기준 장착 방향에서 잔차를 얻는다. helper의 encoder_j6_deg 및 절대 엔코더 가산을 제거한다.
- [ ] J6 기준 자세/티칭 변경 시 어떤 관측 관계가 목표인지 검증한다. 임의 자세의 관측만으로 절대 영점을 결정할 수 있다고 가정하지 않는다.
- [ ] 내부 계산은 residual/보정 증분을 반환하고, 실행기는 현재 적용 보정량에 증분을 누적한다. damping은 증분에 적용한다. 절대 목표와 증분을 혼용하지 않는다.
- [ ] v1.3 compute_wrist_joints_from_3axis_sweeps를 Joint로 이동한다. 수동/Full Auto에 같은 품질·수렴 결과를 전달한다.
- [ ] 측정 거부, 비수렴, 취소 시 후보 저장·다음 단계 진행을 차단한다. hard-coded converged=True를 제거한다.

**통과 조건:** 미보정 상위 관절 자세/영점에 대한 불필요한 의존성이 없고, 반복 보정이 실제 관측 원의 목표 관계를 개선한다. 기하학적 식별 한계가 있으면 이 단계에서 보고하고 FK 방식을 다시 넣지 않는다.

### Task 4 — 브래킷을 원 기반 경로로 복구하고 중복 계산 제거

**Files:** core/calibration/MarkerCalibrator.py, core/calibration/bracket_fitting.py, main_ui.py, 기존 workflow 테스트.

- [ ] 09/08에 추가된 fit_encoder_bracket 호출과 encoder-relative least squares 경로를 Step1에서 제거한다.
- [ ] 09/07 원 축/반지름/중심 기반 코드에서 순수 기하 부분만 선택한다. pts_ee를 만드는 미보정 FK와 이를 사용하는 translation 추정은 복원하지 않는다.
- [ ] 좌표계는 관측 마커 기준과 고정 설계 치수/장착 규약으로 명시한다. 동적인 FK 위치를 설계 치수 대신 사용하지 않는다.
- [ ] Marker 결과에서 관절 값을 재추정하거나 갱신하지 않는다. 원 기반으로 결정되지 않는 브래킷 성분은 명시적으로 미결정 처리한다.
- [ ] 수동/Full Auto 모두 같은 함수에 관측 원을 전달한다. 자료 부족 시 실제 필요한 스윕을 안내하며 새 FK fit을 유지하기 위해 무조건 3축을 요구하지 않는다.
- [ ] 호출자가 사라진 bracket_fitting.py를 삭제한다.

**통과 조건:** SDK/엔코더 없는 브래킷 재생, 양팔/버전별 검증, 관절 저장값을 브래킷이 변경하지 않는 테스트 통과.

### Task 5 — 마커 입력과 내부파라미터를 한 곳으로 통합

**Files:** core/marker_detection.py, core/simulation_model.py, core/camera_intrinsics.py, main_ui.py, core/calibration_core.py, core/wizard_widget.py, 캘리브레이터 공통 수집부.

- [ ] Marker_Transform에 sim 하나를 둔다. --ui는 카메라를 열지 않으며, 카메라 부재일 때 자동 sim을 사용한다. 설정 파싱 오류를 카메라 부재로 숨기지 않는다.
- [ ] 가상 생성 모델과 UI의 SimulatedMarkerTransform을 마커 파일 내부로 통합한다. 세션당 가상 정답/난수 발생기는 하나이며 추정값 저장으로 정답을 바꾸지 않는다.
- [ ] Step1 수집 API는 마커 pose를 전달한다. q_full을 필수 샘플 필드로 묶지 않는다. 가상 센서가 필요로 하는 q는 센서 내부에서 읽는다. Step2의 엔코더 포함 데이터 계약은 유지한다.
- [ ] 실물/가상에 동일한 관측 시간창·반환 타입·단위·실패 처리를 적용한다. 실제 mm→m 변환과 가상 m의 중복 변환을 막는다.
- [ ] 카메라 검출기는 camera_intrinsics.yaml만 검증·적용한다. factory/calibrated UI와 runtime 선택 키를 제거한다.
- [ ] 저장/재로드/적용은 마커 클래스가 관리한다. IntrinsicsCalibrator는 계산 결과를 제공하고 UI는 요청만 전달한다. 수집 도중 설정 변경은 거부한다.
- [ ] camera_intrinsics_d435i.yaml은 보존하되 자동 선택하지 않는다.
- [ ] 재연결 시 모든 단계가 같은 제공자를 참조하도록 한다. 작업 중 전환을 막고 관측 출처가 바뀌면 명시적인 새 세션을 요구한다.
- [ ] ui_only/is_mock/클래스명 판정/Step2 sim 선택/가짜 성공/별도 수집 주기를 제거한다. NPZ 재생은 보존한다.
- [ ] 전원 이외 SDK 구동·servo·준비 자세·스윕·재시도는 공통 경로로 통합한다.

**통과 조건:** 같은 마커 입력을 주면 실물/가상에서 같은 동작·판정 이벤트열이 나오며, 시뮬레이션 구분이 추정기 내부에 존재하지 않는다.

### Task 6 — 파일 통합과 유지보수 정리

| 파일 | 최종 처리 |
|---|---|
| numeric_fields.py | main_ui.py로 두 함수 이동, 정밀도 유지 후 삭제 |
| camera_intrinsics.py | marker_detection.py로 고정 파일 로딩 통합 후 삭제 |
| simulation_model.py | marker_detection.py의 내부 가상 센서 모델로 통합 후 삭제 |
| joint_reference.py | encoder 의존을 제거한 마커 품질/기하 기능만 Joint로 통합 후 삭제 |
| bracket_fitting.py | Step1의 encoder/FK 추정 경로 제거 후 삭제 |
| head_camera_zero.py | Task 7에서 기능 의미 검토 후 유지할 함수만 HeadCameraCalibrator.py로 이동 |
| config_store.py | 공통 원자적 저장으로 유지 |
| paths.py | 공통 경로 관리로 유지 |

- [ ] main_ui/Base/Joint/Marker/HeadCamera의 import를 함께 변경하고 core.*와 flat import 이중 로딩을 정리한다.
- [ ] HeadCamera와 optimizer의 상호 import는 호출 시점 import로 관리해 순환 참조를 막는다.
- [ ] sim 시작에 RealSense 하드웨어 초기화가 필요하지 않도록 카메라 import/생성을 분리한다.
- [ ] 무효 debug plot 함수, 읽히지 않는 sim_q_full/use_calib_int, 제거된 옵션을 정리한다.
- [ ] 테스트 대역은 테스트에만 둔다. scratch·사용자 데이터는 자동 삭제하지 않고 삭제 모듈 참조를 개별 확인한다.
- [ ] setup.py와 camera_calibrator.spec의 import/패키징을 확인한다.

**통과 조건:** 새 운영 Python 파일 0개, 원칙에 어긋난 계산의 임시 호환 wrapper 0개, 최근 추가 보조 소스 6개 통합/제거.

### Task 7 — Step1.5/Step2 변경을 독립 검증하고 출력 연결

**Files:** HeadCameraCalibrator.py, calibration_optimizer.py, head_camera_zero.py, homeoffset_core.py, main_ui.py; 기존 head/regression/persistence 테스트.

- [ ] 09/07과 09/08의 Step1.5 기하 방식/헤드 엔코더 모델 방식의 입력과 식별 가능한 출력을 비교한다. 어느 쪽도 미보정 엔코더를 물리 정답으로 고정하지 못하게 한다.
- [ ] Step2의 SE(3) exp/log 수정은 독립 행렬 지수/로그 및 Jacobian 검증으로 판단한다. 오래된 구현이라는 이유로 수치 오류를 되돌리지 않는다.
- [ ] Step1 anchor 변경, camera prior/bounds, head tilt gauge와 camera-forward zero를 개별 비교한다. 변경을 파일 통합에 섞어 숨기지 않는다.
- [ ] 새 Step1 결과와 Step2의 부호·단위·보정 의미를 연결한다. 기하적으로 결정되지 않은 값을 물리 영점으로 전달하지 않는다.
- [ ] 헤드 기구학적 영점과 카메라 정면 기준을 구분한다. camera_command_to_encoder를 임의로 실제 이동 명령에 연결하지 않는다.
- [ ] 비수렴/관측 불가능 결과 적용 차단, 유효하지 않은 헤드 결과의 기계 홈 쓰기 차단은 유지한다.
- [ ] 추가 계산 정책이 필요한 경우 검증 결과와 변경 효과를 별도로 보고한다. 이번 Step1 원칙을 이유로 Step2 전체의 엔코더를 삭제하지 않는다.

**통과 조건:** 기존 결과 JSON/NPZ 읽기, 보정 부호, 헤드 출력의 의미 및 미관측 성분 보호를 확인한다. 물리 홈 쓰기는 시험하지 않는다.

### Task 8 — 회귀·시뮬레이터·실물 순서로 검증

- [ ] 기존 테스트의 “encoder bracket fit이 정답” 같은 전제를 새 명세로 바꾼다. 구 테스트가 통과하는 것만으로 완료 판단하지 않는다.
- [ ] 원 기하학/상위 영점 변화/준비 자세 변화/공통 좌표변환/양팔/버전/노이즈/불량 관측 회귀를 수행한다.
- [ ] 수동과 Full Auto의 단계 순서, 실패/비수렴/취소, 재연결, 설정 저장 실패 보존을 검사한다.
- [ ] 전체 테스트는 OSQP가 있는 실행 환경에서 실행한다. 풀이기 실패를 비제약 해법으로 우회하지 않는다.
- [ ] 명시적으로 확인된 시뮬레이터에서 실제와 같은 스윕 설정으로 실행한다. 운영 설정을 덮어쓰지 않고 시험 결과를 별도 기록한다.
- [ ] 사용자와 실물 시험을 진행한다. 서로 다른 준비 자세에서 원 잔차와 보정값의 재현성을 확인한다. 새 로봇 구동/홈 변경은 별도 허가 없이 수행하지 않는다.

## 5. 승인 전 검증 상태 (과거 기록)

직전 턴의 현재 코드 오프라인 검증:
- J6/재시도 관련 20개 테스트 통과.
- 전체 66개 실행, 오류 11건(subtest 포함), 연결 테스트 2개 건너뜀.
- 현재 Python의 qpsolvers.available_solvers는 빈 목록. 오류는 QP 단계에서 발생했다.
- 이 테스트들은 기존 구현 기준이며 새 마커 전용 설계의 통과 증거가 아니다.

계획 작성 시점에는 Git 이력/소스 비교와 본 계획 교체만 수행했다. 당시에는 새 알고리즘 구현·성능 시험·실물 오류의 최종 원인 확정을 하지 않았다. 실제 최신 실물 데이터 없이 09/08 변경 하나가 J6 실패의 유일 원인이라고 결론내리지 않는다.

## 6. 승인 후 실행 우선순위

비교 기준 고정 → 마커 전용 원 추정 → J3/J5/J6 기하 계산 → 원 기반 브래킷 → 단일 sim/고정 intrinsics → 파일 정리 → Step1.5/Step2 개별 검증 → 연결 시험.

핵심은 09/07 코드를 통째로 복원하는 것이 아니라, 09/07의 의도와 최신 사용자 원칙을 기준으로 순수 기하 계산을 복구하면서 기존부터 존재하던 의존성도 제거하는 것이다. 수학 유도가 성립하지 않는 부분은 구현 단계에서 중단·보고하며 FK나 가상 정답으로 보충하지 않는다.

## 7. 승인 후 구현·검증 결과 — 시뮬레이터 전체 실행 통과

최종 판정: **요청받은 127.0.0.1:50051 시뮬레이터의 전체 보정 프로세스 통과.** `connected-05`가 실제 SDK 이동과 운영 기본 시간으로 양팔 Step1 → 설정 사본 저장·적용 → Step1.5 → Step2 64자세 수집·최적화·결과 저장까지 완료했다. 종료 코드 0, `CONNECTED ALL-STEPS PASS` 확인. 실행 중 소스 해시와 원본 설정 보존 검사도 통과했다.

중간에 왼팔 J5 관측 1회가 품질 기준으로 거부됐고, 기존 자동 재시도에서 정상 수렴했다. 따라서 “품질 경고가 단 한 번도 없었다”는 결과는 아니며, **실패 자료를 저장하지 않고 재시도해 전체 완료한 결과**다. 원자료에 노이즈가 있으므로 모든 실행의 무재시도나 실물 정확도까지 보증하지 않는다.

### 최종 결과 요약

| 관절 | 오른팔 적용 보정(°) | 오른팔 최종 잔차(°) | 왼팔 적용 보정(°) | 왼팔 최종 잔차(°) |
|---|---:|---:|---:|---:|
| J5 | −5.37174 | −0.05922 | +2.98506 | +0.01553 |
| J6 | −2.35425 | +0.00898 | −3.50206 | −0.00173 |
| J3 | −0.47920 | −0.02755 | −0.69040 | −0.00004 |

- 모든 Step1 잔차는 0.06° 기준 이내이며 원 일치/품질 검사도 통과했다. 위 보정은 명령에 더하는 값이다. Step2에서 보고하는 물리 오프셋과 부호를 혼동하지 않는다.
- 브래킷 회전축 교점 RMS: 오른팔 0.11130 mm, 왼팔 0.10276 mm. 저장 원자료의 SDK 없는 재생에서도 확인했다.
- Step1.5: RMS 0.3140 mm, rank 9/9, 저장·적용 성공.
- Step2: 64/64자세, 21회 최적화에서 수렴, rank 21/21, 위치 RMS 0.026968 mm, 회전 RMS 0.003346°. 이는 수집 관측에 대한 피팅 잔차이지 실물 절대 정확도 인증값이 아니다.
- 가상 정답 비교는 계산 이후 진단만 수행했다. J6 제외 팔 관절의 최대 차이는 0.052757°, J6 차이는 오른팔 0.086550°/왼팔 0.005800°다. J6는 공급한 브래킷 기준에 조건부인 유효 오프셋이다.
- 최종 회귀검사 **99개 전부 통과, 실패 0, 건너뜀 0**. 마지막 두 SDK 연결 검사는 읽기 전용이며 추가 이동/홈 쓰기를 수행하지 않았다.

검증 자료:

- [최종 실행 요약 및 소스 해시 경로](/tmp/camera-calibration-verify.XvfoTZ/connected-05/summary.json)
- [전체 연결 실행 로그](/tmp/camera-calibration-verify.XvfoTZ/connected-05.log)
- [최종 99개 회귀검사 로그](/tmp/camera-calibration-verify.XvfoTZ/unit-connected-final.log)
- [최적화 결과](/tmp/camera-calibration-verify.XvfoTZ/connected-05/optimizer.json)
- 원자료·그림·설정 사본: `/tmp/camera-calibration-verify.XvfoTZ/connected-05/` 아래 `txt_dir`, `plot_dir`, `result_dir` 및 YAML 파일. 임시 폴더이므로 장기 보관이 필요하면 별도로 복사해야 한다.

사용자 `config/`와 기존 `result/`는 변경하지 않았다. 로봇 홈 오프셋 쓰기, Git commit/push, 실물 구동은 하지 않았다.

### 구현한 변경

- Step1 표본은 순서가 있는 4×4 마커 pose만 보관한다. 엔코더 각도 기반 원 피팅·FK 축 prior·encoder bracket fitting을 삭제했다.
- 원 중심/법선/반지름은 관측점에서 추정한다. 시간 순서 외적과 명령 스윕 부호로 양의 회전축 방향을 결정한다.
- J3/J5의 보정 부호에 필요한 후보 관절 축은 추가 `-15° → 0°` 스윕으로 직접 측정한다. 이 추가 스윕도 실물/가상 공통이다. J3/기존 J5는 평행도와 원 일치, v1.3 J5는 직교 관계를 검사한다.
- J6 결과는 현재 자세에서 측정한 상대 보정량이다. 누적 보정은 명령 계층에서만 수행한다. 동축 브래킷 회전과 물리 J6 영점의 분리 불가능성은 명시적으로 유지한다.
- 브래킷은 마커 좌표계에서 관측한 J4/J5/J6 회전축 선의 교점과 J5/J6 방향으로 계산한다. 고정 도구 길이만 URDF 설계값(v1.2: 126.1 mm, v1.3: 125 mm)을 쓴다. 실행 엔코더/FK로 위치를 추정하지 않는다.
- 마커 생산은 Marker_Transform의 sim 하나로 통합했다. 고정 intrinsics 파일은 마커 계층에서 관리한다. 새 운영 Python 파일 없이 보조 소스 6개를 통합/제거했다.
- Head camera zero 함수는 HeadCameraCalibrator로 이동했다. Step1.5/Step2의 기존 미지 오프셋·게이지 추정 의미는 유지했다.

### 교차 리뷰로 추가 확인한 실패 방어

- 평행하지만 중심/반지름이 다른 원을 0° 보정 성공으로 오인하지 않도록 차단.
- 원 방향이 같아도 마커 좌표계의 축 위치가 도중에 바뀌는 관측은 차단.
- 고정 손목 구조와 양립 불가능한 J5/J6 비직교 관측은 차단.
- 수렴 실패 결과의 UI 저장/적용, 카메라 실패 시 오래된 프레임 재사용, wizard 저장 실패 후 다음 단계 이동을 차단하고 실패 재현 테스트를 추가했다.
- J3의 계산기 허용 범위(-5°~0°) 안에서 승인된 결과를 UI가 다시 -3°로 잘라내던 중복 제한을 제거했다.
- core.*와 flat import 혼용으로 같은 Base 클래스가 두 번 로딩되던 경로를 통일했다. 순환 import와 RealSense 드라이버 없는 시작도 검사했다.

### 검증 환경과 실행 이력

- 기존 전역 Python에 QP 풀이기가 없어 `/tmp/camera-calibration-verify.XvfoTZ/venv`에 osqp를 설치했다. Matplotlib의 시스템/사용자 namespace 충돌도 이 임시 환경 안에서만 분리했다.
- 검증 버전: Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.9, qpsolvers 4.11.0, OSQP 1.1.3, PySide6 6.11.0, rby1_sdk 0.9.1. 기존 requirements.txt의 OSQP 고정 버전은 1.1.1이므로 이번 환경과 동일한 의존성 설치 시험으로 해석하지 않는다. 전역 환경 설치와 배포 빌드는 수행하지 않았다.
- 연결 전 자동 테스트 99개 실행: 97개 통과, 장비 연결 조건부 2개 건너뜀, 실패 0. 최종에는 SDK 읽기 연결 조건을 켜 99개 모두 통과했다. `git diff --check` 통과.
- `connected-02`: M@v1.2 시뮬레이터에서 양팔 Step1, Step1.5, Step2 64개 자세 수집 및 최적화 완료. Step1.5 RMS 0.2863 mm, Step2 위치 RMS 0.02795 mm/회전 RMS 0.00512°, 관측 rank 21/21. 단, 실행 도중 소스가 보강됐으므로 최종 코드의 통과 증거로 사용하지 않는다.
- `connected-03`: 실패. 오른팔 Step1 전체, 왼팔 J5/J6/브래킷은 통과했으나 왼팔 J3의 원 중심 차이가 0.905 mm, 재시도에서 0.613 mm로 거부됐다. 후속 Step1.5/Step2로 진행하지 않았고 실패 값을 저장하지 않았다.
- `connected-04`: 실패. 오른팔 J5/J6는 통과했으나 브래킷의 J5/J6 축 각도가 90.89361°로 거부됐다. 관절에만 반영했던 공동 원 추정이 브래킷에는 빠져 있었으며, 작은 J6 원호(반지름 약 54 mm)의 법선이 불안정했다.
- `connected-05`: 관절·브래킷 공통 원 추정과 방향 메타데이터를 반영한 전체 기본 프로세스 검증 통과. 시작 후 소스 고정 및 원본 설정 보존 해시를 마지막에 확인했다. 왼팔 J5 품질 재시도 1회 후 전체 완료.
- 이 실행의 오른팔 J6 TXT를 SDK 연결 없이 별도 재생했다. 각 프레임 16개 수치(4×4 pose)만으로 첫 보정 -2.353455°와 재측정 잔차 -0.002391°가 연결 실행과 동일하게 재현됐다.
- 사용자 설정/기존 로그/실물 홈은 보존한다. 시험 결과와 설정 사본은 위 임시 폴더에만 저장한다. scratch의 옛 알고리즘 실험 스크립트는 기록으로 남기며, 삭제한 API에 대한 호환 wrapper를 운영 코드에 재도입하지 않는다.

### 연결 실패에서 발견하고 수정한 J3/J5 원 추정 문제

기존 구현은 보정각에 A/B 원축과 관측 C 원축의 직교 관계를 사용하면서 원 중심은 독립 피팅 결과로 남겼다. 30°처럼 짧은 원호에서는 법선의 작은 오차가 외삽 중심의 큰 이동으로 나타났다. 같은 원+0.1 mm 노이즈의 독립 피팅 50회 중 12회가 0.5 mm 중심 차이 기준을 잘못 넘었다. 실패 TXT에서도 상대 원호 예측 RMS는 0.146/0.145 mm로 작았다.

수정은 BaseCalibrator.refine_adjacent_circles에서 A/B/C 원을 함께 피팅해 `nA·nC=0`, `nB·nC=0`이라는 고정 인접 관절 구조를 적용하는 것이다. C 자체도 마커 관측점에서 추정하며 로봇/FK/엔코더/명목 전역축은 입력하지 않는다. 세 중심과 세 반지름은 각각 자유롭게 추정한다. 원 일치를 강제로 적용한 최적화가 아니므로 잘못된 중심/반지름 데이터는 여전히 거부한다. Joint와 Marker가 같은 함수를 사용한다.

- 기존 0.06° 수렴, 0.5 mm 원 일치, 0.5 mm 관측 RMS, 스윕 범위/시간은 변경하지 않았다.
- 실패 블록 2 재생: 보정 -0.027708°, 중심 차이 0.457568 mm, 반지름 차이 0.429641 mm.
- 실패 블록 4 재생: 보정 +0.040545°, 중심 차이 0.191792 mm, 반지름 차이 0.030262 mm.
- 새 테스트는 0.03 mm 미만의 짧은 원호 관측 오차가 독립 중심을 1.257 mm 벌리는 실패를 먼저 재현한 뒤 수정으로 통과했다.
- 100 mm 중심 이동/다른 반지름을 성공으로 강제하지 않는 회귀검사와 양팔·v1.2/v1.3의 미지 상위 영점 시험을 확인했다. 수정 후 전체 연결 재실행은 `connected-05`에서 통과했다.
- 브래킷은 (J4,J6,J5) 순서와 명령 방향 (+,+,−)으로 같은 공동 원 추정을 사용한다. Marker 수집부가 방향을 기록하므로 내림차순 J5의 축 부호를 뒤집지 않는다. 같은 카메라/상위 관절 준비 자세에서 얻은 스윕이어야 하며, 서로 독립적으로 재교시하거나 카메라를 움직인 스윕을 섞으면 안 된다.
- `connected-04` 실패 원자료 재생: J5/J6 각도 90.00421°, 교점 RMS 0.144452 mm. 기존 회전축 직교/교점 기준 통과. 교점은 공동 원 피팅에서 강제로 맞추지 않는다. 자세 회전 불연속, 70°의 잘못된 축 관계, 20 mm 이동한 축선은 여전히 거부한다.

### 모드 경계와 검증의 의미

| 구간 | 실물/가상 공통 처리 | 허용한 차이 |
|---|---|---|
| 초기 전원 | 같은 SDK 연결·상태 확인·servo 정책 | 실물 `48v`, 확인된 로컬 시뮬레이터 `.*` |
| 마커 생산 | 같은 제공자 API, metre 단위 4×4, 같은 시간창·집계·실패 반환 | 실제 영상 검출 또는 고정 가상 오프셋으로 생성한 pose |
| Step1 | 준비 자세 → A/B 및 필요한 방향 스윕 → 관측 원 품질 검사 → 상대 보정 → 재측정/수렴 → 저장·적용 | 없음 |
| Step1.5/Step2 | 같은 SDK 이동·엔코더/마커 수집·최적화·수렴 검사 | 없음 |
| 화면/진단 | 같은 결과 의미·저장 계약 | 카메라 없는 모드에 온도/영상 생성 안 함, 가상 GT 비교는 진단 출력만 |

Step1.5/Step2에는 엔코더를 관측값으로 사용하면서 미지 영점 오프셋을 함께 추정하는 기존 모델이 남아 있다. Step1의 원 기반 계산에 엔코더를 다시 넣은 것이 아니다. 헤드 tilt와 카메라 장착 회전의 게이지, J6와 동축 브래킷 회전의 분리 불가능성은 물리 영점 정답으로 표시하거나 홈에 쓰지 않는다.

연결 시험은 Qt 화면을 offscreen으로 생성하고 운영 worker/저장 함수를 실행한다. 실제 SDK 명령·피드백과 가상 마커 생산을 검증하지만, 수동 버튼 클릭 전체·실제 영상 검출·실물 충돌/정확도·배포 패키지 빌드까지 검증한 것은 아니다. 운영 Python 모듈 6개 삭제는 Git 기준 커밋에서 복원 가능하며, 사용자 데이터 파일은 삭제하지 않았다.

기존 비교 그림의 `Angle Dev`는 두 관측 원 법선 사이 각도이다. 특히 J6에서는 약 90°가 정상이며, 이것을 J6 보정 잔차로 읽으면 안 된다. 수렴 판정값은 로그의 `relative correction`/`Step Correction`이다. `tests/replay_j6_sweeps.py`는 과거 58열 TXT 재생용이고, 새 원자료는 `# ordered T_camera_marker`로 나뉜 16열 pose 블록이다.

### 재현 명령

프로젝트 루트 `/home/jsm/camera_ws`에서 자동 회귀검사:

```sh
env PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  PYTHONPATH=/tmp/camera-calibration-verify.XvfoTZ/venv/lib/python3.10/site-packages:tests:. \
  RBY1_MODEL_DIR=/home/jsm/sdk/rby1-sdk/models/rby1m/urdf \
  /tmp/camera-calibration-verify.XvfoTZ/venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

연결 재검증에는 위 환경으로 `tests/run_connected_calibration.py --confirm-local-simulator --output <새로운_결과_폴더>`를 실행한다. 이 명령은 실제 SDK 이동을 수행하므로 127.0.0.1:50051이 시뮬레이터임을 확인하고 다른 제어기를 함께 실행하지 않아야 한다. `--sweep-seconds`/`--resume-step2`/`--only-step1`을 쓰지 않은 실행만 이 보고서의 전체 기본 프로세스 검증에 해당한다.
