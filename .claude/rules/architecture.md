---
paths:
  - "ui/**"
  - "core/**"
  - "config/**"
  - "main_ui.py"
---
# 코드 구조와 계층

## 기본 폴더 구조
```plain text
project/
├─ main_ui.py         진입점: UI 생성과 실행만
├─ ui/                사용자 인터페이스 → core 기능 호출만
├─ core/
│  ├─ comm/           통신 & 센서
│  ├─ control/        제어
│  ├─ compute/        연산
│  └─ sequence/       (보완) 여러 기능을 조합하는 작업 흐름
├─ config/            공유 변수·상수·경로 (yaml) + 로더
├─ tests/             자동 테스트
└─ scripts/           실행·재현·빌드 스크립트
```
폴더 이름은 프로젝트에 맞게 바꿔도 되지만, 분류와 책임은 아래를 따른다. 실제 폴더와의 대응은 report 2.3 아키텍처에 **파일 트리**로 적는다 (줄마다 `# 역할`, `[계층]`, 기능 번호). 기능별 배치와 이름은 detail의 기능 구성과 report의 명칭 사전에 적는다.

## 일에 따라 먼저 볼 곳
| 하려는 일 | 먼저 볼 곳 |
|---|---|
| 화면, 버튼, 입력, 결과 표시 | `ui/` (core 기능을 호출만 한다. 색·버튼·글자는 `rules/ui.md`) |
| 장비·센서 통신, 로그·패킷 파싱, 공유 자원 만들기 | `core/comm/` |
| 로봇·시뮬레이터 제어, 모션 명령, 상태 조회 | `core/control/` |
| 최적화, 기하·수치 계산, 필터, IK | `core/compute/` (I/O와 SDK 호출 금지) |
| 여러 기능을 묶는 작업 흐름, 자동 실행 시퀀스 | `core/sequence/` |
| 상수, 단위, 임계값, 경로, 장비 파라미터 | `config/` |
| 테스트, 측정, 검증 (겹치면 새 파일 말고 기존 파일을 고쳐 범용으로) | `tests/` |
| 실행·재현·빌드 | `scripts/` |

## 계층별 책임
| 계층 | 책임 | 예시 | 하지 않는 것 |
|---|---|---|---|
| **ui** | 사용자 입력을 받아 core 기능을 호출하고 결과를 표시 | 버튼 → core 함수 호출, 진행 상태·결과 표시 | 계산, 장비 직접 접근, 파일 형식 처리 |
| **core/comm** 통신 & 센서 | 외부 데이터를 받아 파싱하고 공유 자원을 만든다 | 카메라 데이터 취득, 로그·패킷 파싱, 공유 메모리 생성 | 제어 명령 결정, 최적화 계산 |
| **core/control** 제어 | 전용 SDK 기반으로 로봇·시뮬레이터를 제어 | 모션 명령, 상태 조회, 파라미터 쓰기 | 최적화 계산을 내부에 직접 작성 (compute 호출) |
| **core/compute** 연산 | QP 솔버, 최적화, 기하·수치 계산 | QP 최적화, 원 피팅, IK, 필터 | I/O, SDK 호출, UI 접근 — 데이터를 받아 결과만 반환 |
| **core/sequence** (보완) | comm·control·compute를 조합해 하나의 작업 흐름을 실행 | 캘리브레이션 단계, 자동 실행 시퀀스 | 새 계산·새 장비 접근 로직 작성 (각 계층에 둔다) |
| **config** | 여러 모듈이 공유하는 변수·상수·경로를 한 곳에서 정의·로드 | 단위, 임계값, 장비 파라미터, 경로 | 실행 결과 저장 (결과는 별도 결과 폴더) |

## 의존 방향
- 화살표 방향으로만 import한다. **core는 ui를 import하지 않는다.**
- comm과 control은 서로 직접 부르지 않고 sequence가 조합한다. (보완)
- compute는 config 외에는 의존하지 않는다. 그래서 장비 없이 오프라인 테스트할 수 있다. (보완)

## 클래스·모듈 규칙
- 기능은 **클래스 단위**로 나누고, 파일은 기능 단위로 **모듈화**한다. 한 파일에 여러 계층의 기능을 섞지 않는다.
- 클래스·모듈·config 키 이름은 report 2.5의 **명칭 사전**을 따른다. 새 이름은 먼저 명칭 사전에 등록한다.
- 장비 없이 테스트·재현할 수 있도록 comm·control은 실기 / 시뮬레이터 / 녹화 재생으로 교체 가능한 형태로 만든다. `[wbm]` (보완)
- 테스트·스크래치 코드는 프로덕션 폴더와 분리한다. 레거시 shim을 두지 않고 정식 경로로 import한다. `[cam]`

## 공유 변수와 config
- 여러 모듈에서 쓰이는 변수는 **이름·단위·기본값을 통일**하고 config에서 정의한다. 코드에 같은 값을 흩어 쓰거나 매직 넘버를 두지 않는다.
- config 폴더와 파일은 기능 구현 **전에** 먼저 만들고, 로더는 한 곳(클래스 하나)에 둔다.
- 단위를 이름이나 config 주석에 명시한다 (예: `_deg`, `_mm`). (보완)
- 공칭 기본값과 사용자가 바꾼 값을 구분하고, 저장은 원자적으로 한다. 저장 실패는 조용히 넘기지 않는다. `[cam]` (보완)
- 로컬 절대 경로를 하드코딩하지 않는다. 경로는 config 또는 OS 표준 경로에서 만든다. `[rby1]`

## 이 프로젝트의 대응 (camera_ws)
이 저장소는 `core/comm` 같은 폴더가 따로 없고, 아래 파일이 각 계층을 맡는다. 파일별 역할은 report 2.3 아키텍처의 파일 트리에 있다.

| 공통 계층 | 이 프로젝트 |
|---|---|
| ui | `main_ui.py`, `ui/` (`ui/core_bridge.py`가 core 호출 창구) |
| core/comm | `core/camera_processing.py`, `core/marker_detection.py` |
| core/control | `core/robot/` (`robot_core.py`, `motion.py`, `home_offset.py`) |
| core/compute | `core/calibration/calibration_optimizer.py`, `observation.py`, `data.py` |
| core/sequence | `core/calibration/calibration_core.py`, `core/calibration/sequences/`, 각 `*Calibrator.py` |
| config | `config/*.yaml`, `core/storage.py`(로더·저장) |

- 구조 규칙은 `tests/test_module_boundaries.py`가 검사한다.
- 예외 (2026-10-06 사용자 승인): `rby1_sdk`와 `core.robot`은 `*Calibrator.py`, `calibration_core.py` 등 여러 곳에서 직접 import해도 된다.
- 해결된 위반 (2026-10-14): `marker_detection.py`가 시뮬 센서 모델 때문에 `calibration_optimizer`를 import하던 것(comm → compute). `SimulationModel`은 `core/calibration/simulation.py`로 옮겼고, `CalibrationCore.connect_camera`가 `Marker_Transform(simulation_factory=SimulationModel.create)`로 넘긴다. `tests/test_module_boundaries.py`가 comm 모듈의 `core.calibration` import를 막는다.
- 테스트·합성 검증 스크립트는 `tests/`와 `scratch/`에 둔다. `scratch/`의 스크립트는 프로덕션 코드(`core/`)를 바꾸지 않는다.
