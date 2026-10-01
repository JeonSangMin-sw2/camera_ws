## File reads

This project uses [graft](https://github.com/flyingrobots/graft) as
a context governor. Prefer graft's MCP tools over native file reads:

- Use `safe_read` instead of `Read` for file contents
- Use `file_outline` to see structure before reading
- Use `read_range` with jump table entries for targeted reads
- Use `graft_diff` instead of `git diff` for structural changes
- Use `explain` if you get an unfamiliar reason code
- Call `set_budget` at session start if context is tight

These tools enforce read policy, cache observations, and track
session metrics. Native reads bypass all of that.

---

# camera_marker_calibration 프로젝트 규칙

공통 규칙은 전역 `~/.claude/CLAUDE.md`(노션 공통 하네스 요약)를 따른다. 여기는 이 프로젝트의 추가 규칙과 예외만 적는다.
노션 프로젝트 문서: camera_marker_calibration (프로젝트 카테고리) → report / design / process / project_harness (2026-09-26 개편, 옛 Harness·References·Development Log는 Archive).
노션과 코드가 충돌하면 **현재 코드가 기준**이며, 판단이 필요하면 사용자에게 묻는다. (예: 옛 Trap 1·15 "카메라 위치 고정"은 2026-09-26 project_harness에서 "09-15부터 위치 추정"으로 정정됨)

## 프로젝트
- RB-Y1 헤드 카메라(RealSense D405/D435) 외부 파라미터와 양팔·헤드 관절 홈 오프셋 보정. PySide6 GUI.
- Python 3.11.9 (`.venv`). 실기 `192.168.1.40:50051` 모델 "a", 시뮬 `127.0.0.1:50051` 모델 "m".

## 명령
| 용도 | 명령 | 권한 |
|---|---|---|
| lint (G1) | .py 수정 시 hook이 `py_compile`로 자동 검사 | 자동 |
| 테스트 (G2) | `.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests` | ASK |
| 앱 실행 (실기 동작 가능) | `.\.venv\Scripts\python.exe main_ui.py` | ASK |
| 시뮬 시작·중지 | `scripts\windows\start_sim_wsl.bat` / `stop_sim_wsl.bat` | ASK |
| 시뮬 전 과정 검증 (G4, 시뮬 로봇 동작) | `scratch\validate_step1_2_sim.py` | ASK |
| exe 빌드 | `build_windows.*`, `build_linux.sh`, PyInstaller | **DENY** — 사용자가 직접 빌드 (2026-09-22) |

`scratch/arm_sign_test.py`, `camera_swap_test.py`, `randomized_sweep_test.py`는 시뮬에 접속해 모델만 읽고 로봇을 움직이지 않는다 (합성 관측).

## 계층 대응
| 공통 계층 | 이 프로젝트 |
|---|---|
| ui | `main_ui.py`, `ui/` (`ui/core_bridge.py`가 core 호출 창구) |
| core/comm | `core/camera_processing.py`, `core/marker_detection.py` |
| core/control | `core/robot/` (`robot_core.py`, `motion.py`, `home_offset.py`) |
| core/compute | `core/calibration/calibration_optimizer.py`, `observation.py`, `data.py` |
| core/sequence | `core/calibration/calibration_core.py`, `core/calibration/sequences/`, 각 `*Calibrator.py` |
| config | `config/*.yaml`, `core/storage.py`(로더·저장) |

구조 규칙은 `tests/test_module_boundaries.py`가 검사한다. 알려진 위반(보고만, 수정은 사용자 결정): `*Calibrator.py`와 `calibration_core.py`가 `rby1_sdk`·`core.robot`을 직접 import (계산과 흐름이 섞임), `marker_detection.py`가 `calibration_optimizer`를 import (comm → compute).

## 🔴 반드시 지킬 것
- 보정 수학을 바꾸기 전에 노션 프로젝트 문서(design 4장 원리·수식, project_harness 트랩, process 진행 기록)를 먼저 읽는다.
- Step 1 수렴 기준 0.06°를 올려서 통과시키지 않는다 (사용자가 거부한 결정).
- 카메라 내부 파라미터 모드(`calib_intrinsics_mode`)를 바꾸지 않는다. 로봇별 상수·로봇 시리얼을 yaml에 넣지 않는다. 예외: 카메라 시리얼별 내부 파라미터 파일은 허용 (2026-09-30 사용자 결정).
- 테스트·분석 스크립트는 결과 경로를 임시 폴더로 돌린다. `result/`를 덮어쓰지 않는다.
- `config/home_reset_baseline.json`은 Home Offset Reset마다 새로 써진다. `created_at`을 확인하고 같은 리셋 이후 결과끼리만 비교한다.
- Home Offset Apply는 로봇 홈에 실제로 기록된다. 이후 분석에서 오프셋을 다시 더하면 이중 적용이 된다.
- `main_ui.clear_old_plots`는 실행 시작 시 `result_txt`의 .txt를 모두 지운다. 보존할 기록은 먼저 복사한다.
- Step 2 오프라인 재계산 시 `joint_offsets_store`의 부호는 결과 JSON과 반대다.

## 안티패턴 (실제 사례)
| 날짜 | 실수 | 원인 → 지금 규칙 |
|---|---|---|
| 2026-09-09 | 실행 도중 소스를 수정한 증거 실행(`connected-02`) | 증거 실행 중 소스 수정 금지, 해당 실행은 증거 제외 |
| 2026-09-14~15 | AI가 `build_linux.sh`를 직접 재실행 | 빌드는 사용자가 한다 → settings.json DENY |
| 2026-09-15 | runaway 오탐 수정 중 폴백을 되살려 수렴을 통과시키는 안이 나옴 | 사용자와 상의해 엄격 기준 유지, 기준 완화 금지 |
| 2026-09-21 | J0 비교에서 서로 다른 리셋의 baseline을 섞음 | baseline `created_at` 확인 후 같은 리셋끼리만 비교 |

## 래칫 기록
| 날짜 | 실수·요구 | 반영 계층 | 위치 |
|---|---|---|---|
| 2026-09-26 | AI 직접 커밋 방지 | 권한 DENY | `.claude/settings.json` |
| 2026-09-26 | exe 빌드 방지 | 권한 DENY | `.claude/settings.json` |
| 2026-09-26 | 테스트는 물어보고 실행, lint는 자동 | 권한 ASK + PostToolUse hook (py_compile) | `.claude/settings.json`, `.claude/hooks/ask_tests.py` |
| 2026-09-26 | 계층 경계 (compute 순수성, comm의 로봇 제어 금지) | 센서(테스트) | `tests/test_module_boundaries.py` |
| 2026-09-28 | 파일 읽기에 권한 창이 뜸 (묶은 명령·sed) → sed 허용, 읽기는 단일 명령/Read·Grep | 권한 ALLOW | `.claude/settings.json` |
| 2026-09-30 | `start/stop_sim_wsl.bat`이 LF 줄바꿈이라 한글 줄과 `chcp 65001`에서 명령이 잘려 실행됨 (`ker`) | 저장소 설정 (체크아웃 시 CRLF 강제) | `.gitattributes` (`*.bat`, `*.cmd` eol=crlf) |

## 진행 상태
작업 중 상태는 `docs/progress.md`에 갱신한다. 노션은 사용자가 요청할 때만 정리한다.
