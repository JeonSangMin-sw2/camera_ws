# 캘리브레이션 테스트 진입점

2026-09-10: 제품 수정 전 원인 분석을 위해 백업 테스트의 검증 목적을 현재 API에 맞춰 복원했다. 현재 등록된 모듈 테스트는 **2개이며 둘 다 알려진 결함으로 실패한다**. 실패를 skip/expectedFailure로 숨기지 않는다. 제품 수리 완료를 뜻하지 않는다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_step1_head_contracts.py -v
```

로봇·시뮬레이터 서버·카메라 연결 없이 실제 계산 함수와 반복기를 호출한다. 반복기의 취득 함수만 일정한 관측으로 대체하고 로그는 임시 디렉터리에 저장한다. GUI나 실제 설정 저장을 호출하지 않는다.

| 범위 | 백업에서 확인한 원본 | 현재 대체 |
|---|---|---|
| 헤드의 SE(3) 보존 | `tests/test_head_camera_zero.py`, `tests/test_calibration_regression.py` | `test_step1_head_contracts.py::test_head_solution_preserves_stationary_marker_in_3d` |
| Step1 반복 수렴 | `tests/test_joint_retry.py`, `tests/test_legacy_j6.py` | `test_step1_head_contracts.py::test_step1_constant_absolute_j6_target_converges_to_target` |
| 런타임 이상 감지 및 복구 | `scratch/test_anomaly_recovery.py` | `test_calibration_anomaly_recovery.py` (이상 감지, 티칭 콜백, 스텝 클램핑, 직교 수렴) |
| 3D 기하 원 피팅 검증 | `scratch/test_circle_fit_bug.py` | `test_geometry_fitting_contracts.py` (3D 원 중심/반경/법선 및 Robust Outlier 억제) |
| 마커 브래킷 기하 계약 | `scratch/test_bracket_recalc.py` | `test_marker_calibrator_contracts.py` (Joint 6 결합 각도 보정, 티칭 포즈 우선권) |
| 저장 마커 관측 분석 | scratch의 실물 데이터 분석 스크립트 | `analyze_calibration_results.py` (분석 도구, 추가 모듈 테스트 아님) |

백업은 `backup_twisted_20260910`, 커밋 `ccfd79136dbd5f3a247b2d3ed748755e7c6fe4fa`다. `test/`가 아니라 `tests/`에 18개 파일(테스트 파일 13개, 지원/실행/fixture 5개)이 있다. 작업 시작 시 현재 `tests/`에는 `.py` 없이 `__pycache__`만 있었다.

백업 전체는 `SimulationModel`, `camera_forward_zero`, `FullAutoCalibrationService`, `legacy_j6`, 설정 저장 서비스 등 현재 없는 API에 의존한다. 이를 통째로 복원하거나 백업 제품 코드를 함께 덮어쓰지 않는다. 이번에는 두 검증 목적만 이식했다. 백업의 나머지 테스트는 원래 커밋에 보존돼 있으며 현재 호환/통과로 간주하지 않는다.

```powershell
# 백업 원본 확인
git show ccfd791:tests/test_head_camera_zero.py
git show ccfd791:tests/test_joint_retry.py

# 고정된 실물 데이터의 관측 통계 재생
.\.venv\Scripts\python.exe -X utf8 tests/analyze_calibration_results.py --root reports/step1_head_audit_20260910/snapshot --output reports/step1_head_audit_20260910/marker_audit.json
```

`scratch/test_head_camera_calibrator_offline.py`는 이름과 달리 서버 연결과 전원/서보 명령을 호출하므로 이 모듈 테스트의 대체물이 아니다. 기존 scratch 파일은 삭제하지 않았으며, 이번 범위의 공식 진입점은 위 명령이다. 전체 scratch 257개 파일의 일괄 이관은 하지 않았다.

분석: [수정 전 진단 보고서](../reports/step1_head_audit_20260910/report.md). 개발 기준: [캘리브레이션 지침](../docs/calibration-development-guidelines.md).
