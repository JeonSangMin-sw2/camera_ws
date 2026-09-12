# Scratch Workspace

개발 중 임시 실험 및 일회성 디버그 스크립트를 작성하는 작업 공간입니다.

- **공식 회귀 및 계약 테스트**: [`tests/`](../tests/README.md) 디렉터리에 작성합니다.
- **과거 실험 기록**: 2026-09-10 이전의 260여 개 과거 실험/분석 스크립트는 Git 커밋 이력(`b9efde9` 등)에 완전히 보존되어 있으며, 필요 시 `git checkout` 또는 `git show`로 복원할 수 있습니다.
- **주요 검증 로직 이관**:
  - 런타임 이상 감지 및 복구: [`tests/test_calibration_anomaly_recovery.py`](../tests/test_calibration_anomaly_recovery.py)
  - 3D 기하 원 피팅 알고리즘 검증: [`tests/test_geometry_fitting_contracts.py`](../tests/test_geometry_fitting_contracts.py)
  - 관절/마커 계약 검증: [`tests/test_marker_calibrator_contracts.py`](../tests/test_marker_calibrator_contracts.py), [`tests/test_step1_head_contracts.py`](../tests/test_step1_head_contracts.py)
  - 저장 마커 관측 감사: [`tests/analyze_calibration_results.py`](../tests/analyze_calibration_results.py)
