# 진행 상태 (로컬)

AI가 작업 중에 갱신하는 파일입니다. 노션 정리를 요청받으면 이 파일과 대화 보고를 근거로 노션 Development Log·Status를 갱신합니다.
최신 항목이 위에 옵니다. 항목 양식은 맨 아래를 참고합니다.

## 재개 지점
- 브랜치 `compute_jsm`, 마지막 커밋 `39c2083` (2026-09-20). 미커밋 변경 36건 (2026-09-26 기준).
- 노션은 2026-09-26에 4문서 구조(report / design / process / project_harness)로 개편됐고, 09-21~22 작업도 process 진행 기록에 옮겼다.
- 2026-09-30 노션 동기화: 09-28~09-30 항목을 process 진행 기록 13행 + 재개 지점·일정·남은 과제, design(Step 2 모델·파라미터 표·D405 사실), project_harness(T31 Apply 기준 소실, T32 약한 방향, T33 시뮬 재초기화, G2 기록, 래칫 3행)에 반영. 다음 노션 정리는 이 파일의 2026-09-30 이후 항목부터.
- 2026-09-30 저녁 노션 추가 동기화: UI 개편(진행 중)·setting.yaml 브라켓 v1.3 값(미해결) 2행, process 재개 지점, design 화면 구성·시리얼 내부 파라미터. 다음 정리는 이 뒤부터.
- 내일(10-01) 재개: ① UI 개편 G2 실행 ② 실제 화면 확인(상단 탭 크기, 위자드 1-1·1-2, 카메라 탭) ③ setting.yaml 브라켓 값 v1.3 원인 확인·복원 결정 ④ 손목 다양화 자세 실기 확인
- 결정 대기 목록은 노션 process 4.1에 있다 (왜곡 계수 검증 방법, J0 공통모드 댐핑 존폐, 구조 위반 처리, Python 기준 버전).
- 2026-10-06 밤 노션 동기화 (사용자 요청): 이 파일의 2026-10-06 항목 전부(모니터 창, 점검 보고서, 다른 로봇 1·2차, 실행 기록, v1.3 시뮬, 스윕 파일 이름)를 process 진행 기록 7건과 4.8 현황에, 결정은 cps C20~C23에, harness(G2 기록, 결과 파일 계약, T10·T12·T27, T34 신규, 5장 예외), report(상단 결정 대기, 결과 경로, Step 1 순서 설명), detail(한계 2줄)에 반영. 볼트(옵시디언)는 수정하지 않음. 다음 노션 정리는 이 뒤부터.
- 지금 결정 대기: v1.3 J5 연산의 구체안 (cps C23, process 4.8), v1.3 J6를 브라켓 스윕에서 구할지.
- 2026-10-16 밤 노션 동기화 (사용자 `/record`): 이 파일의 10-08~10-16 항목을 process 진행 기록 19건과 4.10 현황(시스템 확정)에, 결정은 cps C24~C31에, harness(원칙 4 카메라, T4/T18, T34 상태, T35~T39 신규, G2 324·G4, 5장 예외 해결, 안티패턴 3행, 결과 JSON `camera_mount_constraint`), detail(4.5.1 v1.3 브라켓, 4.5.2 설계 위치 모드·광축 회전, 4.5.3 표, 4.8, 4.9 파라미터 4행), report(상단 현재 상태, 결정 대기 3건 결정됨, 2.1 Python·시뮬·카메라, 2.3 파일 트리, 2.4 v1.3)에 반영. 저장소 `CLAUDE.md` 안티패턴 3행 추가. 볼트(옵시디언)는 수정하지 않음. 다음 노션 정리는 이 뒤부터.

---

## 2026-10-27 — Python 3.12 실기 확인, README 문구
- 오늘 실기 실행은 모두 `.venv` Python 3.12.10(패키지: rby1-sdk 0.10.0, pyrealsense2 2.59.1, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, PySide6 6.12.0, opencv-contrib-python-headless 4.13.0.92, qpsolvers 4.11.0, osqp 1.1.1). README 패키지 목록 문구 "offline tests pass" → "Tested with this set (… RB-Y1 v1.2 robot …)". 3.12 exe 빌드는 미확인(사용자)
- 노션 기록 요청 → `record` 스킬은 모델 실행 금지 설정이라 사용자 `/record` 입력 필요

## 2026-10-27 — 19:06 실행(69자세, 촬영 1 s): 사용자 "제일 잘된 것 같다"
> 검증 수준 🟢 v1.2 로봇 1대(모델 a), 사용자 눈 판정 + 이전 실행과 비교 · 백업 `archive/runs/20261027_skiphalf1s_v12_robot/`
- 전체 18분 32초(처음 20분), Step 2 수집 4분 26초(5분 39초). pan −0.66°, J1 오른쪽 +3.51 / 왼쪽 −2.59°, 잡음 추정 0.72° / 0.49 mm
- 확인 자세에서 다른 실행 결과와의 차이(19:06 결과로 명령 시): 16:54 0.4 / 0.5 mm, 0.20 / 0.10° · 14:42 0.5 / 1.0 mm · 17:41 1.0 / 2.1 mm · 18:32(0.5 s) 2.3 / 1.0 mm
- 현재 v1.2 설정: 손목 다양화 자세 100% 켜짐, `skip_half_steps: true`, `sample_time_s: 1.0`, 엘보우 하한 0.12°·한도 −4°, Step 2 후 검증 init 재이동 없음
- 이 결과가 의미하지 않는 것: 로봇 1대·같은 리셋에서의 결과. v1.3과 헤드 없는 로봇은 이 설정으로 실기 미검증

## 2026-10-27 — 18:32 실행(69자세, 촬영 0.5 s): 덜 맞음 → 촬영 1.0 s로 복귀
> 검증 수준 ⚪ (오프라인 비교) · 적용 안 함 · 백업 `archive/runs/20261027_skiphalf_v12_robot/`
- Step 2 수집 5분 39초 → 3분 52초, 전체 18분. 사용자: J1이 고르게 안 벌어짐, 대칭 자세에서 오른손 +, 왼손 − 회전
- 16:54 대비 확인 자세(`archive/reports/v13_step2_20261023/skiphalf_run_check.py`): 오른손 y −2.5 mm, z축 회전 −0.79°, 왼손 +0.50°. J1 오른쪽 +3.57 / +3.58°, 왼쪽 −2.23 / −2.56°. pan −0.48 / −0.63°
- 원인: 기본 자세 위치 잔차 0.39 → 0.66 mm(회전 0.38° 그대로) → 0.5 s 촬영. 16:54 데이터를 같은 69자세로 풀면 원래와 0.3 mm·0.06° 이내 → 자세 줄이기는 영향 작음
- **수정 (사용자 "1초로")**: `setting.yaml` `step2.sample_time_s: 1.0`(`skip_half_steps: true` 유지), `load_step2_plan_options` docstring에 0.5 s 결과 기록

## 2026-10-27 — Step 2 절반 크기 단계 제거 옵션 + 촬영 0.5 s
> 검증 수준 🟠 (오프라인 재풀이 + G2) · 실기 미검증
- 헤드 없는 실기 데이터 2건(10-02, 10-06, 54자세)을 지금 코드로 재풀이(`archive/reports/v13_step2_20261023/step2_headless_subsets.py`): 54 → 36자세에서 확인 자세 손끝 0.09~0.24 mm, 관절 최대 0.07°
- **수정 (사용자 "1번 옵션, v1.2·1.3 다, 촬영 0.5초")**: `data.load_step2_plan_options()` (`step2.skip_half_steps` 기본 false, `step2.sample_time_s` 기본 1.0), `motion.build_incremental_motion_plan(skip_half_steps=)` (J0 헤드없음·J1/J2/J4·RPY·XYZ의 y/z·헤드 pan/tilt의 절반 크기 단계 제거, 헤드 64 → 44 / 헤드없음 54 → 36), `collection.py`와 `main_ui` Step 2 탭 두 곳에 적용, 수집 촬영 시간을 설정값으로. `setting.yaml`에 `skip_half_steps: true`, `sample_time_s: 0.5`. 테스트 3개. G2 통과 336 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_step2_plan_options.log`)
- 이 결과가 의미하지 않는 것: 촬영 0.5 s의 영향(잡음)은 오프라인으로 확인 불가, v1.3은 이 자세 줄이기를 데이터로 확인하지 않음(같은 계획 구조라 적용만). 검증 단계의 촬영(1 s)은 그대로

## 2026-10-27 — 검증 init 자세 재이동 제거 실기 확인(17:41), Step 2 자세 줄이기 오프라인 검토
> 검증 수준 ⚪ (오프라인 재풀이) · 백업 `archive/runs/20261027_noreadymove_v12_robot/`
- 17:41 실행: 검증이 init 자세로 다시 가지 않고 보정 자세로 바로 이동, 검증 21 s → 9 s. 보정 자세 카메라 대칭 0.63 mm / 0.29°. pan −0.46°(16:54 −0.63°, 14:42 −0.62°). 확인 자세 눈 판정은 사용자 보고 대기
- Step 2 자세 부분집합 (`archive/reports/v13_step2_20261023/step2_pose_subsets.py`, 16:54·17:41 데이터, 전체 89자세 결과 대비 확인 자세 손끝): 절반 크기 단계 20개 빼기(64 → 44) + 손목 24 → 0.25~0.33 mm(약 1분 15초 절약), 기본 1/2 + 손목 24 → 0.41~0.56 mm, 기본 1/3 + 손목 24 → 0.58~0.77 mm, 손목 1/2 → 0.33~0.59 mm(손목 자세 줄이면 영향 큼). 두 실행끼리 차이 약 1 mm
- 손목 자세 하나씩 빼기 분석(`wrist_pose_contribution.py`)은 위 결과로 대체, 중단
- 결정 대기: 절반 크기 단계 제거 반영 방식(설정 옵션 / 기본 변경), 자세당 시간(이동 1.8 s, 안정 0.4 s, 촬영 평균 1 s) 단축은 실기 시험 필요

## 2026-10-27 — 손목 100% 재실행(16:54) 재현 확인, Step 2 후 검증의 init 자세 재이동 제거
> 검증 수준 🟠 (오프라인 비교 + G2) · 백업 `archive/runs/20261027_wristfull2_v12_robot/`
- 16:54 실행 pan −0.63°(14:42는 −0.62°), 두 결과의 확인 자세 손끝 차이 0.5 / 0.7 mm. 사용자: 눈으로도 맞음. 다른 설정 대비 2~4 mm
- 시간(20분): Step 1 팔당 약 5분 45초(엘보우 6~7회 2분 24초~2분 47초, 브라켓 2차 측정 약 1분), Step 1.5 2분 12초, Step 2 수집 5분 39초(89자세, 손목 24자세 약 1분 40초), 계산·검증 40초. 브라켓 2차 측정은 매번 z +0.13~0.55 mm, J6 −0.12~−0.40°로 같은 방향 보정 → 유지
- **수정 (사용자 "init pose 가는 것도 빼줘")**: Step 2 수집 계획의 마지막 단계(restore_baseline)가 이미 init 자세라 검증이 같은 자리로 한 번 더 이동하던 것. `collection.py` 샘플에 `baseline` 표시, `calibration_core`가 마지막 샘플이 baseline이면 `run_post_step2_verification(reference=...)`. 검증은 그 오른팔 관절을 왼팔에 거울로 옮긴 대칭 자세를 기준으로 오프셋 보정 자세로 바로 이동(그 자세의 카메라 vs 모델 비교는 빠지고 보정 자세 측정은 유지), 팔 간격 안전 검사는 현재 실제 자세 기준. 저장된 샘플로 푸는 경우는 기존대로 이동. 테스트 `tests/test_post_step2_verification.py` 3개. G2 통과 333 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_verify_no_ready_move.log`). 실기 미확인

## 2026-10-27 — 손목 자세 60% 실행(16:01)에서 다시 오른손 약 3 mm 뒤 → 100% 범위로 되돌림
> 검증 수준 ⚪ (오프라인 재풀이) · 적용 안 함 · 백업 `archive/runs/20261027_elbow012_v12_robot/`
- 엘보우 하한 0.12° 적용: 오른쪽 영점 근처 추정 5개 −3.58, −3.45, −3.54, −3.58, −3.63 → −3.548, 왼쪽 4개 → −2.892. 잡음 추정 0.37° / 0.39 mm
- Step 2 pan: 13:55 −0.23(손목 없음, 오른손 3 mm 뒤) / 14:42 −0.62(손목 100%, 잘 맞음, 손목 빼면 −0.44) / 16:01 −0.41(손목 60%, 오른손 3 mm 뒤 + 약간 돌아감, 손목 빼면 −0.39). 60%에서는 손목 자세가 pan을 거의 안 움직임
- 판단: 좌우 앞뒤는 pan(약 7 mm/°)이 결정, 카메라는 pan 오차를 못 봄. 100% 손목 자세만 pan을 눈과 맞는 쪽으로 옮김(1회 관찰). 근본 해결은 몸통 고정 기준 마커(외부 기준) — 제안만
- **수정 (사용자 "1번, 100%로")**: `ready_poses.yaml` 원래 목록 복원(커밋본과 목록 동일, 주석 3줄만 추가), 테스트 한도 25/20/30 복원. G2 통과 330 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_wrist_div_full.log`)

## 2026-10-27 — 실행 간 흔들림: 엘보우 추정 잡음 하한 0.05 → 0.12°
> 검증 수준 🟠 (세 실행 오프라인 분석 + G2) · 실기 재확인은 사용자
- **두 실행(13:55, 14:42 기본 64자세) 비교** (`archive/reports/v13_step2_20261023/run_to_run_j6.py`): 확인 자세 오른손 3.0 / 왼손 1.3 mm 차이(앞뒤 0.1 mm). Step 1 값 몫 3.1 / 1.2 mm, Step 2 데이터 몫 0.1 / 0.1 mm. J6를 Step 1에 고정(v1.3 방식)하면 3.7 / 1.8 mm, 잔차 1.110 / 0.953σ로 더 나쁨 → v1.2는 현행 유지
- **엘보우** (`elbow_inputs_per_run.py`, `elbow_axis_angle.py`): FK로 쓴 카메라 자세는 세 실행 동일(0.02 mm, 0.02°). 영점 근처 J2/J4 원 중심이 0.9~3.9 mm 어긋남 → 스윕당 추정 흔들림 약 0.11~0.12°(실행 내 통합). 수렴 판정 하한 0.05°라 추정 2~3개로 수렴, 실행 평균 −3.50 / −3.67 / −3.40°. J2/J4 축 사이각 독립 추정도 오른쪽 −3.2~−3.5°, 왼쪽 약 −2.7°(엘보우 3° 초과는 실제로 판단, 사용자 수용)
- **수정 (사용자 "0.12로")**: `JOINT_ESTIMATE_NOISE_DEG["elbow"] = 0.12` (기준 0.06°·상한 8회 그대로, 더 엄격) → 영점 근처 추정 4개 이상 필요. 테스트 `test_elbow_needs_four_agreeing_sweeps`. G2 통과 330 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_elbow_noise_012.log`)
- 이 결과가 의미하지 않는 것: 세 실행의 평균 차이(표준편차 0.14°)는 잡음만의 예상(약 0.07°)의 두 배 → 백래시 등 실행 의존 요인이 남을 수 있음. 남으면 왕복 스윕 검토

## 2026-10-27 — 손목 다양화 자세 실기 첫 사용(14:42), 범위 60%로 축소
> 검증 수준 ⚪ (오프라인 재풀이) · 백업 `archive/runs/20261027_wristdiv_v12_robot/`
- 89자세(손목 24개). 사용자: 확인 자세 잘 맞음, 다만 움직임이 큼. Step 2 pan −0.62°(3 mm 설명에 필요했던 −0.62~−0.70°와 일치), 잡음 추정 0.30° → 0.64°, 0.20 → 0.38 mm
- 오프라인(`archive/reports/v13_step2_20261023/wrist_range_subsets.py`): 손목 자세를 빼고 이번 데이터로 풀면 pan −0.44°, 확인 자세 차이 약 1 mm → 손목 자세의 몫은 약 1 mm. 같은 리셋 세 번의 pan −0.70 / −0.23 / −0.44° → 실행 간 확인 자세 흔들림 약 ±2 mm(추정). 손목 자세 잔차 1.06 mm / 2.04°(기본 0.35 mm / 0.39°), 70·79·82번 3.5~8.6°. 24개 중 21개가 범위의 85~99%
- **수정 (사용자 "2번, 60%")**: `ready_poses.yaml` v1.2 `step2_wrist_diversity` 손목 변화량 ×0.6(J4 ±15 / J5 ±12 / J6 ±18°), 헤드 그대로, 원래 목록은 주석. 테스트 한도 15/12/18. G2 통과 329 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_wrist_div_60pct.log`). 줄인 자세의 시야·충돌은 오프라인 재확인 안 함

## 2026-10-27 — 엘보우 −4° 이후 재실행(13:55): 확인 자세에서 오른손이 약 3 mm 뒤
> 검증 수준 ⚪ (오프라인 재풀이) · 적용 안 함 · 백업 `archive/runs/20261027_1355_v12_robot_elbow4/`
- 엘보우 오른쪽 −3.74°(그대로 저장, 같은 리셋의 1회차 −3.48°). Step 2가 J6를 오른쪽 −2.25 → −0.46°, 왼쪽 +3.62 → +1.10°로 바꿈. pan Step 1.5 −3.77°, Step 2 −0.23°(1회차 −0.70°)
- 후보별 확인 자세 R−L 앞뒤(관찰 −3 mm) / 잔차(0.813σ): pan −0.70 → −3.1 mm / 0.999σ, pan −0.45 → −1.5 / 0.865, 오른 J3 −3.0 → −0.4 / 0.804, −3.48 → −0.2 / 0.801, J6 Step 1 고정 −0.9 / 1.110, 카메라 x ±3 −0.2 / 0.810, 카메라 자유 −1.0 / 0.894 (`archive/reports/v13_step2_20261023/check_pose_candidates.py`)
- 결론: pan은 3 mm를 설명하지만 데이터가 거부, 엘보우는 앞뒤에 영향 거의 없음(좌우 y에 영향). 남은 가설: 64자세가 못 보는 J4/J6 조합(약한 방향 RJ6/RJ4/LJ4/LJ6, R−L −2.5 mm/단위, +0.14σ/단위, T32)
- **조치 (사용자 "켜줘")**: `setting.yaml` `step2.wrist_diversity_poses: true` (실기 첫 사용). 확인: 로그 `Step 2 wrist-diversity poses: N (step2.wrist_diversity_poses on, v1.2)`

## 2026-10-27 — v1.2 실기(모델 a, Python 3.12) Full Auto 결과가 안 맞음: 엘보우 클램프
> 검증 수준 🟠 (실행 로그 + 오프라인 재풀이) · 아직 적용 안 함 · 백업 `archive/runs/20261027_1258_v12_robot_py312/`
- **원인 (확인)**: Step 1 오른팔 J3가 −3.484°로 수렴했지만 `main_ui.handle_full_auto_joint_finished`의 `np.clip(recommended, -3.0, 0.0)`이 −3.000°로 저장(로그 경고 없음). Step 2는 J3를 Step 1 값 ±0.05°에 묶으므로 그대로 결과에 들어감. 리셋 때 엘보우를 +5.5°까지 뒤로 보내 처음으로 −3°를 넘음
- **영향 (오프라인 재풀이, `archive/reports/v13_step2_20261023/elbow_clamp_effect.py`)**: 실행 결과를 그대로 재현한 뒤 J3 −3.484로 다시 풀면 Step 2 잔차는 같고(0.816σ) 오른팔 J0/J2/J4/J6가 각각 −0.7/+0.7/−0.8/+0.9° 바뀜. v1.2 확인 자세에서 오른손 3.6 mm(y +3.2, z +1.6), 왼손 1.6 mm 이동 → Step 2가 J3 오차를 다른 관절로 흡수해 64자세는 맞추지만 확인 자세에서 어긋남
- 그 밖 (영향 미확인): 왼팔 J6 방향법·위치법 차 0.62/0.88°, Step 2에서 왼팔 J6 +3.71 → +1.32°, 왼팔 브라켓 J4–J6 축 88.5°, 카메라 x·z가 Step 1.5·Step 2 모두 범위 끝(+1.0/+0.5 mm)
- 사용자 지적: 엘보우를 끝까지 밀어도 실제로 3° 넘게 젖혀질 것 같지 않음 → −3.48°는 Step 1 측정 하나에만 근거, Step 2는 −3.0과 −3.48을 구분 못 함(잔차 동일). 엘보우가 원인이라고 확정하지 않음
- **수정 (사용자 "−4도까지로")**: `np.clip(recommended, -4.0, 0.0)`, 잘리면 `[WARN]` 로그. G2 통과 329 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_elbow_clamp_4deg.log`)

## 2026-10-27 — 리셋 자세 검사 방향 반대·한국어 고정 문구 수정
> 검증 수준 🟠 (G2 + 저장된 실제 리셋 자세 9건 재검사) · 실기 재확인은 사용자
- **증상 (사용자, 3.12 첫 실행)**: 영어 UI인데 한국어 경고, 오른쪽 엘보우를 뒤로(+5.6°) 보내자 "음수(-) 방향(뒤쪽)이어야" 거부
- **원인 (확인)**: `validate_home_offset_joint_limits`의 부호 반대 + 한국어 하드코딩. 실제 리셋 기준 파일 9건 모두 오른쪽 J1 +0.96~+3.24, 왼쪽 J1 −1.09~−2.67, J3 +1.62~+5.64 → 5° 여유 안이라 통과해 왔고 오늘 J3 +5.64에서 처음 걸림. 기존 테스트도 반대 방향을 정상으로 고정하고 있었음
- **수정**: 오른쪽 J1 < −5, 왼쪽 J1 > +5, J3 < −5일 때만 거부. 문구는 `tr("dialogs.home_reset_posture.*")`(i18n.yaml en/ko). 아무 관절도 리셋하지 않은 거부는 "finished, but some joints failed" 대신 문구만 표시(`main_ui.on_home_offset_reset_finished`). 테스트 3개 교체·4개 추가, G2 통과 329 / 0 / 0 / 0 (`archive/test_logs/g2_20261027_reset_posture_signs.log`). 기록 9건 재검사 모두 통과
- 발견 (미수정): 거부된 리셋도 `config/home_reset_baseline.json`을 먼저 덮어씀(오늘 12:13 기록, 실제 리셋 없음) — `calibration_core` 리셋 경로가 기준 저장 후 검사. 다음 리셋 성공 때 다시 써짐
- 배포 주의: 실행파일은 `config/`에 없는 파일만 복사하므로, 이전 exe 폴더의 `config/ui_config/i18n.yaml`은 새 키가 없어 키 이름이 그대로 보임

## 2026-10-26 — Python 3.12 확인, README를 3.12 기준으로
> 검증 수준 🟠 (Windows 3.12에서 G2 + 오프라인 재풀이) · GUI·실기·exe·Linux·Jetson은 3.12 미검증
- `py install 3.12` → 3.12.10 (Python 설치 관리자가 `py help` 실행 때 26.4로 자동 업데이트됨). 세션 임시 폴더에 별도 venv, `pip install -r requirements.txt` → 최신판 설치: numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, PySide6 6.12.0, pyrealsense2 2.59.1 (`.venv` 3.11은 numpy 2.4.6, scipy 1.17.1, PySide6 6.11.2, pyrealsense2 2.58.4)
- G2 통과 328 / 0 / 0 / 0 (`archive/test_logs/g2_20261026_py312.log`). v1.2 저장 결과 재풀이 차이 0.000°, v1.3 재풀이도 3.11과 같은 값
- PyPI: 3.12용 설치 파일이 Windows·Linux x86_64·aarch64 모두에 있는 버전 — rby1_sdk 0.10.0, pyrealsense2 2.59.1(aarch64는 cp39·cp310·cp312만), osqp 1.1.1. PySide6 aarch64는 6.8.3 이후 glibc 2.39 필요(6.7.2는 2.31) → 22.04 Jetson은 pip가 옛 PySide6를 고름(미검증). deadsnakes jammy python3.12 3.12.15 게시됨(arm64 여부는 페이지에 표시 없음)
- README: Python 3.12, 시스템 `python3`를 바꾸지 말 것, Jetson은 `apt-cache policy python3.12` 먼저, 테스트한 패키지 버전 목록, Jetson PySide6 버전 차이, 3.11 `.venv` 재생성 안내
- **3.12 고정 (사용자 "그냥 3.12로 고정")**: `setup.py` `python_requires=">=3.12,<3.13"`, `build_windows.bat`·`build_windows.ps1`은 3.12만 찾음, `build_linux.sh`·`run_linux.sh`는 3.12만 허용(24.04 주석 정리), README의 3.11 `.venv` 안내 삭제
- README Ubuntu 1단계를 하위 단계로: PPA 추가 + `apt-cache policy python3.12` 확인 → 설치 → `python3 --version`(3.10 유지)·`python3.12 --version` 확인, `update-alternatives`·`ln`·`sudo pip` 금지, 이미 바뀌었으면 되돌리는 명령
- **`.venv` 교체 (사용자 "제안 순서대로 진행")**: 기존 3.11.9 `.venv` → `.venv_py311_backup`(이름만 변경, `.gitignore` 추가), 새 `.venv` 3.12.10에 `requirements.txt` + `pip install -e .` (`archive/test_logs/pip_venv312_20261026.log`). G2 통과 328 / 0 / 0 / 0 (`archive/test_logs/g2_20261026_venv312.log`). CLAUDE.md Python 줄 갱신
- 남은 것 (사용자): 새 `.venv`로 GUI 실행·실기 확인, exe 빌드. 문제 시 `.venv`를 지우고 `.venv_py311_backup`을 `.venv`로 되돌리면 원상 복구

## 2026-10-26 — README 간소화 (사용자 지시)
- PowerShell 안내 삭제(Windows는 `.bat`·명령 프롬프트), Ubuntu 22.04 기준, Python 3.11 고정(3.12 미검증), 카메라 D405 권장·D435, 성능은 대칭 확인 자세 손끝 약 1~3 mm만, 성능 저하 요인 NOTE(헤드 틀어짐·밝기·렌즈 왜곡), 상단 문의처 rby.support@rainbow-robotics.com, v1.2↔v1.3 설정 이동·v1.3 차이 절 삭제
- 지원 OS에 "Ubuntu 24.04 미지원" 명시. SDK는 `requirements.txt`·`setup.py`를 `rby1_sdk>=0.10.0`으로 README(0.10.x 이상)에 맞춤 (설치본 0.10.0)
- 남은 결정: Jetson(aarch64)은 PyPI `pyrealsense2` 2.59.1(최신)에 aarch64 휠이 cp39·cp310·cp312만 있고 cp311이 없어 3.11 고정과 충돌 (2026-10-26 PyPI 확인)

## 2026-10-26 — README 갱신 (v1.3 실기 반영)
- v1.3 "시뮬만 검증" → "로봇 1대 검증", v1.3 차이점 항목 추가(브라켓 각도 설계값, J6 Step 1, D405에서 J5는 Step 2), Step 1.5 언급, `cd rby1-calibration`(clone 주소는 그대로), `[!INFO]` → `[!NOTE]`, 오타, 수동 빌드에 `pip install pyinstaller`
- 오프라인 분석 (코드 변경 없음): J4 축을 J5 축 둘레로 재는 J5 방식은 pass 간 0.01~0.14°로 재현되나 Step 2와 0.6~1.0° 차이(`archive/reports/v13_step2_20261023/j5_from_j4_about_j5.py`). 현행(Step 2가 J5) 유지. 고아 함수 22개 목록만 확인, 정리는 보류

---

## 2026-10-23 — v1.3 브라켓 회전 전부 설계값 고정, J5는 Step 2에서 풂
> 검증 수준 🟠 (실기 데이터 오프라인 재풀이) + G2 · 시뮬 G4 미검증
- **실기 (2026-10-26 사용자 보고)**: 수정 후 v1.3 보정이 잘 됨. 로그·수치는 받지 않음 (사용자 관찰만)
- **근거 (확인)**: 같은 J6 스윕에서 원(위치) 법선과 마커 회전축이 브라켓 스윕 4.7~5.5°, J5 단계 5.4~7.4° 어긋남(J4는 약 1°). Step 1 J5(J4–J6 원 법선 각)와 브라켓 yaw가 모두 이 법선에 기댐. J5 단계 각: 원 89.1~90.0°, 회전축 85.1~85.7°(차 약 4°, 매 반복 일정)
- **재풀이 (10-23 14:41 데이터, J6 Step 1 고정)**: 브라켓 yaw −90°+J5 Step 2 → 위치 0.29/0.27 mm, 회전 0.16/0.16°, 남는 일정 회전 없음(이전 0.71/2.66°). J5 Step 1 → Step 2: 오른쪽 −4.37 → −2.21, 왼쪽 −3.12 → +0.22°(결과 부호). J1 오른쪽 +0.41, 왼쪽 −1.15° → 영점+오프셋 미리보기 URDF 한계 안. 약한 방향 최소 0.081σ/단위(실행 때 0.037)
- 원인 (추정): J6 원이 카메라를 정면으로 보는 평면이라 깊이 오차에 평면 기울기가 민감. 회전축 기반 J5(오른쪽 약 0.4, 왼쪽 약 −1.5°)는 Step 2와 1.4~1.8° 달라 미해결
- **수정 (사용자 "브라켓 회전 전부 고정, J5는 (가)")**: `MarkerCalibrator.compute_marker_bracket_from_orthogonal_sweeps` v1.3 yaw도 설계값(`yaw_measured_e`로 로그만), `step1.py` 로그 1줄. `QPCalibrationOptimizer(anchor_joint5=)`, `step2.py` `solve_j5 = 범위 모드 and v1.3`, 결과 JSON `joint5_step1_deg`, `format_mount_constraint(j5_solved=)`. 테스트 4개 추가·1개 수정. G2 통과 328 / 0 / 0 / 0 (`archive/test_logs/g2_20261023_v13_bracket_design_j5_step2.log`). 분석 `archive/reports/v13_step2_20261023/`
- 이 결과가 의미하지 않는 것: 브라켓 회전이 실제로 설계값인지는 사용자 판단(CAD=실물 전제). Step 1 J5 단계는 그대로 돌고 그 값은 준비자세·Step 2 시작값에만 쓰임. 시뮬·실기 미검증
- 트랩 후보 (사용자 확인 후 harness 반영): 실기 J6 원 법선이 마커 회전축과 5~7° 어긋남 → 원 법선 기반 v1.3 브라켓 yaw·J5 신뢰 낮음

---

## 2026-10-23 — v1.3 실기 Full Auto 완료, Step 2 결과 이상 → v1.3 Step 2는 J6를 Step 1 값에 고정
> 검증 수준 🟠 (실기 데이터 오프라인 재풀이) + G2 · 실기 재확인은 사용자
- **실행** (14:41, 백업 `archive/runs/20261023_1441_v13_robot_full/`): J5 단계 12 s로 양팔 수렴(오른팔 7회, 왼팔 4회). Step 2 결과 왼팔 J1 +3.13, J2 −2.46, J4 +6.02, J6 +3.71°(Step 1 −0.06°), 대칭 자세 앞뒤 −8.2 mm. Zero+offset 미리보기가 "Failed to move robot with offset pose"로 실패
- **원인 1 (확인)**: 미리보기가 왼팔 J1을 −3.13°로 보냄. v1.3 URDF 왼팔 J1 하한 −1.0°(오른팔 J1 상한 +1.0°). 이동 부호(−offset)는 v1.2와 같은 코드이며 버전 분기 없음
- **원인 2 (확인)**: 카메라 설계 범위 모드에서 Step 2가 J6를 풀게 돼 있음(10-08, v1.2 기준 결정). v1.3 J6(손목 roll)는 Step 1에서 직접 잼. 같은 데이터로 J6를 Step 1 값에 고정하면 왼팔 J1 −0.59, J2 −0.15°, 위치 잔차 양팔 0.43 mm. 대신 마커에 포즈와 무관한 회전 차이가 남음(왼팔 2.7°, 오른팔 0.7°, 흩어짐 0.1° 이하) → 브라켓 회전 오차 모양
- **QP 범위 (확인)**: 관절 오프셋 범위는 코드대로 적용됨: J0·J1·J2·J4 ±15°, J3·J5·(고정 시)J6는 Step 1 값 ±0.05°, 헤드 pan ±15°·tilt ±2°. J1 방향 제한은 Step 2에 원래 없음(리셋 전 자세 검사뿐). URDF 관절 한계(`q_lower/q_upper`)는 읽기만 하고 범위에 안 씀
- **브라켓 회전**: yaw −90° 고정·J6 고정으로 풀면 데이터가 원하는 추가 회전(ee 기준) 오른팔 x −0.13, y +1.01, z −2.29°, 왼팔 x +0.35, y +1.55, z +3.81°. 같은 J6 스윕에서 마커 방향 회전축과 위치 원 축이 4.7~5.5° 어긋남 → Step 1 yaw(원 축 기반)는 신뢰 낮음. 미해결
- **수정 (사용자 "a로 해보고")**: `step2.py` `solve_j6 = mount_constraint is not None and 버전 != "1.3"`, 두 패스 `anchor_joint6=not solve_j6`. `format_mount_constraint(..., j6_solved=)` 로그 문구. 테스트 `test_v13_holds_j6_at_step1_with_a_constraint`. 이번 데이터 재풀이가 J6 고정 결과와 같음. G2 통과 325 / 0 / 0 / 0 (`archive/test_logs/g2_20261023_v13_step2_j6_held.log`)
- 이 결과가 의미하지 않는 것: 대칭 자세 앞뒤 차이(재풀이 −10.7 mm, 왼팔 J4 +4.3° vs 오른팔 +2.2°)는 그대로. 실기 미검증

---

## 2026-10-23 — 첫 v1.3 실기: J5 단계 미수렴 → J5 단계 스윕 12 s
> 검증 수준 ⚪ (로그·스윕 원본 분석) + G2 · 실기 재확인은 사용자
- **실행** (14:0x, 백업 `archive/runs/20261023_1409_v13_robot_j5_fail/`): 리셋 전 관절 OK → 오른팔 브라켓 x 66.91·z −1.11 mm, J6(`wrist_roll_v13`) 3회 수렴 −0.83° → J5(`wrist_pitch_v13`) 8회 미수렴. 추정값 5.92(클램프 5) / 4.26 / 4.72 / 4.16 / 4.16 / 4.30 / 4.64 / 4.67°, 흩어짐 약 0.24° (시뮬 0.10°)
- **원인 (추정)**: J5 단계의 J6 스윕이 기본 8 s(60°, 약 240프레임). 같은 J6 축을 12 s로 도는 J6 단계는 축 사이 각이 87.60~87.63°(0.015°)로 안정. harness T20(J6 스윕 12 s 미만 금지, v1.2 실기 8 s에서 흔들림 3배)을 v1.3 J5 단계가 지키지 않고 있었음. 시뮬은 지연·동역학이 없어 8 s와 12 s 차이가 없었다(10-15). J4 원(r 68 mm) 맞춤 잔차 0.20 mm, J4 축 방향은 반복 간 약 0.1° 안. J6가 −15°인 준비자세는 J4–J6 축 사이 각에 영향 없음
- **수정 (사용자 "12초로")**: `JointCalibrator.JOINT_SWEEP_DURATION_BY_MODE_S["wrist_pitch_v13"] = 12.0` (v1.3 전용, 기준·상한 그대로). G2 통과 324 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261023_v13_j5_sweep_12s.log`)
- 이 결과가 의미하지 않는 것: 12 s로 흩어짐이 줄지는 실기에서 확인 전

---

## 2026-10-21 — 실기(192.168.1.70, v1.2) 재조정 창 반복·상태 조회 불가 (사용자 문의)
> 검증 수준 ⚪ (화면 로그 `result/result_log/gui_log_20261021.txt` 분석) · 로봇 동작 없음
- **흐름**: 19:26 Home Offset Reset(전 관절 OK) → Full Auto: 왼팔 브라켓 Axis 5 스윕 "motion failed or was cancelled"(19:33) 재조정 2회 후 취소 → 이후 오른팔 J5 Joint A 스윕이 매번 같은 실패(19:39~20:01) → 19:43부터 gRPC `UNAVAILABLE`(Connection timed out, `tcp handshaker shutdown`) 반복, 영 자세 이동 실패 3회
- **원인 (추정)**: PC–로봇 통신이 끊기거나 멈춤. 스윕 중 상태 조회가 막혀 스윕이 실패로 끝나고 재조정 창이 뜬다. 09-29에도 동작 중 `get_state`가 4.5 s씩 막힌 같은 증상이 로봇 재부팅으로 풀림
- **위험 (코드 확인 + 추정)**: 19:54·19:56 Step 2 탭의 BASELINE 적용(19:26 리셋 되돌리기)이 통신 시간 초과로 중간에 끊김. 리셋 순서는 오른팔 0~6 → 왼팔 0~6 → 헤드이고 실패하면 거기서 멈춤(`reset_current_pose_home_offsets`). 1차는 left_arm_1에서, 2차는 left_arm_4에서 끊김 → 1차에서 이미 리셋된 오른팔 전체와 left_arm_0이 2차에서 같은 이동량만큼 한 번 더 움직인 뒤 리셋됐을 가능성(이중 적용: 오른팔 J0 −0.30, J1 −1.06, J3 −2.49, J5 −0.17°, 왼팔 J0 −0.10°). left_arm_1·4는 응답만 끊겨 적용 여부 불명, left_arm_5·6·헤드는 미적용(이동량 ≈ 0). 기준 파일은 19:26 그대로
- 권고: 통신 정상화 후 위자드의 Home Offset Reset(기준 자세)부터 다시 — 전 관절을 한 번에 다시 잡으므로 섞인 상태를 덮어쓴다
- **그 전 마지막 완료 실행** (19:04 리셋 → 19:20 Step 2 → 19:25 Apply, 백업 `archive/runs/20261021_1920_v12_robot_applied/`): 사용자 관찰 "왼쪽 마커가 약 4 mm 위". 데이터 품질이 평소보다 나쁨 — Step 1.5 3D 잔차 0.533 mm(왼 0.577, 10-16·10-19는 0.16~0.17), 카메라 x·z 모두 상한, 지연 66.6 ms(전에는 54), Step 2 위치 잔차 0.256 mm(평소 0.08~0.10), 헤드 pan Step 1.5 −4.32° → Step 2 −0.06°·tilt −0.21 → −0.73°로 두 단계가 어긋남, 적용 후 마커 0.59 mm / 0.28°(평소 0.2~0.3 mm). 추정: 이후 드러난 통신 지연이 이때부터 상태 조회를 늦춰 엔코더-영상 짝이 어긋났을 가능성(미확인). 이 결과는 쓰지 않는 게 안전
- **발견 (미수정)**: `main_ui.py:5031` `handle_full_auto_joint_finished`가 Step 1 팔꿈치(J3) 결과를 `np.clip(-3.0, 0.0)`로 조용히 자른다(09-15 `5774239`부터). 이번 왼팔 −3.0028 → −3.0000(0.003°). 팔꿈치를 펴는 쪽으로 리셋하는 지금 방식에서는 J3가 −2.3~−3.0°로 경계에 붙어, 더 크면 경고 없이 잘린다

---

## 2026-10-20 — 프로젝트 정리: scratch 정리 (사용자 "1번으로 진행")
- 결정 (사용자): 10-19 로봇의 확인 자세 오차(양손 z 1~3 mm)는 한계로 둔다. 프로젝트 정리 시작
- **삭제** (모두 커밋 `a860b09`에 있음, git 이력으로 복원 가능): 일회성 분석 25개 (`*_20261008.py` 10개와 `step15_direction_check_audit.py`, `camera_height_recheck_20261015.py`, `step2_rerun_headless_20261002.py`, `verify_j4_j5_misalignment.py`, `wrist_axes_check.py`, `joint_sweep_scatter.py`, `v13_j5_noise_vs_arc.py`, `v13_sweep_view_check.py`, `v13_wrist_axis_estimators.py`, `v13_j5_ready_search.py`, `v13_sweep_clearance.py`, `v13_sweep_view_margin.py`, `v13_bracket_position_from_radii.py`, `v13_bracket_position_debug_sim.py`, `v13_bracket_yaw_estimators.py`)와 고장 난 `validate_step1_2_sim.py`, `scratch/__pycache__`
- **살림**: `arm_sign_test.py`, `camera_swap_test.py`, `randomized_sweep_test.py` — `get_both_arm_config`·`get_head_config` import를 `core.calibration.data`로, 없어진 입력 데이터셋(`dataset_20260913_121257.npz`, 어디에도 없음)을 `archive/runs/20261019_0954_v12_robot_applied/result_step2/dataset_20261019_095439_157666.npz`(자세 틀만 씀)로. v1.2 시뮬에서 `N_TRIALS=2`로 세 개 모두 완주 (예전 자유 카메라 모드의 스트레스 시험이라 수치는 운영 모드와 비교 대상 아님). 출력 json은 지움, 시뮬은 다시 정지
- **이름 변경**: `step2_rerun_20261008_1750.py` → `step2_rerun.py` (불러 쓰는 4개와 설명 수정). 10-19 실행 재계산으로 저장 결과 재현 확인
- **남긴 scratch 14개**: `validate_full_v13_sim.py`, `summarize_v13_sim_runs.py`, `v13_stage_scatter_sim.py`, `validate_wrist_diversity_sim.py`, `step2_rerun.py`, `step2_with_step15_roll.py`, `head_camera_roll_check.py`, `step15_tilt_scan.py`, `check_pose_error_cost.py`, `bracket_roll_scan.py`, `calibration_report.py`(+`tests/test_calibration_report.py`), 스트레스 테스트 3개
- 참고: `sed -i`로 고친 scratch 파일 일부가 줄바꿈만 바뀌어 수정으로 보임(내용 차이 없음). `.graftignore` 삭제는 이 작업에서 하지 않았음(원인 확인 필요)
- **이어서 (사용자 결정)**: 코드에서 안 쓰이는 이미지 10개 삭제(`additional_head_bracket_.png`, `camera_intrinsics.png`, `conect_robot_set_zeroset.png`, `head_onoff.png`, `home_offset_position.png`, `main_ui_step1.png`, `main_ui_step2.png`, `marker_bracket_assemble.png`, `process.png`, `step1_calib.png`). graft 관련 삭제: `scripts/generate_architecture.py`, `.gitignore`의 graft 줄, `CLAUDE.md`의 "File reads"(graft) 절. `.gitignore`에 `archive/` 추가 (이미 추적 중인 1,337개는 `git rm -r --cached archive`를 해야 빠짐, 사용자 판단). 미사용 함수(A 목록)는 남김 (사용자: 시스템 전체 점검을 하지 않기 위해)
- **리드미·환경 세팅 (사용자: 스크립트 안내 대신 매뉴얼 단계로)**: `Readme.md`의 Overview(OS, Python 3.11/3.12), "2. Environment Setup"(Windows / Ubuntu 단계: Python, venv, `libxcb-cursor0`, RealSense udev 규칙, USB 3.0, 화면, 로봇 네트워크), "3. Notes"(카메라 교체·버전 이동 설명 유지), 실행 명령, 빌드 스크립트 경로, "4. Target PC Requirements"(exe 대상 PC 필요 조건, glibc·아키텍처 제약)를 다시 씀. 사라진 `main_ui.py`의 Qt 경로 처리 설명은 뺌. 저장소 주소는 그대로(사용자 확인 대기)
- **OpenCV headless** (사용자 동의: 같은 버전이라 기능 차이 없음): `requirements.txt`·`setup.py` `opencv-contrib-python==4.13.0.92` → `opencv-contrib-python-headless==4.13.0.92` (OpenCV 자체 Qt 플러그인이 없어 PySide6와 충돌 원인 제거, 앱은 OpenCV 창을 쓰지 않음 — 유일한 `cv2.imshow`는 `imshow_when_detect = False` 디버그 경로). 이 PC `.venv`도 교체: aruco·AprilTag 36h11 사전·PySide6 정상, `imshow`만 없음. G2 통과 324 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261020_opencv_headless.log`)
- **리드미 재구성 (사용자 순서)**: Overview → 1. Supported Systems → 2. Performance(10-16·10-19 v1.2 실측: Full Auto 16~22분, Step 2 위치 잔차 0.08~0.10 mm, 적용 후 마커 0.2~0.3 mm·0.1~0.2°, 실행 간 0.1~0.2°, 확인 자세 1~3 mm(육안), tilt 공통 오차 ~1 mm, v1.3은 시뮬만) → 3. Required Equipment(마커, ChArUco 8×5·36/27 mm·DICT_5X5_100 = 저장소 A4 PDF, 100 % 인쇄, 시뮬) → Environment Setup(실행파일 대상 PC에도 필요한 단계 표시) → Running(실행, 카메라·버전 노트) → Standalone Executable. 저장소 주소는 배포용이라 유지(사용자). 빌드는 사용자가 한다
- G2 1차 324 중 1 실패: `test_camera_model_config` 보관 테스트가 실제 `config/camera_intrinsics_d405.yaml` 값을 기대값으로 써서, 10-19부터 작업 파일이 시리얼 260322277376 카메라 값(fx 655.56, 저장 파일은 651.83)일 때 실패. 앱 동작(나가는 작업 파일을 보관 후 복원)은 맞음 → 기대값을 테스트 시작 시점 작업 파일 값으로 수정. G2 통과 324 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261020_cleanup_2.log`)

---

## 2026-10-19 — 또 다른 v1.2 로봇 09:54 실행·적용: 확인 자세에서 오른손이 로봇 x축 − 방향으로 돌아 보임 (사용자)
> 검증 수준 ⚪ (로그·오프라인 재계산·FK 계산) · 코드 변경 없음
- **실행**: 09:38 리셋 → Full Auto → 09:54 Step 2 → 09:58 Apply. 16:31 이후 v1.3 로봇 연결 기록(분석 대상 아님). 백업 `archive/runs/20261019_0954_v12_robot_applied/`
- **값**: Step 1.5 광축 회전 −0.657°(10-16 로봇 −0.65°와 거의 같음, 같은 카메라·브라켓인지 미확인 — 로그에 시리얼 없음), 헤드 pan +2.05°, 3D 잔차 0.161 mm, 카메라 x 상한·z 하한. Step 2 위치 잔차 0.077 mm, 검증 자세 마커 0.32 mm / 0.18°. 오프셋 R `−0.63 +2.73 +0.36 +2.09 −0.13 +0.69 +1.35`, L `+2.57 −1.68 +0.10 +2.22 −0.15 +0.19 −0.13`, 좌우 대칭 차 J0 +3.19°·J1 +1.05°·J6 +1.22°
- **카메라 영향 (오프라인)**: 설계 x 45.0으로 Step 1.5→Step 2를 다시 풀어도 관절 변화 0.05° 이하(오른 J1 +2.73 → +2.76), Step 2는 x 44.5 mm를 원함. 위치 범위 ±8 mm 0.15° 이하, 각도까지 풀어도 오른팔 0.12° 이하 → **카메라 설정은 원인이 아니고 45 mm여도 같았을 것**
- **확인 자세(v1.2 `check_calib`)에서 오른손의 로봇 x축 둘레 회전 민감도** (URDF FK, 1°당): J1 +0.91°, J4 −0.75°, J5 −0.52°, J2 −0.39°, J0·J3·J6 ≈ 0. Step 2 약한 방향 4번(LJ4 +0.47, LJ6 −0.47, RJ6 −0.42, RJ4 +0.35, 단위당 잔차 상승 0.33σ)이 오른손을 x축 둘레 −0.26°/단위 돌린다 → 확인 자세의 x축 회전은 Step 2 자세 범위 밖의 J4·J6 조합(과 J1)에 약하게만 묶여 있다 (추정)
- 오른팔 J6: Step 1 +1.21° / Step 2 +1.35°로 일치. 왼팔은 −0.62 / −0.13°로 0.49° 차이
- **사용자 추가 관찰**: 확인 자세에서 오른손 x 위치는 맞고 roll(로봇 x축 둘레)이 틀어져 z가 약 2~3 mm 어긋나 보임, 오른팔 J0이 맞지 않음. 왼팔은 모름
- **확인 자세 오차의 비용** (`scratch/check_pose_error_cost.py`, Step 2를 다시 풀어 최적화기를 받아 min |J d|² s.t. 확인 자세 손 이동 = 목표): 오른손 z +2.5 mm·x,y 0이고 왼손 자유면 잔차 RMS 0.733 → 0.853σ (J3/J5 고정) — 카메라 z −0.93 mm, 헤드 tilt −0.47°, R/L J0 −0.42 / −0.39°의 공통 방향이라 왼손도 +2.3 mm 같이 움직이고 오른손 roll −0.05°. 왼손을 고정하면 0.733 → 1.815σ (J1 양팔 −0.65°, 카메라 y −1.2 mm, 오른손 roll −0.42°) → 오른손만의 roll·z 오차는 데이터가 분명히 거부한다
- **오른쪽 브라켓 roll 스캔** (`scratch/bracket_roll_scan.py`, 설계 90° ± 0.6°): 잔차 0.812 / 0.779 / 0.751 / **0.733** / 0.726 / 0.738 / 0.763σ (−0.6 … +0.6), 최소는 +0.2° 근처지만 평평. roll은 오른팔 J4(−0.72 … +0.30°)·J6(+1.88 … +0.90°)과 맞바뀌고 J0은 ±0.12°만 움직임 → 브라켓 roll은 오른팔 J0 문제의 원인이 아님 (데이터가 roll을 ±0.3° 안으로 둠)
- **판단 (추정)**: 양손이 같이 z로 어긋났다면 J0 공통·헤드 tilt·카메라 높이의 약한 방향(관측 한계). 오른손만이라면 Step 2 모델 밖(팔 기구의 축 기울기 등)이라 오프셋으로는 못 맞춘다. 왼손 확인이 필요
- 사용자: 양손 공통일 것 같다
- **Step 1.5 tilt 정밀도** (`scratch/step15_tilt_scan.py`, 이번 헤드 스윕, 운영 조건 = 설계 범위·광축 회전 추정, tilt 고정 스캔): 자유 해 tilt +0.090°, 3D RMSE 0.1570 mm. 고정 tilt ±0.2° → +0.002 mm, −0.4 / +0.4° → +0.009 / +0.007 mm (약 5%), ±0.6° → +0.020 / +0.016, ±1.0° → +0.052 / +0.041 mm (약 30%). → Step 1.5가 tilt를 약 ±0.4° 폭으로만 잡는다 (10-08 로봇과 같은 수준)
- **그 tilt가 Step 2에 주는 영향**: tilt −0.4° → J0 R/L −0.16 / −0.17°, 확인 자세 양손 z 약 +1.0 mm, tilt +0.4° → J0 +0.12 / +0.09°, 양손 z 약 −0.6 mm. Step 2 위치 잔차 0.077 → 0.079 / 0.082 mm (거의 그대로)
- **판단 (추정)**: tilt 관측 한계로 설명되는 것은 양손 공통 z 약 ±1 mm까지. 2~3 mm가 되려면 tilt가 1° 넘게 틀려야 하는데 그러면 Step 1.5 잔차가 30% 넘게 커진다 → 남는 1~2 mm와 roll은 tilt·카메라 밖(팔 기구 형상 등, 모델 밖)일 가능성. 손끝 2~3 mm는 육안 추정이라 크기 자체의 불확실성도 있음

---

## 2026-10-16 밤 — 시스템 확정 (사용자 "이걸로 시스템은 픽스하자", 노션 기록 요청)
- **확정 상태**: 21:25 실행·21:29 Apply(다른 v1.2 로봇)의 코드와 설정. camera_info.yaml D405 `mount_to_cam` x 43.3 / y 9.0 / z 63.0 mm, 범위 x −3/+1, y 0, z ±0.5 mm. 광축 회전은 Step 1.5 추정(±5°)·Step 2 고정
- **오늘 들어간 변경 (모두 미커밋)**:
  - v1.3 브라켓: 위치를 J5·J6 반지름으로(J4 제외), pass 1 위치를 J6 오프셋 반영해 다시 풀어 pass 2와 비교, yaw 축을 원 피팅 축 + robust 매핑으로
  - Home Offset Reset: 준비자세 관절 한계 검사 삭제(자세 부호 검사는 유지)
  - 카메라: 범위 모드에서도 광축 회전 추정(Step 1.5), Step 2는 Step 1.5 각도로 고정, 최적화기 위치 상자는 가장 가까운 설계축 기준
- **보류**: 설계 카메라 x 45.0(적용값 대비 J0 0.15~0.2° 차이, 실행 간 흩어짐 크기). 카메라 y 범위
- **남은 작은 것**: 검사에서 막혀도 뜨는 팝업 문구("finished, but some joints failed", `main_ui.py:4805`), 막힌 리셋에도 `home_reset_baseline.json`이 새로 써짐, `sequences/step1.py`의 "브라켓은 J5/J6 오프셋과 무관" 주석(v1.3 위치에는 이제 안 맞음)
- **트랩 제안 (재시도 금지)**: v1.3 브라켓 위치에 J4 반지름 쓰지 않기 / 브라켓 축을 가운데 프레임 기준 회전만으로 구하지 않기 / 카메라 광축 회전을 설계값에 묶지 않기(이 로봇 −0.65°가 양팔 J1 −1.1°로 들어감)

---

## 2026-10-16 밤 — 광축 회전 반영 후 20:39 실행: "이제야 제대로된 듯", 왼팔 J0만 조금 뒤로 (사용자)
> 검증 수준 ⚪ (로그 + Step 2 오프라인 재계산)
- **백업**: `archive/runs/20261016_2039_v12_robot_optical_roll/` (19:49·20:39 결과 json·npz, result_txt·log·img 전체, setting·camera_info·ready_poses·리셋 기준 파일)
- **20:39**: Step 1.5 광축 회전 −0.651°, 3D 잔차 0.173 mm, 헤드 pan +0.012 / tilt +0.029°. Step 2 위치 잔차 0.098 mm, 검증 자세 마커 0.24 mm / 0.17°(19:11은 1.00 mm / 0.59°). 카메라 x는 Step 1.5·Step 2 모두 상한(+1 mm)
- **왼팔 J0 재계산** (`step2_rerun_20261008_1750.py`, `archive/reports/step2_rerun_20261016_2039/`): 카메라 x를 ±5로 풀면 x 45.30 mm에서 L J0 +0.30 / R J0 +0.23°, xyz ±8이면 x 45.20·y 10.07에서 L +0.43 / R +0.08°, 각도까지 풀면 L +0.53 / R −0.03°. z만 풀면 0.05° 이하. 왼팔 J0이 카메라 x(·y) 고정에 0.3~0.5° 민감하고 오른팔은 덜 민감
- 세 로봇 모두 자유 해의 카메라 x가 45.0~45.3 mm(설계 43.3, 상한 44.3). 설계 x(모듈 앞면 47 − 깊이 시작점 3.7)가 ~2 mm 작을 가능성. 결정 대기: 설계 x 변경 또는 상한 확대(사용자 결정, camera_info.yaml)
- **설계 x 45.0 오프라인** (사용자 "1번으로 해서 오프라인으로 먼저", `head_camera_roll_check.py --design-x 0.045` → `step2_with_step15_roll.py ... --design-x 0.045`, `archive/reports/head_camera_roll_check_20261016_x45.0/`, `step2_rerun_20261016_2039/`, `step2_rerun_20261008_1750/`; config는 안 바꿈):
  - Step 1.5: 네 실행 모두 카메라 x가 범위 안(설계 대비 −0.55~+0.32 mm), 잔차 거의 그대로(20:39 0.170 → 0.170 mm)
  - Step 2 20:39: 카메라 x 45.23·z 62.83(범위 끝 아님), 위치 잔차 0.098 → 0.088 mm, **J0 L −0.20 → +0.07(+0.27), R −0.08 → +0.11(+0.19)** → 대부분 양팔 공통 이동, 왼팔만의 차이는 0.08°
  - Step 2 10-08 17:50: 관절 변화 0.14° 이하, 카메라 x 43.80(이 로봇 Step 2는 45를 원하지 않음)
  - 왼팔만의 J0 차이는 카메라 y(설계 9.0 고정)에 더 민감: y ±2를 풀면 y 10.13에서 L +0.17 / R −0.14°(좌우 차 0.31°)
- **21:25 재실행 (사용자 "디게 잘됐다", 21:29 Home Offset Apply)**: 코드·camera_info는 20:39와 같음(마지막 코드 수정 20:10). Step 1.5 광축 −0.650°, 3D 잔차 0.172 mm, Step 2 위치 잔차 0.092 mm, 검증 자세 마커 0.19 mm / 0.12°. 20:39 대비 J0 L −0.20 → −0.05, R −0.08 → +0.04, L J6 +0.52 → +0.73, L J4 −0.31 → −0.48 → 실행 간 흩어짐이 0.1~0.2° 수준. 20:39의 왼팔 J0 차이도 이 범위 안. 설계 x 45.0은 적용 안 함(결정 보류). 백업 `archive/runs/20261016_2125_v12_robot_applied/`. **Apply됨: 이후 분석에서 이 오프셋을 다시 더하지 말 것**
  - 설계 x 45.0으로 같은 데이터 오프라인 재계산(적용값 대비): R J0 +0.15 J1 +0.08 J6 +0.13, L J0 +0.20 J1 −0.08 J4 −0.10 J6 −0.09, 나머지 0.05° 이하, 헤드 +0.01 / +0.03°. 카메라 x 45.15·z 62.52(z가 하한 근처로), 위치 잔차 0.092 → 0.084 mm. 차이가 실행 간 흩어짐(0.1~0.2°)과 같은 크기

---

## 2026-10-16 저녁 — 다른 v1.2 로봇 19:11 결과: 양팔 J1이 오른쪽으로, J0이 뒤로 (사용자 육안, J5는 맞음)
> 검증 수준 ⚪ (로그·결과 분석 + Step 2 오프라인 재계산) · 원인 미확정
- **실행**: 18:55 리셋(팔꿈치 뒤로, 한계 검사 삭제 후) → Full Auto. v1.2로 인식, v1.2 브라켓(y ±54, z −48.8 mm) 사용 → v1.3 값 섞임 아님. Step 1 수렴(J3 −1.90 / −2.27°: 리셋 의도대로 한 방향), Step 1.5 카메라 x 상한·z 하한, 3D RMSE 0.437 mm, Step 2 카메라 x 상한·z 상한, 검증 자세 마커 1.00 mm / 0.59°, 대칭 차 J0 −1.5°·J6 +1.4°
- **입력 보존**: `archive/runs/20261016_1911_v12_robot/` (result json·npz, Step 1.5 json, GUI 로그, 리셋 기준 파일)
- **오프라인 재계산** (`scratch/step2_rerun_20261008_1750.py <run> <ts>`로 인자화, `archive/reports/step2_rerun_20261016_1911/`): 저장 결과 그대로 재현. 카메라 **위치** 범위를 풀어도(z ±8, x ±5, xyz ±8) 관절 변화 0.2° 안팎(y만 풀면 0.77°) → 위치 범위는 주원인 아님. 카메라 **각도**까지 푸는 예전 모드(`old_mode`)는 위치 잔차 0.198 → 0.087 mm, 카메라 pitch −0.87°, 헤드 pan +0.31 → −0.07°, J0 R −0.56 / L +1.07°, J1 R −0.66 / L −1.07° 변화
- **추정 (미확정)**: 이 로봇 카메라 장착 각도가 설계와 ~0.9° 다르고, 각도를 설계값에 고정하는 모드(10-08~)에서 그 오차가 헤드 pan과 양팔 J0·J1 공통 성분으로 들어갔을 수 있다. 사용자가 본 방향과 맞는지는 부호 규약을 실측하지 않아 판단하지 않았다
- 사용자: 카메라를 재조립해도 같음, tilt가 조금 내려간 듯, pan은 확인 어려움, "헤드가 문제인가?"
- **Step 1.5 재계산** (`scratch/head_camera_roll_check.py`, 헤드 스윕만 = 팔과 독립, `archive/reports/head_camera_roll_check_20261016/`): 광축 회전을 자유로 풀면 10-08 robot3 +0.06°, 10-08 17:50 로봇 −0.16°, **이번 로봇 −0.65°** (3D 잔차 0.32 → 0.10 mm). 헤드 pan/tilt 오프셋 자체는 −0.11 / −0.03°로 작다. Step 2 각도 자유 해(−0.87°)와 같은 방향
- **Step 2에 회전을 넣으면** (카메라 고정 예전 모드에서 같은 조건 비교; 위치 범위 모드는 최적화기가 설계축 정렬 회전만 받아 불가): 회전 0° → −0.65°에서 위치 잔차 0.276 → 0.118 mm, **양팔 J1이 함께 −1.14°**, J0 R −0.78 / L +0.84°
- **판단**: 카메라가 헤드 tilt 축 기준으로 광축 둘레로 ~0.65° 돌아가 있다(두 독립 신호). 카메라 재조립으로 안 바뀌었으니 브라켓 부품 자체나 헤드 쪽 장착면일 가능성. 데이터로는 둘을 구분할 수 없다. 10-08부터 각도를 설계값에 고정해서 이전 로봇(−0.16°)에서는 드러나지 않았다
- 결정 대기: Step 1.5의 광축 회전 추정 되살리기(10-08 결정 변경, Step 2 최적화기 수정 필요) / 하드웨어 점검 / 이 로봇만 예전 모드
- **결정 (사용자 "1번 하자")**: 범위 모드에서도 광축 회전 추정. 바뀌는 규칙: "범위 모드 = 카메라 각도 설계값 고정"(10-08) → "광축 회전은 Step 1.5가 추정(±5°), Step 2는 Step 1.5 각도에 고정, 위치 범위는 그대로". 볼트 harness 4번(2026-09-15~ 위치·광축 회전 추정)의 원래 설계로 돌아감
  - 수정: `HeadCameraCalibrator._solve_head_camera`(범위 모드에서 광축 회전 고정 줄 삭제), `sequences/step2.py`(범위 모드의 시작 각도 = Step 1.5 각도, 위치는 설계값), `calibration_optimizer.QPCalibrationOptimizer`(위치 범위 상자를 가장 가까운 설계축으로, 6° 넘게 돌아간 카메라는 계속 거부)
  - Step 2가 직접 광축 회전을 풀지 않게 한 이유: Step 2에서는 양팔 J1 공통 성분과 맞바뀐다(이 로봇 −0.87° vs Step 1.5 −0.65°). Step 1.5는 팔과 독립
  - 테스트 `tests/test_camera_mount_constraint.py`: 범위 모드에서 광축 회전 −0.65° 찾기, 회전된 카메라에서 위치 상자 유지, Step 2가 Step 1.5 각도 사용 (3개 추가), Step 1.5 범위 끝 테스트는 "광축 외 회전은 그대로"로 수정. G2 통과 324 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_camera_optical_axis_rotation.log`)
  - 오프라인 (`scratch/head_camera_roll_check.py`, `scratch/step2_with_step15_roll.py`): Step 1.5 범위 모드 광축 회전 — 이 로봇 −0.648°(3D 잔차 0.323 → 0.159 mm), 10-08 17:50 로봇 −0.160°(0.142 → 0.123), robot3 +0.022°. Step 2 — 이 로봇 위치 잔차 0.198 → 0.107 mm, J1 R −0.77 / L −1.01°, J0 R −0.46 / L +0.69°, 헤드 pan +0.31 → +0.03°, 카메라 z 범위 끝 벗어남(63.34). 10-08 17:50 로봇 관절 변화 0.11° 이하
  - 이 결과가 의미하지 않는 것: 실기 확인 자세에서 J1·J0 기울기가 사라지는지는 미확인. 19:49에 또 한 번 실행이 있었다(코드 수정 전에 띄운 앱이면 옛 코드). Apply 기록은 없음

---

## 2026-10-16 — v1.3 로봇에서 Home Offset Reset이 항상 막히던 문제 (사용자 로그)
> 검증 수준 🟠 (단위 테스트) · 실기 미확인
- **증상** (사용자 GUI 로그, 모델 M v1.3): Reset 때 "Joint 'right_arm_5' angle +90.4° (ready +90.0° + offset +0.4°) exceeds physical joint limits [−50.0°, +50.0°] under pose target 'marker'" 로 중단
- **원인 (확인)**: `home_offset.validate_home_offset_joint_limits`가 연결된 로봇 버전과 상관없이 `ready_poses.yaml`의 v1.2·v1.3 준비자세를 모두 검사. v1.2 marker 자세의 J5 +90°를 v1.3의 J5 한계 ±50°에 댔다. 10-14 전에는 이 검사가 매번 예외로 끝나 통과로 처리됐기 때문에 드러나지 않았고, 10-14 수정(검사가 실제로 돌게 함)으로 나타났다. 같은 검사를 쓰는 Apply 경로(`reset_current_pose_home_offsets`)도 v1.3에서 같은 이유로 막혔을 것
- **수정**: 같은 파일의 기존 방식대로 `robot.get_robot_info().robot_model_version`에 "1.3"이 있으면 v1.3, 아니면 v1.2 준비자세만 검사
- **테스트**: `tests/test_system_regressions.py` `TestHomeOffsetResetValidation`에 1개(J5 한계 ±50°에서 v1.3 로봇은 통과, v1.2로 보고되면 right_arm_5로 거부). G2 통과 322 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_home_reset_version_poses.log`)
- **이 결과가 의미하지 않는 것**: 실기에서 Reset·Apply가 통과하는지는 확인하지 않았다. 버전 조회가 실패하면 v1.2로 보고 검사한다(앱의 버전 분류와 같은 기본값)
- **이어서 (다른 로봇, v1.2)**: 팔꿈치를 뒤로 보낸 리셋 자세(오른팔 J3 +1.7°)에서 "right_arm_3 angle +1.7° (ready +0.0° + offset +1.7°) exceeds [−150.0°, +1.0°] under 'joint_elbow'"로 거부. 사용자 의도: 보정값이 한 방향(굽히는 쪽)으로만 나오도록 팔꿈치를 펴는 쪽으로 보낸 자세를 새 0으로. 검사는 리셋 전 영점을 맞다고 보고 한계를 그 기준으로 댔는데, 리셋 후 J3 = 0은 팔이 지금 있는 바로 그 위치다. 10-14 전에는 이 검사가 예외로 돌지 않아 드러나지 않았다(10-14 수정이 되살린 문제)
  - 수정: 관절별 허용 범위를 [min(하한, 현재 위치), max(상한, 현재 위치)]로. 현재 위치보다 더 한계 쪽으로 가는 준비자세는 계속 거부
  - 테스트 2개(+1.7° 팔꿈치 리셋 통과, 현재 위치 너머로 가는 준비자세 거부). G2 통과 324 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_home_reset_current_pose_limit.log`)
  - **변경 (사용자 "지금 자세로 범위를 넓히지 말고, 그냥 이거 기능 삭제해")**: 위 범위 확장과 같은 날 넣은 버전별 준비자세 선택을 포함해, 준비자세 관절 한계 검사를 통째로 삭제. 자세 부호 검사(어깨 roll·팔꿈치 방향, 5° 여유)는 유지. 함수 이름 `validate_home_offset_joint_limits`는 호출처 때문에 그대로 두고 docstring에 삭제 사유를 남김. 중단 로그 "aborted due to the reset posture"로. 테스트는 부호 검사 2개 + 팔꿈치 +1.7° 리셋 통과 1개만 남김. G2 통과 321 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_home_reset_limit_check_removed.log`)
  - 남은 것(미수정): 검사에서 막혀도 팝업이 "Home Offset Reset finished, but some joints failed to reset"(`main_ui.py:4805`), 막힌 시도에도 `config/home_reset_baseline.json`이 새로 써짐

---

## 2026-10-16 — v1.3 브라켓 위치: J4 반지름을 빼고 J5·J6 반지름으로 (시뮬에서 범위 끝에 걸리던 문제)
> 검증 수준 🟠 (단위 테스트 + 시뮬 브라켓 스윕 1회) · 실기 미검증
- **증상**: 10-15의 3반지름 풀이가 시뮬 실행에서 오른팔 (69.0, 0, 2.0) mm로 ±2 mm 범위 끝에 걸림. GUI 실행 기록 파일로는 맞았음
- **원인 (확인)**: J4 반지름이 72.05 mm(정답 브라켓이면 67.5). J4 반지름은 J5 실제 각도에 1차로 민감(약 2.1 mm/°)하고, 시뮬 정답 J5 오프셋 −2.1°에서 67.5·cos2.1° + 123·sin2.1° ≈ 72.0. v1.3은 브라켓을 J5보다 먼저 구하므로 pass 1에서 J5 오프셋을 모른다. 기록 파일이 맞았던 것은 J5 오프셋이 적용된 pass 2 스윕이었기 때문으로 보임(추정)
- **기각한 진단**: "스윕 원 피팅의 카메라 좌표 축 사전값 때문"(10-15) — 사전값 없이 다시 맞춰도 72.05 mm. 그때 넣은 재피팅은 제거하고 스윕이 계산한 반지름을 그대로 씀
- **수정** (`MarkerCalibrator.v13_bracket_position_from_radii(marker_data_5, marker_data_6, arm_side, nominal_mm)`): J6 반지름(오프셋 무관)으로 z, J5 반지름(J6 오차에 2차로만 민감)으로 x, y는 설계값. FK 전에 알려진 J6 오프셋(`wrist_roll`)을 빼서 pass 2에서는 정확. 로그 출처 문구 "J5/J6 sweep radii, y held at the design value"
- **pass 1 오차 예상**: J6 오프셋을 모르는 동안 x가 −(123²·sin²ε)/(2x) 정도 작게 나옴. 시뮬 정답 기준 오른팔 −0.2 mm(2.3°), 왼팔 −0.4 mm(3.5°). pass 간 브라켓 변화 한도 0.5 mm 안
- **시뮬 1회** (`scratch/v13_bracket_position_debug_sim.py`, 50052, 오른팔, 오프셋 0 = pass 1 조건, `archive/reports/v13_bracket_position_debug_20261016_103256/`): 반지름 J4 72.14 / J5 140.20 / J6 123.00 mm → (67.27, 0.00, 2.00) mm, 범위 끝 아님. 정답 (67.5, 0, 2.0)
- **테스트**: `V13BracketPositionFromRadii` 4개(J6 오프셋 제거 테스트 신규, 스윕 반지름 불신 테스트 삭제). G2 통과 318 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_v13_bracket_j5j6_radii.log`)
- **시뮬 Full Auto 1회** (사용자 "pass 2까지 확인되게 Full Auto 한 번", `scratch/validate_full_v13_sim.py --addr 127.0.0.1:50052`, 잡음 위치 0.1 mm·각도 0°, 1204 s, `archive/reports/sim_v13_full_20261016_104027/`):
  - 브라켓 위치 x (pass 1 → pass 2, 정답): 오른팔 67.26 → 67.51 (67.5), 왼팔 67.52 → 67.98 (68.0). z ±2.00 (±2.0). pass 1의 x 부족분은 예상대로 J6 오프셋 크기를 따른다(2.3° → 0.25 mm, 3.6° → 0.46 mm). pass 간 브라켓 위치 변화 0.25 / 0.47 mm (한도 0.5 mm)
  - Step 1 반복: 오른팔 J6 3, J5 6, J3 3 / 왼팔 J6 3, J5 4, J3 3. pass 2는 모두 건너뜀(pass 1 수렴)
  - 정답 대비 최종 오차: 오른팔 관절 최대 0.100°(J1), 왼팔 최대 0.095°(J6), 헤드 pan 0.019° / tilt 0.083°, 카메라 0.38 mm / 0.071°, 브라켓 위치 오른팔 0.01 mm, 왼팔 0.50 mm(추정하지 않는 y 0.5 mm). 고치기 전 실행(10-15 18:44, 브라켓 위치가 범위 끝)은 관절 0.2~0.42°, 헤드 pan 0.45°, 카메라 0.43 mm
  - 경고: 브라켓 z가 정답 ±2 mm = 허용 범위 끝이라 pass마다 WARN, 양팔 브라켓 z 차이 4 mm로 대칭 강제 안 함(시뮬 정답이 비대칭), Out/Back 미수렴(진단용), Step 2 약한 방향 1개(1.3 mm/°)
- **시뮬 Full Auto 1회, 각도 잡음 0.37°** (사용자 "잡음 0.37° 조건으로도 한 번", `--orientation-noise-deg 0.37`, seed 기본, 1176 s, `archive/reports/sim_v13_full_20261016_112243/`):
  - 브라켓 x (pass 1 → pass 2, 정답): 오른팔 67.26 → 67.45 (67.5), 왼팔 67.49 → 68.01 (68.0), z +2.00 / −1.99
  - Step 1 반복: 오른팔 J6 3, J5 4, J3 3 / 왼팔 J6 3, J5 5, J3 3
  - 정답 대비 최종 오차: 오른팔 관절 최대 0.086°(J4), 왼팔 최대 0.166°(J6), 헤드 0.004° / 0.031°, 카메라 0.14 mm / 0.071°, 브라켓 각도 오른팔 0.189°·왼팔 0.102°
  - **왼팔 pass 간 브라켓 위치 변화 0.519 mm > 0.5 mm 한도 → WARN** (pass 2 값 유지). 위에서 예상한 pass 1 x 부족분(J6 3.6° → 약 0.46 mm)에 잡음이 더해진 것
- **남은 점 → 대응 (사용자 "1번으로 진행")**: pass 1 x 부족분은 J6 오프셋의 제곱에 비례해(5°면 약 0.85 mm) pass 간 0.5 mm 한도 WARN이 뜬다. 최종값은 pass 2라 결과는 맞지만, pass 간 비교가 반복성 대신 pass 1 편향을 재고 있었다. 한도는 바꾸지 않음
  - 선택지: ① pass 평가 때 pass 1 스윕을 pass 1에서 찾은 J6 오프셋으로 다시 풀어 비교(채택: 추가 동작 없음, 0.5 mm 한도가 반복성 확인으로 돌아감) ② pass 1에서 브라켓 위치를 J6 단계 뒤에 구하기(미채택: Step 1 순서 변경, 영향 범위 큼) ③ 그대로 둠(미채택)
  - 수정 (`sequences/step1.py` pass 평가, v1.3·pass 1만): J6 단계가 이미 `marker_calibrator.joint_offsets[arm]["wrist_roll"]`에 넣어 둔 오프셋으로 `v13_bracket_position_from_radii(res_5, res_6, …)`를 다시 불러, 그 위치를 pass 1 값으로 비교에 쓴다. 저장되는 브라켓 값은 바꾸지 않는다(pass 2가 덮어씀). 로그 "[INFO] … pass 1 bracket position with the J6 offset found in pass 1: x …, z … mm (solved before J6: …)"
  - 시뮬 확인 (`scratch/v13_bracket_position_debug_sim.py --side left --wrist-roll-offset -3.6043`, 50052, `archive/reports/v13_bracket_position_debug_20261016_115206/`): 왼팔 pass 1 조건 스윕 x 67.54 → J6 −3.60° 반영 68.03 mm (정답 68.0), z −1.99
  - 테스트: `tests/test_step1_head_contracts.py` `V13PassOneBracketResolvedWithJ6` 2개(다시 풀면 pass 2 수렴, 다시 풀지 않으면 J6 5°의 편향 0.85 mm만으로 한도 WARN). G2 통과 320 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_v13_pass1_bracket_resolve.log`)
  - **시뮬 Full Auto 1회, 잡음 0.37°** (사용자 "한 번 더", 1155 s, `archive/reports/sim_v13_full_20261016_124641/`): pass 1 재계산 오른팔 x 67.26 → 67.43, 왼팔 67.50 → 67.95. **pass 간 브라켓 위치 변화 0.04 / 0.09 mm** (이전 잡음 실행 0.19 / 0.52 mm) → 위치 WARN 사라짐
  - **새로 드러난 것 (이번 수정과 무관한 값)**: pass 간 브라켓 **회전** 변화 0.36 / 0.37° > 0.15° 한도로 WARN(이전 잡음 실행 0.07 / 0.05°). 최종 오른팔 브라켓 yaw +0.53° / 정답 +0.05°(0.48°), 그 영향으로 보이는 오른팔 J4 0.33°(이전 0.09°). 정답 대비 최종: 오른팔 관절 최대 0.331°(J4), 왼팔 0.106°, 헤드 0.023 / 0.101°, 카메라 0.21 mm / 0.071°
  - 추정 원인(미확인): v1.3 브라켓 yaw가 `extract_axis_from_rotations`(마커 회전만 쓰는 축 추출)로 J6·J5 축을 잡는다. 10-15에 J6 단계는 같은 이유로 원 피팅 축 + robust 매핑으로 바꿨지만 브라켓 yaw 경로는 그대로다. 잡음 실행 2회의 차이(0.05° vs 0.37°)가 실행마다의 흩어짐인지는 2회로 판단할 수 없다
  - 이 결과가 의미하지 않는 것: 반복 통계 아님(각 1회). 실기 미검증
  - **브라켓 yaw 방식 비교 (오프라인, 사용자 "진행해봐", `scratch/v13_bracket_yaw_estimators.py`, 세 실행의 마지막 pass 브라켓 스윕)**: 현재 방식(`extract_axis_from_rotations`, 가운데 프레임 기준 상대 회전)을 오프라인으로 재현한 값이 실행 로그의 Step 1 yaw와 같다(+0.05/+0.06, +0.21/+0.00, +0.53/−0.08). 정답 대비 yaw 오차 — 잡음 0°: 현재 0.001/0.000, 원 피팅 축 + `axis_in_marker_frame_robust` 0.007/0.003. 잡음 0.37° 1회차: 현재 +0.161/−0.055, 원 피팅 +0.004/−0.030. 2회차: 현재 +0.477/−0.139, 원 피팅 −0.005/+0.011 (°, 오른팔/왼팔). 원 피팅 방식이 잡음 조건 4개 추정 모두 0.03° 이내
  - **수정 (사용자 "코드 변경하고 시뮬 Full Auto까지")**: `compute_marker_bracket_from_orthogonal_sweeps`(v1.3 전용, 호출처 step1 v1.3 분기와 `compute_unified_bracket_calibration_v1_3`)의 J6·J5 축을 원 피팅 축(`axis_opt`) + `axis_in_marker_frame_robust`로. 원 피팅 축이 없으면 예전 회전 기반으로 돌아간다. J4 축과 `compute_wrist_joints_from_3axis_sweeps`는 그대로
  - 테스트 `V13BracketYawFromCircleFitAxes` 1개(가운데 프레임만 2° 틀어진 합성 스윕에서 yaw 오차 0.02° 이내; 같은 데이터에서 예전 방식은 1.26°). G2 통과 321 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261016_v13_bracket_yaw_circle_fit.log`)
  - **시뮬 Full Auto 1회, 잡음 0.37°** (1186 s, `archive/reports/sim_v13_full_20261016_142039/`; 직전 시도 `…_141923`은 WSL이 내려가 시뮬 연결 실패로 시작 못 함, 증거 제외): pass 간 브라켓 변화 위치 0.12 / 0.05 mm, **회전 0.023 / 0.006°** → 양팔 "All parameters converged physically", WARN 없음. 브라켓 yaw 오차 0.00 / 0.01°. 반복 오른팔 J6 3·J5 6·J3 3 / 왼팔 3·4·3. 정답 대비 최종: 오른팔 관절 최대 0.090°(J6), 왼팔 0.053°(J4), 헤드 0.026 / 0.074°, 카메라 0.17 mm / 0.071°. 같은 잡음의 이전 두 실행은 관절 최대 0.166°, 0.331°
  - 이 결과가 의미하지 않는 것: 수정 후 실행 1회라 반복 통계 아님. 실기 미검증
- **이 결과가 의미하지 않는 것**: 조건마다 한 번의 실행이라 반복 통계가 아니다(고치기 전과의 비교도 각 1회). 실기는 확인하지 않았다

---

## 2026-10-15 저녁 — 사용자 GUI v1.3 시뮬 Full Auto (18:17) 점검: 관절 오차의 원인은 브라켓
> 검증 수준 🟠 (시뮬 데이터 오프라인 재계산) · 코드 변경 없음
- **실행**: 사용자가 GUI로 v1.3 시뮬(50052) Full Auto. Step 1 J6 3회(양팔), J5 4·6회, J3 3회. Step 2 결과의 정답 대비 오차 최대 0.99°(왼팔 J6), J0·J1·J2 0.3~0.6°, J3·J5 0.05° 이하, 헤드 0.08°, 카메라 z 범위 위쪽 끝 63.5 mm(정답 62.8)
- **원인 확인** (`archive/reports/v13_gui_run_20261015/`): 같은 데이터로 Step 2만 다시 풀면 — Step 1 브라켓(실행 그대로) 최대 0.99° / 설계 브라켓 1.17° / 정답 각도만 0.48° / 정답 위치만 0.81° / **정답 브라켓 0.07° 이하, 헤드 0.05°, 카메라 z 62.84 mm**. 시뮬 정답의 브라켓 오차(위치 0.5~2 mm, 각도 0.5~0.8°)를 v1.3 Step 1이 추정하지 못해(위치·roll·pitch는 설계값 고정, yaw만 추정) 관절로 들어간다
- **v1.2와의 차이**: v1.2는 브라켓 x·z를 엔코더 각 반지름으로 구하고 pitch를 맞추며 roll은 설계값으로 고정(사용자: 브라켓 각도 오차 0.1° 미만). v1.3에는 브라켓 위치 추정이 없다
- **그 밖의 경고는 오류 아님**: Out/Back 미수렴 경고(진단용, 값은 본 해와 일치), 확인 자세 비대칭 J6 6.99°(시뮬 정답 오프셋이 좌우 다르게 설정됨), 기준선 비교 0(시뮬에는 리셋 기준선이 없음), Step 1.5 헤드 pan 1.96°(초기값, Step 2에서 0.72°로 정답 0.8°에 근접)
- **결정 (사용자 "둘 다 진행")**:
  - ① `config/simulation.yaml` 정답 브라켓 각도 오차 0.5~0.8° → 오른팔 [−0.08, −0.06, 0.05]°, 왼팔 [0.07, −0.05, 0.06]° (사용자가 측정한 장착 정확도 0.1° 미만). 위치 오차(0.5~2 mm)는 그대로
  - ② v1.3 브라켓 위치를 엔코더 각 스윕 반지름으로: `MarkerCalibrator.v13_bracket_position_from_radii` 신규. J4/J5/J6 축이 손목 중심에서 만나므로 마커 원점은 축 k에서 |(p − w) × a_k| 거리(축·중심은 각 스윕 첫 프레임의 FK). y는 반지름에 1차로 나타나지 않아(감도 0) v1.2처럼 설계값 고정, x·z는 `bracket_offset_tolerance_mm`(기본 2 mm) 안에서 맞추고 끝이면 WARN. `compute_marker_bracket_from_orthogonal_sweeps`(v1.3 전용)의 예전 위치 경로(FK 점 중앙값이 설계값과 40 mm 넘게 달라 조용히 설계값 반환, 그 대체 경로의 반지름 식도 v1.3 형상과 맞지 않음)를 대체. `sequences/step1.py`가 위치와 출처를 로그로 남김
  - 확인(GUI 실행 스윕 기록, 오프라인, `scratch/v13_bracket_position_from_radii.py`): 측정 반지름이 정답 브라켓 모델과 0.02 mm 안. 제품 함수 결과 오른팔 (67.57, 0, 2.00) / 정답 (67.5, 0, 2.0), 왼팔 (67.89, 0, −2.00) / (68.0, 0.5, −2.0) mm. y 0.5 mm는 추정 불가. 정답 z 2 mm가 허용 범위 끝과 같아 WARN이 뜬다
  - 테스트 3개(`tests/test_marker_calibrator_contracts.py` `V13BracketPositionFromRadii`). G2 통과 317 / 실패 0 / 오류 0 / skip 0 (`archive/test_logs/g2_20261015_v13_bracket_position.log`)

---

## 2026-10-15 오후 — v1.3 J6 축을 v1.2 방식으로, J6 단계 스윕 확대, Step 2 실패 원인 로그 (사용자 "다 진행")
> 검증 수준 🟠 (시뮬 반복 측정, G2) · 실기 미검증
- **v1.3 J6 축 계산**: `MarkerCalibrator.compute_unified_bracket_calibration`의 `axis_in_marker_frame`이 v1.3만 원 피팅 축을 쓰지 않고 회전만으로(가운데 프레임 하나 기준) 축을 구하던 조건(`if not self.is_v13()`)을 없앴다 → v1.3도 v1.2와 같은 `axis_in_marker_frame_robust`(위치 궤적 원 축, 전체 프레임 견고 평균). v1.2 경로는 그대로. 영향 범위: v1.3 J6 단계와 v1.3 수동 마커 보정 순서(`sequences/marker.py`)
  - 시뮬 J6 단계 흩어짐(같은 자세 6회, `scratch/v13_stage_scatter_sim.py`): 방향 잡음 0.37°에서 수정 전 0.5~0.9° 흔들림·미수렴 → 수정 후 0.15~0.18°. 방향 잡음 0에서는 0.006~0.011° → 0.12~0.13°로 오히려 커짐 (짧은 호의 원 축 잡음)
- **v1.3 J6 단계 스윕 확대**: `JOINT_CONFIGS["wrist_roll_v13"]` J6 ±20 → ±30°, J5 ±15 → ±25°. 흩어짐 방향 잡음 0: 0.04~0.05°, 0.37°: 0.08~0.10°. 평균은 회전 방식 잡음 없는 값과 0.05° 안. 화면 여유 58~138 px, 기울기 ≤ 39°. 손 끝부분 충돌 간격(근사) ≥ 132 mm
- **Step 2 실패 원인 로그**: `sequences/collection.py`가 자세 목록 크기를 로그로 남기고, 목록이 비면 `max_x`와 함께 "plan is empty" 오류, 샘플 0개면 "N poses in the plan, markers missed at M" 오류. 테스트 2개(`tests/test_step2_wrist_diversity.py`). 오전 시드 2 실패의 원인은 여전히 미확정
- **G2**: 1차 314개 중 오류 1(메인 화면 하위 프로세스 20초 시간 초과, 시뮬 2대 화면 부하 중 — 단독 9.2 s 통과), v1.2 시뮬을 내리고 재실행 통과 314 / 실패 0 / 오류 0 / skip 0. 로그 `archive/test_logs/g2_20261015_v13_j6_axis_step2_log*.log`
- **반복 실행** (`archive/reports/v13_sim_batch_20261015b/`): 사용자 지시로 중단(시간 부족). 끝난 것은 잡음 없음 1회(17:10~17:30)와 v1.2 수준 잡음 1회의 앞부분뿐
  - 잡음 없음: J6 단계 3회, J5 단계 4회, 팔꿈치 3회(양팔). Step 2 자세 64개, 카메라 정답 대비 0.70 mm·0.07°, 헤드 0.03 / 0.02°. 관절 오차는 오전과 같은 수준(시뮬 정답의 브라켓 장착 오차 몫: 왼팔 J6 0.93°, J0·J2·J4 0.1~0.5°)
  - v1.2 수준 잡음 시드 1(중단): 오른팔 J6 3회, J5 6회, 팔꿈치 3회, 왼팔 J6 3회까지
  - **방향 잡음 조건은 바뀐 설정으로 Full Auto를 돌려 보지 못했다** (J6 단계 반복 측정 0.08~0.10°까지만)
- v1.2 시뮬 컨테이너(`rby1_sim`, 50051)는 테스트 부하 확인을 위해 정지해 둠

---

## 2026-10-15 — v1.3 손목 단계: 수렴 설정, J6 스윕 위치, 시야·충돌 확인, 시뮬 반복 (사용자 "제시한 건 다 진행")
> 검증 수준 🟠 (docker 시뮬, 합성 계산, URDF 근사 형상) · 실기 미검증 · 원칙: v1.2 항목은 그대로, 같은 함수 안에서 v1.3 항목만 추가
- **원인**: v1.3 모드는 수렴 표(`MIN_MEAN_SAMPLES_BY_MODE`, `JOINT_ESTIMATE_NOISE_DEG`)에 없어 기본값(근처 추정 3개, 하한 없음)을 써서 최소 4회였다. v1.2 모드는 2개라 3회
- **스윕 1회당 흩어짐 측정** (`scratch/v13_stage_scatter_sim.py`, 시뮬 v1.2 수준 잡음, 같은 자세 8회 반복): J5 단계 0.10~0.13°(8 s·12 s 차이 없음), J6 단계 0.006~0.011°(시뮬은 방향 잡음 0). 합성 계산(`scratch/v13_wrist_axis_estimators.py`)이 시뮬 값과 맞음(0.12 예측 / 0.11 측정)
- **J5 정밀도 개선 시도**: 준비 자세 J5를 굽히면 J4 반지름이 커져(J5 −30°: 67 → 120 mm) 흩어짐이 0.03~0.06°로 줄지만 마커판이 돌아서 화면 밖(−173 px)·기울기 58° → 기각. 준비 자세 J4·J5 탐색(`scratch/v13_j5_ready_search.py`)에서 시야(60 px)·기울기(45°)를 지키는 조합은 지금 자세뿐
- **시야** (`scratch/v13_sweep_view_margin.py`): 준비 자세 J6 = 0이면 J6 스윕이 화면 아래 끝에 걸림(J5 단계 11~32 px, J6 단계 23~45 px). J6를 −15°(왼팔 +15°)로 돌리면 J5 단계 J6 ±30° 55~77 px, J4 ±30° 123~156 px(기울기 ≤ 42°), J6 단계 89~136 px. 추정 평균은 그대로(편향 없음)
- **충돌** (`scratch/v13_sweep_clearance.py`, URDF 캡슐 + 손목·플랜지·브라켓·마커판 근사): 손목 단계는 반대 팔을 영 자세로 보내고 움직인다(`perform_move_to_ready_pose`). 손 끝부분 최소 간격 147 mm 이상(범위 확대로 약 10 mm 감소). URDF의 손목 충돌 형상은 그리퍼(r 75 mm)라 그대로는 쓸 수 없었다
- **적용한 변경 (v1.3 전용)**: `config/ready_poses.yaml` v1.3 `joint.wrist_pitch`·`joint.wrist_roll` J6 오른팔 −15°, 왼팔 +15° / `CalibratorBase.JOINT_CONFIGS["wrist_pitch_v13"]` J6 ±30°, J4 ±30° / `JointCalibrator`: `wrist_roll_v13` 스윕 12 s(v1.2 J6와 같은 방식이라 같은 값), 수렴 표에 `wrist_roll_v13`(최소 2, 하한 0.06° = v1.2 J6), `wrist_pitch_v13`(최소 2, 하한 0.10° = 측정값). 하한 0.10°에서는 추정 2개의 표준오차가 0.071°라 J5 단계는 추정 3개(보통 4회)가 필요하다 — 기준(0.06°)은 그대로
- **G2**: 통과 312 / 실패 0 / 오류 0 / skip 0. 로그 `archive/test_logs/g2_20261015_v13_wrist_config.log`
- **시뮬 Full Auto 결과** (`archive/reports/v13_sim_batch_20261015/`, 집계 `scratch/summarize_v13_sim_runs.py`). 유효 실행: 잡음 없음 1회, v1.2 수준 잡음(위치 0.1 mm, 방향 0) 4회(시드 1~4), 방향 잡음 0.37° 1회

| 단계 | 잡음 없음 | v1.2 수준 잡음 4회 | 방향 잡음 1회 |
|---|---|---|---|
| J6 (`wrist_roll_v13`) | 3회 | 3회 (8/8) | 오른팔 8회 안에 미수렴 |
| J5 (`wrist_pitch_v13`) | 4회 | 오른팔 4회, 왼팔 4·5·7·4회 (8/8) | (진행 전 중단) |
| 팔꿈치 | 3회 | 3회 (8/8) | - |

  - Step 2(유효 3회 + 잡음 없음): 카메라 정답 대비 0.7 mm·0.07°, 헤드 0.04~0.10°. 관절은 J3·J5 0.06° 이하, J0 0.27~0.44°, J1 최대 0.74°, J2 0.47~0.66°, 왼팔 J6 0.91~0.96° — 시뮬 정답의 브라켓 장착 오차(위치 2 mm, 각도 0.6~0.8°)를 Step 1이 다 잡지 못해(브라켓 오차 2.1~2.3 mm, 0.9~1.0°) 관절로 들어간 몫. 잡음 없는 실행도 같다
  - 실제 실행의 화면 여유 최소 45 px(오른팔 J5 단계 J6 스윕), 나머지 73 px 이상
  - **실패·제외**: 시드 2는 Step 2 수집이 "유효한 샘플 없음"으로 실패(이유가 로그에 없음 — 자세 목록이 비었거나 몇 개만 남았을 때의 오류. `core/robot/motion.py:173`은 시작 오른손 x가 `max_x`를 넘으면 조용히 빈 목록을 만든다. 원인 미확정). 시드 5·6과 병렬 4회는 시뮬 6대를 동시에 띄운 부하로 `ControlManagerIdle` 동작 실패 → 제외 (AI 판단 실수)
- **방향 잡음에서 J6 미수렴의 원인 (코드 확인, 미수정)**: v1.3 J6 경로(`MarkerCalibrator.compute_unified_bracket_calibration`의 `extract_axis_from_rotations`, 874행과 482행)는 가운데 프레임 하나를 기준으로 회전축을 평균한다. 기준 프레임의 방향 잡음이 모든 추정에 공통으로 들어가 평균으로 줄지 않는다. v1.2는 같은 종류의 문제를 2026-09-22에 전체 프레임 견고 평균으로 고쳤다(`axis_in_marker_frame`). 프레임마다 독립인 0.37° 잡음은 실제보다 가혹할 수 있다(v1.2 J6는 실기 흩어짐 0.06°)
- **이 결과가 의미하지 않는 것**: 3시 마감으로 반복 횟수가 계획(10회)보다 적어 수렴 횟수의 분포를 말할 수 없다. 실기의 충돌·시야는 근사 계산이다. J5 단계 3회 수렴(v1.2 수준)은 달성하지 못했다 — 하한 0.10°에서는 구조상 최소 4회
- **다음**: ① v1.3 J6 축 추출을 v1.2와 같은 전체 프레임 견고 평균으로(같은 함수 공유 검토) ② Step 2 빈 자세 목록일 때 원인을 로그로 ③ 방향 잡음 조건 재실행

---

## 2026-10-15 — D405 카메라 높이 63 mm로 정정, v1.2 두 실행 재확인 (오프라인)
> 검증 수준 🟠 (저장 데이터 재계산, 실행 2건) · 실기 미검증
- **사용자 정정**: 브라켓 높이 67 mm는 기구물 전체 높이였고, 볼트 위치 기준 실측은 **63 mm**. `config/camera_info.yaml` D405 `mount_to_cam` z 0.067 → 0.063 (범위 ±0.5 mm 그대로). D435 항목(z 0.067)은 그대로 둠 (같은 브라켓이면 확인 필요). `config/simulation.yaml` 정답 카메라를 `[0.0436, 0.009, 0.0628, -90.05, 0.05, -90.0]`로 (설계 범위 안, 각도 오차 0.05°)
- **재계산** (`scratch/camera_height_recheck_20261015.py`, 출력 `archive/reports/camera_height_recheck_20261015/`): Step 1.5(헤드 스윕) → Step 2를 실행 당시 조건 / 67 ±0.5 / 63 ±0.5로. 실행 당시 조건은 두 건 모두 저장 결과와 0.00° 재현
  - **17:50 새 로봇 (적용된 결과는 67 ±1)**: 63 mm에서 Step 1.5 맞춤 오차 0.273 → **0.142 mm**, 카메라 62.85 mm(범위 안), Step 2 위치 잔차 0.106 → 0.086 mm. 관절 변화(적용값 대비) J6 +0.69 / −0.61°, J4 −0.31 / +0.22°, J2 +0.22 / −0.19°, J0 0.06° 이하 → **63 mm가 데이터와 맞는다**
  - **09:51 로봇3 (적용된 결과는 카메라 자유)**: 63 mm에서 Step 1.5 맞춤 오차 0.751 mm(자유 0.186, 67 mm 0.497), 카메라 z·x가 범위 위쪽 끝. 자유 해의 높이 71~73 mm → **이 로봇은 63 mm와 8 mm 이상 어긋난다**. 관절이 가정한 높이에 민감(J6 최대 2.85°, J0 0.6~0.9°). 원인 미확인 (브라켓·헤드 조립 차이 가능성, 확인 필요)
  - **두 로봇 모두 카메라 x가 위쪽 끝(44.3 mm)에 걸린다**: 자유 해는 45.0~45.3 mm. 설계 43.3 mm 기준 +1 mm 범위가 좁을 수 있다 (사용자 판단)

---

## 2026-10-14 — 사용자 확인·결정과 조사 2건 (코드 변경 없음)
> 검증 수준 ⚪ (코드 분석) · J4/J6 수치는 회전 행렬 계산
- **실화면 확인 (사용자)**: 내부 파라미터 보정 화면(시리얼 등록 표시, 계산 중 표시, 저장 직후 등록 반영)이 실제 카메라에서 잘 동작함 → 이 항목은 🟢
- **결정 (사용자)**: `RUN CALIBRATION` 같은 영어 고정 문구는 번역하지 않는다
- **v1.3 방향 정정 (사용자)**: "v1.2와 연산 방식을 맞춘다"는 원 피팅·최적화 같은 로우단 얘기다. v1.3은 팔꿈치만 양옆이 동축이고 손목은 축이 서로 달라서, 손목은 직교성 기반(v1.2 J6 방식)을 따른다. J6도 구한다. → cps C23의 "반지름 기반 J5" 권고는 철회
  - 코드 확인: v1.3 J6(`wrist_roll_v13`)은 v1.2 J6 방향 방식과 같은 분기(`JointCalibrator.compute_calibration_results`)에서 이미 구한다. v1.3 J5(`wrist_pitch_v13`)도 J6 스윕 축과 J4 스윕 축의 각도(직교성)로 구한다. 남은 문제는 짧은 호에서 축 추정이 잡음에 약한 것(시뮬 0.43°/스윕)이고, v1.2 J6는 09-22부터 브라켓 스윕(J6 ±30°)에서 축을 얻는 점이 다르다
- **J4/J6 맞바뀜의 실제 영향** (v1.2 손목 Z-Y-Z, J4 +1° / J6 −1°): J5 0°에서 0, 30°에서 방향 0.52° / 100 mm 앞 0.87 mm, 90°에서 1.41° / 1.75 mm. 손목을 편 자세에서만 완전히 상쇄된다
- **조사 1 — `marker_detection` → `calibration_optimizer` import**: `SimulationModel`(시뮬 센서 모델)이 `make_transform`·`compute_fk`·`so3_exp`를 쓴다. 쓰는 곳은 `Marker_Transform`의 시뮬 분기 2곳(생성)과 1곳(호출), `step2.py` 1곳(읽기), `scratch/validate_wrist_diversity_sim.py`. 수정안은 보고에 있음 (미착수)
- **조사 2 — Python 기준 버전**: `.venv` 3.11.9. 설치된 numpy 2.4.6·scipy 1.17.1·matplotlib 3.11.1은 Python 3.11 이상만 지원한다. `requirements.txt`는 하한만 적혀 있어 3.10에서 설치하면 더 옛 버전으로 내려간다. Windows 빌드 스크립트는 3.11 → 3.10 → 3.12 순으로 찾고, Linux 스크립트는 `python3`를 버전 확인 없이 쓴다
- **발견**: Home Offset Reset의 자세·관절 한계 검증(`core/robot/home_offset.py:381`)이 매번 예외로 건너뛰어진다. `get_limit_q_lower`에 로봇 상태를 넘기는데 SDK는 dynamics state를 요구한다 (10-08 17:34 로그의 `[WARN] Joint limit validation exception`). 그 아래의 어깨 롤·팔꿈치 방향 검사도 같이 실행되지 않는다 → 같은 날 수정 (아래)

## 2026-10-14 — 사용자 "전부 진행": 리셋 검증 수정, 시뮬 모델 분리, Python 3.11/3.12, v1.3 손목 축 비교
> 검증 수준 🟠 (G2 통과 312 / 실패 0 / 오류 0 / skip 0, 로그 `archive/test_logs/g2_20261014_sim_model_reset_check_python.log`) · 실기·docker 시뮬·Linux 빌드는 미검증
- **리셋 검증 수정**: `validate_home_offset_joint_limits`가 `dyn_model.make_state(["link_torso_5", "ee_right"], model.robot_joint_names)`로 한계를 읽는다. 이제 자세 방향(5° 여유)과 준비 자세의 관절 한계 검사가 실제로 돈다 → 잘못된 자세면 리셋이 막힌다. 예외가 나면 여전히 경고만 하고 통과시킨다(기존 동작 유지). 테스트 3개 (`tests/test_system_regressions.py` `TestHomeOffsetResetValidation`)
- **시뮬 모델 분리**: `SimulationModel`·`load_truth_config`·`uses_head_camera`를 `core/marker_detection.py`에서 `core/calibration/simulation.py`로 옮김. `Marker_Transform(simulation_factory=...)`로 받고, `CalibrationCore.connect_camera`가 `SimulationModel.create`를 넘긴다. 없이 시뮬 모드가 되면 명시적 오류. `marker_detection.py`의 고아 import(`FileStorage`, `yaml`, `json`, `hashlib`, `dataclass`, `Path`) 삭제. `tests/test_module_boundaries.py`에 "comm 모듈은 `core.calibration`을 import하지 않는다" 추가, `.claude/rules/architecture.md`의 알려진 위반을 해결로 변경. `scratch/validate_wrist_diversity_sim.py` import 수정. `scratch/validate_step1_2_sim.py`는 원래부터 없는 모듈(`core.paths`, `core.robot_motion`)과 Linux 경로를 써서 이미 동작하지 않는 상태 (손대지 않음)
- **Python**: `setup.py` `python_requires=">=3.11,<3.13"`. Windows 빌드 스크립트는 3.11 → 3.12만 찾음 (3.10 제외). Linux `build_linux.sh`는 python3.12 → python3.11 → python3 중 3.11~3.12를 골라 `.venv`를 만들고, 없으면 deadsnakes 설치 안내와 함께 중단. `run_linux.sh`도 3.11~3.12가 아니면 중단. Ubuntu 22.04에서 빌드하면 24.04에서도 실행된다(glibc). **3.12로는 아직 한 번도 테스트하지 않았다**
- **v1.3 손목 축 비교** (`scratch/v13_wrist_axis_estimators.py`, 합성, 200회): 실기 10-08 17:50 브라켓 스윕의 프레임당 잡음은 위치 0.075~0.140 mm(RMSE), 방향 0.2~0.85°/축. `config/simulation.yaml`의 방향 잡음은 0이라 docker 시뮬은 방향 기반 추정을 과하게 좋게 본다
  - J5 추정(스윕 쌍당 흩어짐): ±15/±15 위치 법선 0.42~0.44°, 방향 축 0.34° / J6 ±22.5·J4 ±30 위치 법선 0.13° / ±30/±30 위치 0.10°, 방향 0.28°
  - 결론: 직교성 방식은 그대로, 축은 지금처럼 위치 원 피팅(v1.2와 같은 로우단)으로 구하고 호를 넓히는 것이 효과가 크다. 방향 기반 축은 실기 잡음에서 오히려 나쁘다. 시야 여유(10-06 확인): J6 ±22.5° 62 px, ±30° 20 px, J4 ±30° 126 px 이상
  - 제안: `CalibratorBase.JOINT_CONFIGS["wrist_pitch_v13"]` 스윕 범위 J6 15 → 22.5°, J4 15 → 30°. 수렴 기준(0.06°)은 그대로. 예상 5쌍이면 통과(8회 상한 안)

## 2026-10-14 — v1.3 J5 단계 스윕 범위 확대 적용 (사용자: "스윕은 한번 올려봐")
> 검증 수준 🟠 (docker 시뮬 v1.3 Full Auto 1회) · 실기 미검증
- **변경**: `core/calibration/CalibratorBase.py` `wrist_pitch_v13` `sweep_range_A` 15 → 22.5 (J6), `sweep_range_B` 15 → 30 (J4). 수렴 기준·반복 상한은 그대로
- **시뮬 1회** (`scratch/validate_full_v13_sim.py`, 출력 `archive/reports/sim_v13_full_20261014_154328/`, 위치 잡음 0.1 mm·방향 잡음 0): Full Auto 완주 1217 s
  - J5 단계: 오른팔 4회째 수렴 1.921° (정답 대비 −0.19°), 왼팔 4회째 수렴 −3.589° (정답 대비 0.000°). 10-06(±15°)에는 왼팔이 8회 안에 수렴하지 못했다
  - J6 단계(`wrist_roll_v13`, 범위 그대로): 양팔 4회째 수렴. Step 1 J6는 정답 대비 약 0.75° (방향 방식, 브라켓 장착 오차가 기준)
  - J3: 양팔 정답 대비 0.05°
- **Step 1.5·Step 2는 이 시뮬로 평가할 수 없다**: `config/simulation.yaml`의 정답 카메라 `[0.050, 0.012, 0.060, -90.3, 0.6, -90.4]`가 설계 위치 고정 모드의 범위(x 40.3~44.3, y 9.0, z 66.5~67.5 mm, 각도 고정) 밖이다. 그래서 카메라가 범위 끝에 걸리고(x +1.0, z +0.5) 그 오차가 관절로 들어갔다: J4 R −2.49 / L +1.03, J6 +0.75 / +1.68, J0 L −1.33, J2 −0.94 / −1.47, 헤드 pan −0.70 / tilt +1.13°. 시뮬 정답 카메라를 설계 범위 안으로 옮겨야 Step 2까지 평가된다 (결정 대기)
- **G2**: 통과 312 / 실패 0 / 오류 0 / skip 0. 로그 `archive/test_logs/g2_20261014_v13_sweep_range.log`
- **이 결과가 의미하지 않는 것**: 1회 실행이라 수렴 횟수나 정확도를 통계로 말하지 않는다. 시뮬은 방향 잡음 0이라 J6(방향 방식)는 실기보다 유리하다. 실기에서 넓어진 손목 동작의 충돌·시야는 확인하지 않았다 (시야 여유는 10-06 핀홀 계산)

---

## 2026-10-08 — 새 로봇(v1.2, 헤드 있음) 17:50 실행 점검 (오프라인, 적용은 이미 끝남)
> 코드 변경 없음 (분석 스크립트 1개) · 검증 수준 🟠 (저장된 데이터셋으로 오프라인 재계산, 실행 1건) · 실기 추가 확인 없음
- **상황**: 사용자가 17:34 Home Offset Reset → Full Auto → 17:53 Optimized 적용까지 진행. **이 실행의 reset baseline은 믿지 않는다 (사용자)**. baseline은 리셋 직전 자세를 이전 영점 기준으로 읽은 값이라, 손으로 맞춘 리셋 자세나 이전 영점이 틀리면 의미가 없다
- **보존**: 적용 뒤 `config/home_reset_baseline.json`은 0으로 지워졌다 (정상 동작). 적용 전 값은 결과 JSON 안에 남아 있고, 실행 기록 전체를 `archive/runs/20261008_1750_newrobot_head/`에 복사했다 (38개 파일)
- **이 실행은 카메라 설계 위치 고정 모드의 첫 실기 실행**이다 (D405 x 43.3 [−3, +1], y 9.0 고정, z 67.0 ±1 mm, 각도 고정, J6는 Step 2에서 풂)
- **결과 (팔 오프셋, deg)**: R `+1.07 +1.99 −0.53 +2.12 +1.73 −0.02 −0.43`, L `+1.87 −2.67 +0.16 +2.10 −1.72 −2.54 +0.64`, 헤드 pan −0.82 / tilt +0.19, 카메라 [44.0, 9.0, 66.0] mm (z 하한에 걸림)
- **자체 일관성**: 준비 자세에서 모델과 측정의 차이 16~17 mm → 0.06 / 0.11 mm, 0.15 / 0.16°. 오프셋 보정 자세에서 두 마커의 좌우 대칭 잔차 0.14 mm, 0.20°. Step 2 가중 잔차 0.73σ
- **재계산** (`scratch/step2_rerun_20261008_1750.py`, 출력 `archive/reports/step2_rerun_20261008_1750/`): 실행 조건 그대로 풀면 저장 결과와 0.000° 일치. 카메라 범위를 바꾼 9가지 경우에서

| 관절 | 변동 폭 | 판단 |
|---|---|---|
| J1, J3, J5 | 0.1° 안팎 | 안정 (J3·J5는 Step 1 값에 고정) |
| J4 | R +1.50~+1.88, L −1.53~−1.78 | 안정. 좌우 거울 대칭 ±1.7°는 데이터가 요구하는 값 |
| J0 | R +0.90~+1.30, L +1.68~+2.21 | ±0.2~0.3° |
| J2 | R −0.61~−0.33, L +0.01~+0.20 | ±0.15° |
| J6 | R −0.65~+0.19, L +0.16~+0.83 | **±0.4° 불확실**. 카메라 높이와 맞바뀜 (z 67 → 64 mm에 약 0.7°) |

- **카메라 높이**: 범위를 풀면 z 64.2 mm, x 44.3~45.2 mm를 원한다 (설계 67.0 / 43.3). 위치 잔차는 0.106 → 0.081~0.087 mm로 조금 줄지만 가중 잔차는 0.72~0.74σ로 같아서 데이터로는 가려지지 않는다
- **J6 추정 세 가지가 0.6° 범위로 갈린다**: 오른팔 Step 1 방향 −0.35 / 위치 −0.96 / Step 2 −0.43, 왼팔 +0.31 / +0.12 / +0.64 (결과 JSON 부호)
- **왼팔 J5 2.53°**: Step 1에서 세 번의 스윕(2.569, 2.572, 2.497)으로 확인된 값. 오른팔은 0.04°
- **헤드 pan**: Step 1.5 −1.39° → Step 2 −0.82° (0.57° 차이). Step 1.5는 카메라 x·z가 범위 끝(+1.0, −1.0 mm)에 걸린 상태
- **이 결과가 의미하지 않는 것**: 양팔이 같은 방향으로 틀어지는 공통 오차(J0 공통, 헤드 tilt, 카메라 앞뒤)는 이 데이터만으로 검증되지 않는다. 확인 자세 육안 판단이 필요하다. 다른 로봇·다른 리셋의 결과와 값 자체를 비교하지 않았다
- **결정 (사용자, 같은 날)**: 실제 브라켓 높이는 67 mm가 맞다. 확정 못 하는 것은 x뿐이므로 D405 높이 범위를 ±1 mm → **±0.5 mm**로 좁힌다. `config/camera_info.yaml` `D405.mount_to_cam_position_range_m` = `[[-0.003, 0.001], [0.0, 0.0], [-0.0005, 0.0005]]`, `tests/test_camera_mount_constraint.py`의 출하값 단언 1개 수정
- **±0.5 mm로 이 실행을 다시 풀면** (오프라인): 카메라 [43.75, 9.00, 66.50] mm (z는 여전히 하한), 팔 오프셋 변화 최대 0.11° (J6: R −0.43 → −0.54, L +0.64 → +0.74), 나머지 0.06° 이하, 위치 잔차 0.106 → 0.115 mm. **이 값은 로봇에 적용되지 않았다** (로봇에는 ±1 mm 결과가 적용된 상태)

---

## 2026-10-08 — 내부 파라미터 저장 직후 "등록 여부"가 재연결 전까지 안 바뀌던 버그 수정 (사용자 보고)
> 검증 수준 🟠 (오프라인 테스트) · 실제 카메라로는 확인 전 · G2 결과는 아래
- **증상**: 보정 후 저장해도 "등록되지 않은 시리얼 번호 / 기본 보정값 사용 중"이 유지되고, 카메라를 재연결해야 등록된 것으로 바뀜
- **원인 (확인)**: `save_intrinsics_calibration`이 상태(`intrinsics_serial_matched`, `intrinsics_source` 등)를 `core.observer`에 대입했는데, 이것은 `ObservationSource` 래퍼다. 읽기는 `__getattr__`로 엔진(`Marker_Transform`)에 넘어가지만 대입은 래퍼에만 남아서, 엔진이 답하는 `get_camera_info()`는 옛 값을 계속 돌려줬다
- **보정값 자체는 원래도 저장 직후 적용되고 있었다** (`apply_calibrated_intrinsics`를 엔진에 직접 호출). 틀렸던 것은 상태 표시와 엔진의 출처 정보뿐
- **수정**: 저장 뒤 재연결과 같은 경로로 파일을 다시 읽고 적용한다. `Marker_Transform.reload_saved_intrinsics()`(상태 초기화 → `_resolve_camera_intrinsics()` → 적용) 신규, `CalibrationCore.apply_camera_intrinsics(path)` → `reload_camera_intrinsics()`로 교체, UI의 래퍼 대입 6줄 삭제. 카메라 스트림은 다시 열지 않는다
- **테스트**: `test_a_save_while_connected_is_recognized_without_a_reconnect`(신규, 실제 `ObservationSource` 래퍼로 재현), `test_the_core_reloads_it_into_the_live_observer`(기존 테스트를 새 메서드에 맞춤)
- **파일**: `core/marker_detection.py`, `core/calibration/calibration_core.py`, `main_ui.py`, `tests/test_tabs_and_camera_serial.py`, `tests/test_camera_model_config.py`
- **G2**: 통과 308 / 실패 0 / 오류 0 / skip 0. 로그 `archive/test_logs/g2_20261008_intrinsics_reload.log`

---

## 2026-10-08 — 내부 파라미터 보정 화면: 시리얼 등록 표시, 계산 중 표시, 한·영 혼합 문구 분리 (사용자 요청)
> 검증 수준 🟠 (오프스크린 UI 테스트와 화면 캡처) · 실제 화면·실제 카메라로는 확인 전 · G2 결과는 아래
- **시리얼 등록 표시**: 카메라 탭 상단 줄이 yaml 파일 이름 대신 등록 여부를 보여 준다. 이 카메라(S/N) 전용 보정 파일을 쓰는 중이면 초록(`#00e676`) "등록되어 있는 시리얼 번호입니다.", 아니면 빨강(`#ff1744`) "등록되지 않은 시리얼 번호입니다. 기본 보정값을 사용 중입니다.", S/N을 못 읽으면 빨강 "시리얼 번호를 읽을 수 없습니다. …". 위자드 1-2(내부 파라미터 보정) 화면 상단에도 같은 줄을 넣었다. 위자드 1-1의 "인식된 카메라" 상자(파일·출처 표시)는 그대로
- **계산 중 표시**: RUN CALIBRATION과 16번째 캡처 뒤 자동 계산 모두에서, 워커가 결과를 돌려줄 때까지 "보정값을 계산하고 있습니다…" 문구와 무한 진행 막대를 보이고 캡처·실행·저장·리셋 버튼을 잠근다 (카메라 탭과 위자드 양쪽). 잠금 전 상태로 되돌리므로 저장 버튼은 성공 전까지 꺼져 있다. 계산 중에는 C 키 캡처도 무시한다
- **같이 고친 것**: 위자드 `step1_run`·`step1_capture`가 워커 시작 직후 `rms_error`를 읽어 이전 실행 값이나 "Calibration Failed"를 잠깐 표시하던 것을 없앴다 (결과는 `_on_intrinsics_finished`가 표시)
- **한·영 혼합 문구**: 영어 값에 섞인 한국어 1건(`chk_confirmed`), 한국어 값의 영어 괄호 병기 12건(Rollback, Apply, Close, Live Feed, Shoulder Roll, Elbow, Zero Pose, Home Position, Home Pose, Teaching Button, NEXT, Direct Teaching), `CONNECT (로봇 연결)`, 언어 콤보 `한국어 (Korean)`, 로그 3줄의 한국어 괄호를 분리했다. Home Offset Reset 안내의 `[확인 (OK)]`는 실제 버튼 이름(`적용`/`Apply`)으로 바꿨다
- **남긴 것 (의도)**: 파일 이름(`camera_intrinsics.yaml` 등), 버튼 이름과 같은 상태 이름 `Baseline`·`Optimized`, 마커에 인쇄된 `Right`/`Left`, `Home Offset Reset`·`Step 1/2`·`Control Manager` 같은 기능 이름
- **고치지 않은 것**: 번역을 거치지 않는 영어 고정 문구 (카메라 탭의 `RUN CALIBRATION` 등, 위자드 상태 줄 `Status: …`, 메시지 상자 일부). 언어를 한국어로 바꿔도 영어로 남는다. 범위가 커서 따로 결정 필요
- **배포 주의**: exe 옆에 예전 `config/ui_config/i18n.yaml`이 남아 있으면 기존 키는 그 파일 값이 우선이라(`LanguageManager._fill_from_bundle`은 없는 키만 채움) 혼합 문구 수정이 반영되지 않는다. 새 키(등록 표시·계산 중)는 반영된다. 헤더 키는 그래서 `main_tabs.camera_info` 대신 새 키 `main_tabs.camera_unit`으로 만들었다
- **파일**: `ui/busy_indicator.py`(신규), `main_ui.py`, `ui/wizard_widget.py`, `config/ui_config/i18n.yaml`, `tests/test_tabs_and_camera_serial.py`(테스트 9개 추가)
- **G2**: 통과 307 / 실패 0 / 오류 0 / skip 0 (직전 298 + 9). 로그 `archive/test_logs/g2_20261008_intrinsics_ui.log`

---

## 2026-10-08 — 백업·디버깅·보고서 폴더를 `archive/`로 이동 (사용자 요청)
> 파일 이동만 (삭제 없음, 1069개·186.3 MB 전후 동일) · 코드 변경은 경로 문자열뿐
- **배경**: `result/` 안과 저장소 루트에 백업·디버깅 폴더가 쌓여 있었다. 앱이 쓰는 것은 `result/`의 `result_step2/`·`result_txt/`·`result_img/`·`result_log/` 4개뿐이고 나머지는 디버깅 때마다 따로 남긴 것
- **이동** (이 줄 아래의 옛 항목들은 옛 경로로 적혀 있다)

| 옛 위치 | 새 위치 |
|---|---|
| `result_backup/<실행>` | `archive/runs/<실행>` |
| `result/` 안의 디버깅 폴더 18개와 낱개 파일 12개 | `archive/debug_2026-09/` |
| `result_d435/` | `archive/result_d435/` |
| `result (260921-2차)`, `result (260921-3차)`, `result (4)` | `archive/result_copies/` |
| `reports/` | `archive/reports/` |
| `backups/` | `archive/test_logs/` |

- **그대로 둔 것**: `result/result_step2/`의 과거 결과 (앱이 최신 결과를 여기서 찾음)
- **경로 수정**: `.claude/hooks/protect_originals.py`(보호 대상: `archive/runs/`, `archive/result_d435/`, `archive/result_copies/`, `archive/debug_2026-09/`), `scratch/calibration_report.py`, `step2_rerun_20261008.py`, `validate_full_v13_sim.py`, `verify_j4_j5_misalignment.py`, `wrist_axes_check.py`
- **git**: 추적 중이던 `result_d435/` 168개와 `result_backup/` 19개가 삭제로 보이고 `archive/`는 새 폴더로 보인다 (커밋은 사용자 판단). 사용자가 추적 파일도 함께 옮기라고 지시
- **문서에 아직 없는 것**: 노션·볼트 문서와 에이전트 메모리의 옛 항목은 옛 경로 그대로다 (메모리에는 대응표 `archive-layout` 추가)

---

## 2026-10-08 — 새 로봇(M v1.2, 헤드 있음, 카메라 브라켓 +6 mm) 실행의 J0 분석 (오프라인)
> 코드 변경 없음 (분석 스크립트만) · 검증 수준 🟠 (저장된 데이터셋·헤드 스윕으로 오프라인 재계산, 실행 1건) · 실기 추가 확인 없음
- **실행 (사용자)**: 09:34 Home Offset Reset → Full Auto 완주(09:51), 수동 노출 6000. 적용 기록은 화면 로그에 없음 (확인 자세 미리보기만). 사용자 관찰: 어깨 pitch(J0)가 조금 뒤로 감
- **백업**: `result_backup/20261008_robot3_head_newbracket/` (스윕·로그·Step 1.5·Step 2 결과와 데이터셋·화면 로그·설정)
- **결과 값**: J0 양팔 −0.69° (리셋 기준 대비 오른 −0.71 / 왼 −1.04°). 헤드 tilt −1.60°, 카메라 `[40.7, 7.5, 71.1]` mm, 위치 잔차 0.241 mm
- **재계산** (`scratch/step2_rerun_20261008.py`, `step15_rerun_20261008.py`, 출력 `reports/step2_rerun_20261008/`): 원 실행과 0.000° 일치. 모델은 v1.2 시뮬에서 받은 URDF (시뮬 앱이 약 35초 뒤 스스로 종료돼 URDF만 받아 접속 없이 계산)
	- 헤드 tilt 기준값 ±: J0가 양팔 같이 약 0.75°/° 움직이고 잔차는 거의 그대로 (tilt −1.0 → +2.0 이동에 0.277 / 0.256 / 0.241 / 0.233 / 0.231 / 0.235 / 0.245 mm). tilt를 +1.0~+1.5° 옮기면 J0가 리셋 기준과 같아진다
	- 카메라 높이 고정 스캔: 69~73 mm 밖에서는 잔차가 상한(1.0 mm)에 닿음 → 데이터가 높이를 약 71 mm로 정한다. 높이를 바꿔도 J0는 −0.7~−0.9°. 카메라 시작값을 CAD+6으로 바꿔도 결과 동일
	- Step 1.5 재계산: 광축 회전을 공칭에 고정해도 tilt −1.606 → −1.613°, J0 변화 0.04° 이내. 카메라 높이를 63 mm(CAD+6)로 고정하면 잔차 0.19 → 0.75 mm
	- Step 1.5 로그의 "Out/Back tilt −0.09 / −0.13°"는 잘못된 값이다 (사용자 지적으로 확인, `scratch/step15_direction_check_audit.py`). 가는 방향·오는 방향 풀이가 평가 횟수 상한(1300회, status 0)에서 멈춘 것이고, 그 잔차(0.273 mm)가 전체 해를 같은 절반에 넣은 잔차(0.181 / 0.195 mm)보다 크다. 전체 해에서 시작해 다시 풀면 −1.607 / −1.592°로 전체 풀이(−1.606°)와 일치한다
	- Step 1.5에서 tilt를 고정해 본 잔차: −1.6° 0.188 mm(최소), ±0.4° 0.194 mm, −0.8° 0.213 mm, 0° 0.272 mm, −3.0° 0.257 mm. 전체 풀이는 시작값을 바꿔도 −1.606°로 수렴
- **판단**: J0는 카메라 높이나 카메라 각도가 아니라 Step 1.5가 준 헤드 tilt(−1.6°)에 묶여 있다 (확인). 그 tilt는 Step 1.5 데이터가 약 ±0.4° 안에서 정한다 (J0로는 약 ±0.3°). 처음 보고에서 "tilt 추정이 단단하지 않다"고 한 것은 수렴하지 않은 Out/Back 값을 근거로 한 잘못된 판단이었다. J0 결과가 틀렸다는 근거는 이 데이터에서 찾지 못했다
- **결함 수정 (사용자 지시)**: Step 1.5의 방향 일관성 점검(Out/Back)이 실기 데이터에서 수렴하지 않은 값을 보고했다 → 절반 풀이를 전체 해에서 시작, `direction_consistency.converged` 추가, 수렴 실패 시 경고 (`core/calibration/HeadCameraCalibrator.py`). 테스트 1개 추가 (`tests/test_step1_head_contracts.py`). 이번 데이터 재계산에서 Out/Back −1.61 / −1.59°로 전체 풀이와 일치
- **G2**: 통과 278 / 실패 5 / 오류 0 / skip 0 (`backups/g2_20261008_direction_check.log`). 실패 5건은 모두 `tests/test_camera_model_config.py`: 사용자가 `config/camera_info.yaml`의 카메라 높이를 0.057 → 0.067로 바꿨고(D405·D435 둘 다, 11:37) 테스트는 실제 `config/`를 복사해 쓰면서 기대값 0.057을 상수로 갖고 있다. 이번 수정과 무관. 테스트 기대값 처리와 D435 값은 사용자 확인 대기
- **사용자 추가 정보 (10-08)**: 결과를 로봇에 적용했다. 카메라의 실제 설계 높이는 67 mm로 보인다. 카메라 브라켓 각도 오차는 0.1° 미만
- **추가 재계산** (`scratch/j0_j6_camz_20261008.py`, `j6_both_arms_20261008.py`, 값은 리셋 기준 대비)
	- J0 공통모드 댐핑: 가중 0 / 200 / 2000에서 J0 변화 0.01° 이내 (이 데이터에서는 사실상 작동하지 않음). 20000에서 −0.62 / −0.95, 200000에서 −0.30 / −0.64이고 그만큼 J4(+1.08 / −1.36)로 넘어간다
	- J6 기준값을 한 팔씩 옮기면: 오른팔은 +1.0°에서, 왼팔은 −1.0~−1.5°에서 잔차 최소 (0.241 → 0.229 / 0.220 mm)
	- 양팔 J6를 함께 바꾸면: 위치 방식 값(패스 2)일 때 잔차 0.211 mm·카메라 높이 68.4 mm·J0 −0.50 / −0.75, 리셋 기준 값일 때 0.208 mm·67.7 mm·J0 −0.43 / −0.72 (원 실행 0.241 mm·71.1 mm·−0.71 / −1.04)
	- 카메라 높이 67 mm: Step 1.5에서 고정하면 잔차 0.19 → 0.47 mm, Step 2에서 카메라를 풀어 두면 71.1 mm로 돌아가 J0 변화 없음. 원 실행의 J6 그대로 높이만 67 mm로 고정하면 잔차가 상한(1.0 mm). J6를 위치 방식 값으로 두고 67 mm로 고정하면 0.222 mm, J0 −0.57 / −0.80, tilt −2.03°
- **판단 (추정 포함)**: 이번 실행의 Step 1 J6(방향 방식)가 양팔 약 1° 치우쳤을 가능성이 높다. 근거: 두 방식의 차(오른 0.47°, 왼 1.12°), J6를 위치 방식·기준 쪽으로 옮기면 Step 2 잔차가 12~14% 줄고 카메라 높이가 설계값 67 mm 근처(67.7~68.4)로 온다, 원 실행에서 J4가 J6를 그대로 상쇄(오른 +0.80 / −0.80, 왼 −1.09 / +1.26). J6가 틀리면 J4·J2와 카메라 높이, J0(약 0.3°)로 번진다. 한계: Step 2는 J6를 약하게만 구분한다(1°당 잔차 5~9%), 실행 1건, 브라켓 yaw와 J6는 구분되지 않는다
- **남은 J0**: J6를 바꿔도 −0.4 / −0.7°가 남고 이것은 헤드 tilt(−1.6°)와 묶여 있다
- **테스트 수정 (사용자 지시)**: `tests/test_camera_model_config.py`의 공칭값 상수 4개를 `config/camera_info.yaml`에서 읽도록 바꿈. G2 통과 283 / 실패 0 / 오류 0 / skip 0 (`backups/g2_20261008_camera_info_tests.log`). 이 통과는 0.067이라는 값이 맞는지를 검증하지 않는다
- **J6 방식 (사용자 의견, 결정 대기)**: 위치 방식은 D435 때 계속 문제가 돼 방향 방식을 택했던 것으로 기억 → 방식을 통째로 바꾸지 않는다. "두 후보로 풀어 경고만" 안은 사용자가 의미 없다고 기각
- **추가 확인 (사용자 질문: 데이터가 잘 측정됐다면 계산 방식 문제인가)**
	- 기각한 가설: "J6는 맞고 브라켓 옆방향 위치만 틀렸다". J6를 그대로 두고 브라켓 x를 두 방식의 차가 뜻하는 만큼(±0.45 / ±1.06 mm) 옮겨도 잔차 0.241 → 0.231 mm, 카메라 높이 71.1~71.6 mm로 거의 그대로 (`scratch/j6_bracket_x_20261008.py`)
	- Step 2 64자세의 자세별 잔차 (`scratch/step2_marker_residuals_20261008.py`): J6 축 둘레 방향 잔차가 평균 ±0.1° 이내, 표준편차 0.21°, 범위 −0.8~+0.7°. 오른팔은 마커의 화면 가로 위치를 따라 변함 (상관 +0.48, 0.05°/°). 마커는 화면 중심에서 가로 +20~+30°(오른) / −25~−16°(왼)에 있고, 64자세의 J5 범위는 −46~−36°(10°)뿐
	- J6를 위치 방식·기준 값으로 바꾸면 전체 방향 잔차 rms 0.39 → 0.33~0.34°(오른), 0.42 → 0.35~0.36°(왼), 위치 잔차 rms 왼 0.40 → 0.31~0.32 mm
	- 카메라를 `[42.0, 8.1, 67.0]` mm에 고정하고 양팔 J6를 방향 방식 값에서 d만큼 옮기면(오른 +d, 왼 −d) 잔차: d=0 0.568 / 0.25 0.265 / 0.5 0.239 / 0.75 0.226 / 1.0 0.220 / 1.25 0.219 / 1.5 0.222 / 2.0 0.238 mm (`scratch/j6_with_camera_height_20261008.py`). 카메라 높이를 알면 J6가 데이터로 정해진다 (카메라를 풀어 두면 1°당 12%뿐). 이때도 J0는 −0.6 / −0.7°로 남는다
- **제시한 방향 (결정 대기)**: J6를 마커 방향(가장 약한 측정) 하나에 걸지 말고, 정밀해진 카메라 브라켓의 위치를 외부 기준으로 삼아 Step 2가 J6를 정하게 하는 안. 전제: 카메라 설계 위치값이 맞아야 한다 (높이 67 mm는 사용자 확인, x는 파일 47 mm인데 데이터는 41~42 mm)
- **사용자 정보 (10-08)**: 광학 중심 기준 위치는 확정하기 어렵다. x 47 mm는 카메라 모듈 면 기준, y는 baseline 기준. 질문: 카메라 각도만 확실히 고정하는 것은 의미가 없나
- **카메라 앞뒤 위치(x) 스캔** (`scratch/j0_vs_camera_x_20261008.py`, 카메라를 `[x, 9.0, 67.0]` mm·공칭 각도에 고정, 값은 리셋 기준 대비)
	- J6를 오른 +1.25 / 왼 −1.25° 옮긴 경우: x 41 / 42 / 43 / 44 / 45 / 46 / 47 mm에서 위치 잔차 0.262 / 0.222 / 0.209 / 0.202 / 0.211 / 0.271 / 상한(1.0) mm, J0 오른 −1.20 / −0.83 / −0.62 / −0.33 / +0.04 / +0.37 / (+0.25), 왼 −1.00 / −0.61 / −0.32 / −0.09 / +0.26 / +0.62 / (+0.46), tilt −2.26 / −1.85 / −1.57 / −1.27 / −0.85 / −0.46 / (−1.05)°
	- J6를 원 실행 값으로 둔 경우: 방향 잔차가 0.40~0.52°로 하한(0.30°)을 넘고 J4가 ±1.5~2.0°
	- 확인: J0·헤드 tilt·카메라 x가 한 방향으로 묶여 있다 (x 1 mm ≈ tilt 0.37° ≈ J0 0.33°). 카메라 각도는 이미 고정돼 있어 이 묶음에 들어 있지 않다. 사용자가 준 y·높이·각도를 고정하고 J6를 옮긴 조건에서 데이터는 x를 44 ± 1 mm(모듈 면에서 약 3 mm 안쪽)로 두고, 그때 J0는 −0.33 / −0.09°
	- 한계: 실행 1건, x의 참값(광학 중심)은 모른다. Step 1.5는 같은 스윕에서 x 45.4 mm·tilt −1.6°를 냈고 Step 2는 tilt −1.6°에 x 43 mm를 짝짓는다 (두 단계의 짝이 x로 약 2.4 mm 어긋남)
- **사용자 결정 (10-08)**: "카메라 y·높이·각도 고정, x는 좁은 범위, J6는 Step 2가 결정" 방향으로 설계안을 만든다. x 범위는 일단 5 mm. 사용자 자료: RealSense 문서의 깊이 시작점 D405 −3.7 mm (전면 유리 기준), 다만 컬러만 쓰므로 광학 중심은 확정하지 못함
- **설계안 제시**: 0단계 scratch 시제(프로덕션 무변경, 이 데이터셋으로 변형 비교) → 1단계 구현(카메라 모델별 선택 키, 없으면 지금 동작) → 2단계 G2·시뮬 → 3단계 실기(사용자). 보정 수학과 원칙 4(카메라 위치 ±25 mm 추정), 09-30 결정(J6 고정 완화 보류)을 바꾸는 변경
- **사용자 결정 (10-08, 진행)**: D405의 x는 43.3 mm(모듈 면 47 − 깊이 시작점 3.7)로 두고 +1 / −3 mm 범위, 범위는 yaml에서 수정 가능. 높이는 67 mm에 ±1 mm (벗어나는 결과를 보고 판단하기 위함). 새 브라켓은 D435에도 쓰므로 D435 높이는 수정된 0.067 그대로
- **0단계 시제 결과** (`scratch/proto_camera_design_20261008.py`, 값은 리셋 기준 대비)
	- Step 1.5 (같은 제약): tilt −1.259°, 카메라 `[44.3, 9.0, 68.0]` mm로 x·z가 범위 위쪽 끝, 잔차 0.468 mm (제약 없을 때 0.188 mm)
	- B 설계안 (카메라 제약, J6 자유): J0 −0.29 / −0.32, J4 +0.67 / +0.25, J6 +0.02 / −0.63, tilt −1.26°, 카메라 `[43.3, 9.0, 66.9]` mm(범위 안), 잔차 0.215 mm. J6를 Step 1 값에서 시작해도 같은 해 (C)
	- A (카메라 제약, J6는 Step 1 값 고정): J0 −0.03 / +0.18이지만 J4 +1.39 / −1.60, 방향 잔차 0.360°, 카메라 z가 위쪽 끝
	- D (y도 ±1 mm): y가 아래쪽 끝(8.0 mm), J0 −0.16 / −0.37, J6 +0.35 / −0.33, 잔차 0.207 mm
	- E (Step 2가 Step 1.5의 x 유지): J0 −0.04 / −0.07, 카메라 z가 아래쪽 끝
- **1단계 구현** (B 설계안)
	- `config/camera_info.yaml`: 카메라 모델별 선택 키 `mount_to_cam_position_range_m` (link_head_2 축별 `[low, high]`, m). D405에만 넣음: `mount_to_cam` x 0.047 → 0.0433, 범위 `[[-0.003, 0.001], [0, 0], [-0.001, 0.001]]`. 키가 없는 모델(D435)은 기존 동작
	- `core/calibration/data.py`: `load_camera_mount_constraint`(형식이 틀리면 ValueError), `format_mount_constraint`, `mount_range_ends`
	- `core/calibration/calibration_optimizer.py`: `QPCalibrationOptimizer(camera_pos_range_m=, anchor_joint6=)`. 범위가 있으면 카메라 세 축 회전 고정·위치는 범위 안, `anchor_joint6=False`면 J6의 Step 1 고정(±0.05°, 가중 1e7)을 걸지 않음
	- `core/calibration/HeadCameraCalibrator.py`: Step 1.5가 설계값을 `camera_info.yaml`에서 직접 읽어 같은 범위로 풀고(광축 회전 고정), 범위 끝에 닿으면 경고
	- `core/calibration/sequences/step2.py`: 제약이 있으면 카메라를 설계값에서 시작(Step 1.5의 카메라 위치는 넘기지 않음), 두 패스에 범위 전달, J6는 Step 2가 결정. 로그·결과 JSON(`camera_mount_constraint`)에 설계 대비 위치, 범위 끝 여부, Step 1의 J6 값
	- 양팔·헤드 있는 모드에만 적용 (단일 팔, 헤드 없는 모드는 기존 동작)
- **검증**: 🟠 오프라인. 키를 무시하면 원 실행과 0.000° 일치, 키가 있으면 시제 B와 0.0000° 일치 (`scratch/verify_camera_design_mode_20261008.py`). 새 테스트 `tests/test_camera_mount_constraint.py` 15개, G2 통과 298 / 실패 0 / 오류 0 / skip 0 (`backups/g2_20261008_camera_mount_constraint.log`)
- **하지 않은 것**: 2단계 시뮬(v1.3 시뮬만 가능, 시뮬 참값의 카메라가 설계값과 다르고 v1.3 Step 1 J5가 기본 잡음에서 멈추는 문제가 겹침), 3단계 실기. 노션·볼트 문서(원칙 4, J6 결정, 결과 파일 계약)에는 아직 반영하지 않음
- **검증 경계**: 근거와 확인이 모두 실행 1건(10-08 09:51)이다. 설계값(43.3 / 9 / 67 mm)이 맞다는 것은 이 결과로 주장하지 않는다. Step 1.5는 이 데이터에서 x·z가 범위 끝에 닿고 잔차가 2.5배라, 헤드 스윕 쪽은 설계값과 어긋난 채다 (원인 미확인)
- **미커밋 파일 (10-08 작업분)**: `core/calibration/data.py`, `calibration_optimizer.py`, `HeadCameraCalibrator.py`, `sequences/step2.py`, `config/camera_info.yaml`, `scripts/windows/start_sim_wsl.bat`, `tests/test_camera_mount_constraint.py`(새 파일), `tests/test_camera_model_config.py`, `tests/test_step1_head_contracts.py`, `scratch/*_20261008.py`, `docs/progress.md`
- **부수**: `start_sim_wsl.bat`에서 `--restart unless-stopped` 제거, 떠 있던 `rby1_sim`의 재시작 정책도 끔 (사용자 요청: WSL을 켤 때마다 시뮬이 켜짐)
- **미커밋 파일**: `scratch/step2_rerun_20261008.py`, `scratch/step15_rerun_20261008.py`, `scripts/windows/start_sim_wsl.bat`, `docs/progress.md`

---

## 2026-10-06 밤 — v1.3(헤드 있음) 시뮬 Full Auto: J5 단계가 잡음에 약함, 스윕 파일 이름 수정
> 커밋 기준 `a21bcb0` + 미커밋 변경 · 검증 수준 🟠 (시뮬 `127.0.0.1:50051`, `rby1-sim:0.10.6-m_v1.3`, 실행 3회 각 1번) · G2 통과 282 / 실패 0 / 오류 0 / skip 0 · 실기 미확인
- **사용자 요청**: v1.3·헤드 있는 구성으로 전 과정이 도는지 시뮬로 확인. 실행별 보관은 하지 않음(덮어쓰기 유지), 노출값은 건드리지 않음
- **방법**: `scratch/validate_full_v13_sim.py` (새 파일). GUI Full Auto와 같은 `CalibrationCore.run("full")`에 노출 확인용 준비 자세·마커 모니터·확인 자세 미리보기를 붙임. `config/`를 `reports/sim_v13_full_<시각>/config`로 복사하고 모든 경로를 그쪽으로 돌림 (저장소 `config/`·`result/` 변경 없음). Home Offset Reset·Apply는 돌리지 않음 (시뮬 센서가 참값 오프셋을 엔코더에 더하는 구조)
- **실행 3회** (시뮬 참값은 `config/simulation.yaml`, 위치 잡음 기본 0.1 mm)

| 실행 (`reports/sim_v13_full_…`) | 조건 | 결과 |
|---|---|---|
| `20261006_223236` | 기본 (잡음 0.1 mm, J5 단계 스윕 ±15°) | **Step 1 실패**: 왼팔 `wrist_pitch_v13` 8회 안에 미수렴. 오른팔 J5는 가까운 추정 3개의 표준오차 0.0589°로 통과 |
| `20261006_224948` | 잡음 0.02 mm (복사본 `simulation.yaml`만 수정) | 전 과정 완주. 실행 도중 소스 2개 수정(스윕 파일 이름) → 증거에서 제외, 참고만 |
| `20261006_231138` | 기본 잡음, J5 단계 스윕 J6 ±22.5° / J4 ±30° (스크립트 안에서만 변경) | 전 과정 완주. 왼팔 J5는 8회째에 통과 (추정 흩어짐 약 0.15°) |

- **원인 (시뮬에서 측정)**: v1.3 J5는 J6 원과 J4 원의 법선 사이 각도로 구한다 (detail 4.5.1, v1.2 J3·J5는 원 중심 방식). 호 30°에서 움직이지 않는 J4 원의 축이 스윕마다 최대 0.43° 흔들림 (rms 0.23°, 반지름 68 mm, 맞춤 오차 0.10 mm). 수렴 기준(표준오차 0.06°, 목표 0.3° 이내 추정만 평균)으로는 8회에 모이지 않음. process 09-15에 "v1.3 wrist_pitch 노이즈 약 1.3°" 기록이 이미 있음
- **합성 스윕** (`scratch/v13_j5_noise_vs_arc.py`, 잡음 0.1 mm, 30회): J5 추정 흩어짐 ±15° 0.43° / ±20° 0.23° / ±22.5° 0.17° / ±30° 0.09° / ±40° 0.05°. J4 원 반지름으로 구하면 0.01° (브라켓 x를 알아야 함: 1 mm ≈ 0.45°)
- **시야** (`scratch/v13_sweep_view_check.py`, 저장된 브라켓 스윕을 연장, D405 핀홀): J4 ±30°에서 마커 기울기 40~43°·여유 126 px 이상, J6는 ±22.5° 여유 62 px, ±30° 20 px, ±40° 화면 밖
- **완주한 실행의 참값 대비 (231138, 추정 − 참값)**: 오른팔 J0 −0.21 / J4 −0.51 / J5 +0.08 / J6 −0.75°, 나머지 0.08° 이내. 왼팔 J0 −0.28 / J4 +0.55 / J5 +0.01 / J6 +0.73°, 나머지 0.03° 이내. 헤드 pan −0.46 / tilt +0.33°, 카메라 4.2 mm / 0.5°. 사후 검증 모델-카메라 12~14 mm → 0.15~0.22 mm, 보정 자세에서 좌우 마커 차이 0.60 mm. J4·J6 차이는 시뮬에 넣은 브라켓 각도 오차(0.5~0.8°)와 같은 크기 (v1.3은 브라켓 roll·pitch를 설계값에 고정, 추정). 브라켓 위치는 설계값 그대로 나옴 (참값과 2.1~2.3 mm 차이)
- **코드 변경 (스윕 파일 이름)**: v1.3에서는 단계끼리 같은 축을 스윕해 파일을 서로 지웠다 (J6 단계의 J6 스윕을 J5 단계가, J5 단계의 J4 스윕을 팔꿈치 단계가 삭제). 관절 보정 스윕 파일 이름에 단계 이름을 넣음: `sweep_points_<팔>_<모드>_joint_A/B_axis_<n>.txt` (`JointCalibrator.sweep_points_tag`, `save_debug_points(file_tag=)`). 브라켓 스윕 파일(`…_marker_axis_<n>.txt`)은 그대로. 테스트 2개 추가 (`tests/test_run_records.py`). 231138 실행에서 단계별 파일이 따로 남는 것 확인
- **사용자 방향 (2026-10-06)**: 호 확대보다 v1.3이 v1.2의 연산 방식을 따르게 하고 싶다 (보정하는 축과 방식은 유지, 연산만 동일하게)
- **결정 대기**: v1.2의 중심 방식은 두 스윕 축이 평행(두 원이 같은 중심)이어야 하는데 v1.3 손목은 J6·J4가 직교(중심 약 130 mm 떨어짐)라 그대로 옮길 수 없다. 제시한 안: ① J5를 J4 원 반지름으로, 브라켓 위치를 v1.2처럼 반지름으로 (권장) ② 법선 방식 유지 + 호 확대 ③ 손목점 FK를 쓰는 일반화한 중심 방식 (Step 1 시점 FK가 13~14 mm 틀려 부적합, 계산값). J6를 브라켓 스윕에서 구하도록 옮길지도 확인 필요
- **검증 경계**: 시뮬은 마커를 합성하므로 입사각·검출 품질이 없고 잡음이 프레임마다 독립이다. 실기 v1.3에서 같은 크기로 흩어지는지는 모른다. 실행은 조건마다 1번이라 통과율을 말할 수 없다. 시야 계산은 렌즈 왜곡·검출 성능을 넣지 않았다
- **환경 메모**: WSL에 붙은 프로세스가 없으면 약 1분 뒤 WSL이 꺼지고 시뮬 컨테이너가 재시작된다. 실행 동안 `wsl -d Ubuntu-22.04 -e sleep <초>`를 띄워 두어야 한다
- **미커밋 파일 (이번 작업분)**: `core/calibration/CalibratorBase.py`, `core/calibration/JointCalibrator.py`, `tests/test_run_records.py`, `scratch/validate_full_v13_sim.py`·`joint_sweep_scatter.py`·`v13_j5_noise_vs_arc.py`·`v13_sweep_view_check.py`(새 파일), `reports/sim_v13_full_*`, `docs/progress.md`

---

## 2026-10-06 — 실행 기록을 result/ 안에 남기기 (노출값, 반복마다 스윕, 화면 로그, 실행 시점 설정)
> 커밋 기준 `a21bcb0` + 미커밋 변경 · 검증 수준 🟠 (오프라인 테스트) · 테스트 통과 280 / 실패 0 / 오류 0 / skip 0 · 실기 미확인
- **목표 (사용자 요청)**: 노출값을 기록에 남기고, 반복마다 스윕 원본을 남기고, 디버깅에 필요한 기록이 모두 `result/` 안에 있게 한다
- **배경**: 10-06 1차 실행의 팔꿈치 미수렴 때 반복 2회차 이후 스윕이 없었고, 2차 실행의 노출값은 어디에도 없었다. 팔별 브라켓 값·중단 사유는 화면 로그에만 나오고 창을 닫으면 사라졌다
- **변경**
	- 노출값: `BaseCalibrator.camera_exposure_record()` (자동 여부, 설정값, 설정 단위, 마지막 프레임의 실제 노출). 관절 보정 로그(`joint_calib_debug_*.txt`) 머리에 한 줄, Step 2 결과 JSON에 `camera_exposure`. 실제 카메라가 없으면(시뮬·테스트) 값을 만들지 않고 None
	- 반복마다 스윕: `JointCalibrator`가 관절 보정의 모든 스윕 원본을 `sweep_points_<팔>_joint_A/B_axis_<n>.txt`에 블록으로 이어 쓴다 (`=== NEW ITERATION ===` 구분, 블록마다 `# pass P, sweep N, staged offset X deg` 줄). 그림(디버그 플롯)은 지금처럼 첫 스윕만
	- 화면 로그: `core/storage.py`의 `SessionLog`가 화면에 나오는 모든 줄을 시각과 함께 `result/result_log/gui_log_<날짜>.txt`에 이어 쓴다 (`CONFIG_PATHS["log_dir"]`). `result_txt`와 다른 폴더라 스윕 시작 때의 .txt 정리에 지워지지 않는다. 저장이 실패하면 화면에 한 번 알리고 저장을 멈춘다. `--ui`(UI 전용) 모드에서는 저장하지 않는다
	- 실행 시점 상태: Step 2 결과 JSON에 `setting_snapshot`(그때의 `setting.yaml` 전체)과 `home_reset_baseline`(기준 파일 내용)을 넣는다. Home Offset Apply가 둘을 0으로 덮어도 결과 파일만으로 알 수 있다
- **테스트**: `tests/test_run_records.py` 9개 (로그 파일 형식, 화면 로그 저장과 실패 처리, 노출 기록, 모든 스윕에 저장 요청과 메모, 저장된 블록 형식), `tests/test_core_execution.py` 2개 (Step 2 결과의 새 항목). `tests/test_step1_head_contracts.py`는 `__init__`을 건너뛴 객체에 `marker_st = None` 한 줄 추가 (새 코드가 읽는 속성)
- **검증**: G2 280/0/0/0 (`backups/g2_20261006_run_records_2.log`). 테스트가 실제 `result/`·`config/`에 쓰지 않은 것 확인. 직전 실행에서 위 기존 테스트 1건 오류 → 테스트 준비 수정 뒤 통과
- **하네스 실동작**: `comment_check` hook이 실제 세션에서 동작함을 확인 ("Step 1 values"로 시작하는 주석 줄을 단계 나레이션으로 잡은 오탐 1건, 줄 시작을 바꿔 해결)
- **검증 경계**: 실제 카메라·로봇에서 노출값이 맞게 찍히는지, 스윕을 매번 저장하는 데 드는 시간(스윕당 FK 계산)이 반복 사이 대기를 눈에 띄게 늘리는지는 확인하지 않았다. `get_exposure()`는 카메라 스트림이 꺼져 있으면 (자동, 6000)을 돌려주는 기존 동작이 있어, 그 경우 기록도 그렇게 남는다
- **아직 result/ 밖이거나 남지 않는 것**: Step 2까지 가지 못한 실행의 `setting.yaml`·기준 파일 (화면 로그는 남음). 다음 실행이 `result_txt`의 파일을 덮어쓰는 것은 그대로다 (실행별 보관은 미구현, 사용자 결정 대기)
- **문서에 아직 없는 것**: 노션·볼트 harness의 "인터페이스와 데이터 계약 → 결과 파일" 목록에 `result/result_log/`, Step 2 결과의 새 항목, 스윕 파일의 블록 형식이 빠져 있다 (요청 시 반영)
- **미커밋 파일**: `core/calibration/CalibratorBase.py`, `core/calibration/JointCalibrator.py`, `core/calibration/sequences/step2.py`, `core/storage.py`, `main_ui.py`, `tests/test_run_records.py`(새 파일), `tests/test_core_execution.py`, `tests/test_step1_head_contracts.py`, `docs/progress.md`

---

## 2026-10-06 — 다른 로봇 헤드 없는 모드 재실행: 완주, 적용됨 (Step 1 21:41~21:50, Step 2 21:54, 적용 22:07)
> 코드 변경 없음 · 검증 수준 🟠 (결과·스윕 파일 오프라인 대조) · 실기 확인 자세 관찰은 사용자에게 아직 듣지 못함
- **실행 (사용자)**: 같은 로봇, 같은 Home Offset Reset(21:24:54) 뒤 다시 실행. 사용자 확인: 카메라와 브라켓은 이전 로봇에서 쓰던 것과 같은 것이다. 결과 `result_20261006_215403_948090.json`, 22:07에 적용 (`setting.yaml` Step 1 값과 baseline이 0이 됨)
- **백업**: `result_backup/20261006_robot2_headless_run2/` (스윕·디버그 21개, Step 2 결과와 데이터셋, 적용 뒤 `setting_2207.yaml`·`home_reset_baseline_2207.json`). 보고서 `reports/calib_check_1006_robot2_headless_run2.png`
- **Step 1**: 오른팔 팔꿈치는 이번에 4회 만에 수렴 (표준오차 0.037°). 1차의 미수렴은 재현되지 않음
- **Step 2**: 위치 노이즈 0.224 mm, 방향 노이즈 0.30°(하한). 10-02 이전 로봇의 헤드 없는 실행은 0.435 mm. 결과(리셋 영점 기준) 오른팔 `[+0.12, +2.41, +0.05, +2.22, +0.73, -0.08, -1.58]`, 왼팔 `[+0.27, -1.66, -0.19, +1.91, +1.99, +0.01, +0.55]`°
- **확인 (브라켓 스윕, 팔별 재계산)**: 헤드 없는 배치의 치우침이 이번에도 같고 좌우가 반대 부호다
	- 마커 화면 위치 가로: 오른팔 +13.2~13.3°, 왼팔 −10.5~−10.8°
	- 마커에서 본 J6 축 pitch: 오른팔 −0.61°, 왼팔 +0.65~+0.69° (같은 브라켓의 헤드 있는 실행 10-01은 0.00 / −0.06~−0.08°)
	- J5–J6 축 사이 각: 오른팔 86.0~86.1°, 왼팔 86.3~86.4° (헤드 있는 실행 88.3~89.9°)
	- 같은 카메라·브라켓이 헤드 있는 배치에서는 0°에 가깝고 헤드 없는 배치에서는 로봇 두 대 모두 ±0.5~0.7°이므로, 브라켓이 아니라 보는 위치에 따른 측정 치우침으로 판단 (마커가 화면의 어느 쪽에 있느냐에 따라 부호가 바뀜)
- **확인 (이번에 결과가 좋았던 이유)**: 좌우 치우침이 반대 부호로 비슷해서 앱의 브라켓 좌우 평균이 적용됐고 서로 상쇄됐다. 최종 브라켓 pitch는 양팔 +0.05°, z −48.79 mm. 팔별 재계산값의 평균 (−0.61 + 0.67) / 2 = +0.03°와 맞는다. 10-02에는 앱의 값이 오른 −0.52° / 왼 +2.23°로 좌우 차 2.75°가 한계 1.5°를 넘어 평균이 적용되지 않았고, 치우친 값이 그대로 Step 2에 들어갔다
- **추정**: 헤드 없는 모드의 결과 품질은 지금 좌우 평균이 적용되느냐에 달려 있다. 10-02에 왼팔만 2.23°까지 간 이유는 가르지 못했다 (팔별 재계산으로는 그날 왼팔도 +0.52~0.54°)
- **검증 경계**: 이번 로봇의 헤드 있는 실행이 없어 이 로봇에서의 직접 비교는 없다. J6 값(오른 +1.6°, 왼 −0.55°)이 맞는지는 판정할 수 없다. 최종 판단은 확인 자세 관찰이다
- **사용자 관찰 (적용 뒤)**: 확인 자세가 잘 맞았다 (🟢 사용자 실기 관찰, 이 로봇·이 실행에 한함)
- **1차와 2차의 차이 (사용자)**: 바꾼 것은 밝기뿐이고, 2차는 1차(수동 노출 4000)보다 조금 더 밝게 했다. 2차의 노출값은 어디에도 기록되지 않았다 (설정에 저장하지 않는 값이고 결과 파일에도 없음)
- **밝기 효과를 데이터로 확인하려 한 결과**: 저장된 스윕(반복 1회차만 남음)에서는 1차와 2차의 차이가 보이지 않는다. 팔꿈치 스윕 원 맞춤 잔차 0.19 / 0.21 mm, 프레임 간 흔들림 중앙값 0.21~0.25 / 0.21 mm, 프레임 수 250 / 250, 뒤집힘으로 걸러진 프레임 스윕당 3.4 / 4.8개. 손목·브라켓 스윕도 같은 수준
- **팔꿈치 추정값의 흩어짐**: 1차 7회 표준편차 0.146°, 2차 3회 0.069°. 2차는 처음 두 개(잔차 0.3° 이내)가 0.074° 차로 가까워 2개 만에 통과했다. 3회로는 흩어짐이 줄었다고 말할 수 없다 → 밝기가 원인인지 우연인지 가르지 못함
- **남은 일 (사용자)**: 2차 노출값. 노출값을 결과·디버그 기록에 남길지, 반복마다 스윕 원본을 남길지. 헤드 없는 준비 자세에서 마커가 화면 중심에 더 가깝게 오도록 바꿀지는 미정

---

## 2026-10-06 — 다른 로봇(v1.2) 헤드 없는 모드 Full Auto: 오른팔 팔꿈치 미수렴으로 중단 (21:24~21:30)
> 코드 변경 없음 · 검증 수준 🟠 (로그·스윕 파일 오프라인 대조) · 실기 추가 확인 없음
- **실행 (사용자)**: 다른 로봇, 헤드 없는 모드, 수동 노출 4000. 21:24 Home Offset Reset → Step 1 오른팔 패스 1: J5 수렴(+0.0565°, 2회) → 브라켓 스윕 → J6 +1.6475° → 팔꿈치(J3) 8회 반복 뒤 미수렴으로 Full Auto 중단. 왼팔과 Step 2는 실행되지 않음
- **백업**: `result_backup/20261006_robot2_headless_step1_fail/` (오늘 만들어진 오른팔 파일 9개, `setting.yaml`, `home_reset_baseline.json`). `result/result_txt`의 왼팔 파일은 10-02 이전 로봇 기록이 남은 것이라 복사하지 않음
- **확인 (팔꿈치 로그)**: 수렴 조건은 목표 근처 추정값 평균의 표준오차 ≤ 0.06°인데 8회 뒤 0.0622°였다 (추정값 6개, 표준편차 0.152°). 이전 실행(09-29, 10-01, 10-02)의 오른팔 팔꿈치는 3~4회에 수렴. 이번에는 거의 같은 적용 오프셋(−2.1294° / −2.1295°)에서 잔차가 −0.214°와 +0.020°로 달랐다 → 반복 계산이 발산한 것이 아니라 스윕마다 측정값이 다르게 나온 것
- **확인 (스윕 파일, 첫 반복)**: 팔꿈치 스윕의 프레임 수 250, 원 맞춤 잔차 0.19 mm, 마커의 화면 내 위치와 거리(367 mm)는 10-02 헤드 없는 실행과 같은 수준 → 한 번의 스윕 품질로는 차이가 보이지 않는다
- **원인**: 미확정. 후보: 이 로봇의 팔(J2~J4) 유격·반복성, 브라켓·마커 판 체결, 스윕 끝에서 매번 걸러지는 IPPE 뒤집힘 프레임(반복마다 1~5개). 반복 2회차 이후의 스윕 원본은 파일로 남지 않아 더 가르지 못함
- **부수 발견 (보고서 `reports/calib_check_1006_robot2_headless.png`)**: 헤드 없는 모드의 브라켓 스윕 숫자가 두 로봇에서 서로 같고, 헤드 있는 실행과는 다르다
	- 마커 화면 위치 가로 +13.3° (10-02 헤드 없음 +12.6~12.8°, 헤드 있음 +3.8~6.0°)
	- J5–J6 축 사이 각 86.1° (10-02 헤드 없음 86.8~87.3°, 헤드 있음 89.1~89.9°)
	- J6 축 pitch −0.69° (10-02 헤드 없음 −0.46°, 헤드 있음 0.00~0.07°), J4 원 반지름 181.6 mm (10-02 헤드 없음 182.3 mm, 헤드 있음 183.5~183.6 mm)
	- J6 두 방식의 차 +0.88° (10-02 헤드 없음 +0.44~0.56°, 헤드 있음 −0.38~−0.83°)
	- 로봇이 달라도 같게 나오므로 로봇·브라켓 개체가 아니라 헤드 없는 배치(마커가 화면 중심에서 13° 벗어남) 쪽에서 오는 치우침으로 보인다 (추정: 같은 카메라·브라켓을 옮겨 달았는지는 사용자 확인 필요)
- **검증 경계**: 오른팔 패스 1 하나뿐이다. 이 보고서의 "기록된 브라켓" 칸은 10-02 값이 남은 `setting.yaml`을 읽은 것이라 이번 로봇의 값이 아니다
- **남은 일 (사용자 결정)**: 다시 실행해 재현 여부 확인, 로봇·브라켓 체결 점검, 반복마다 스윕 원본을 남기도록 할지

---

## 2026-10-06 — 보정 데이터 점검 보고서 (브라켓 스윕·J6 일관성을 한 장으로)
> 커밋 기준 `a21bcb0` + 미커밋 변경 · 검증 수준 🟠 (오프라인: 합성 스윕 테스트 + 저장된 실행 3건과 로그 대조) · 테스트 통과 269 / 실패 0 / 오류 0 / skip 0 · `core/`·앱 코드 변경 없음
- **목표 (사용자 요청)**: 브라켓 데이터의 신뢰성과 J6 값을 사람이 읽고 판단할 수 있게, 분석 숫자를 한 장의 보고서 이미지로 정리한다
- **만든 것**
	- `scratch/calibration_report.py`: 실행 폴더(`result/result_txt` 또는 그 복사본)의 `sweep_points_<팔>_marker_axis_<4|5|6>.txt`와 `joint_calib_debug_*.txt`만 읽는다. 로봇·카메라 접속 없음. 출력은 `reports/calib_check_<이름>.png`와 같은 숫자의 `.json`
	- 원 피팅은 앱의 `BaseCalibrator.fit_circle_3d_and_6dof_misalignment`, 축을 마커 기준으로 옮기는 것도 앱의 `MarkerCalibrator.axis_in_marker_frame_robust`를 그대로 쓴다
	- 팔·패스별 항목: 세 스윕의 원 반지름과 잔차, 마커에서 본 J6 축의 기울기(pitch, roll), 손목 축 사이 각, 스윕 중 마커의 화면 내 위치, J5 보정 뒤 원 크기 차, J6 두 방식 값과 차
	- 표시 규칙: 빨강은 앱이 이미 쓰는 경고 기준만 (J6 두 방식 차 0.5°, J6 원 반지름과 브라켓 y 설계값 차 1.0 mm, 기록된 브라켓 좌우 차 2.0 mm / 1.5°). 노랑은 이전 실행 범위에서 벗어난 정도(범위 폭의 몇 배)를 진하기로 나타낸다. 새 합격 기준은 만들지 않았다
	- 비교 대상은 스크립트의 `DEFAULT_HISTORY` 세 묶음 (09-29 12:08, 09-29 15:00, 10-01 18:17), `--history`로 바꿀 수 있다
- **하지 않은 것과 이유**: 앱의 브라켓 값(x·y·z·pitch) 자체는 다시 계산하지 않는다. 계산에 J5·J4 스윕 때의 J6 각이 필요한데 스윕 파일에 전체 관절값이 없다. 팔별 원래 브라켓 값도 파일로 남지 않는다 (좌우 평균 뒤 값만 `setting.yaml`에 저장). 그래서 보고서는 그 계산의 입력(반지름, 축 방향)을 보여 준다
- **검증**
	- `tests/test_calibration_report.py` 10개: 반지름·기울기·축 사이 각·마커 위치를 아는 합성 스윕에서 같은 값이 나오는지, 로그 파싱, 표시 규칙, PNG·JSON 생성
	- 저장된 실행 3건(패스 12개)에서 다시 계산한 프레임 수가 앱 로그와 모두 같고, J6·J5 원 반지름은 로그(0.1 mm 단위)와 최대 0.048 mm 차
	- G2 전체 269/0/0/0 (`backups/g2_20261006_calib_report.log`), 실행 중 `config/`·`result/`·`reports/` 변경 없음
- **보고서에서 보이는 것 (저장된 실행)**
	- 10-02 헤드 없는 실행: 마커가 화면 중심에서 가로 +12.6° / −9.6°, 세로 −6~−8° (이전 실행은 가로 ±6° 이내, 세로 −1~−3°). 왼팔 J6 축 roll 92.3~92.8° (이전 90.3~90.6°), pitch 양팔 ±0.5° (이전 ±0.08° 이내). 반지름도 오른팔 J4 −1.3 mm, 왼팔 J5 −0.9 mm
	- 10-01 전도 후 실행: 오른팔 J4–J5 축 사이 각 88.2° (이전 88.8~89.0°), J5 보정 뒤 원 크기 차 1.52 mm (이전 0.05 mm)
	- J6 두 방식의 차는 팔마다 거의 일정한 값을 가진다: 09-29·10-01에 왼팔 +1.0~+1.4°, 오른팔 −0.4~−0.8°. 10-02에는 양팔 모두 +0.3~+0.6°로 옮겨 갔다. 왼팔은 정상 실행에서도 앱의 경고 기준 0.5°를 늘 넘는다
- **검증 경계**: 일관성을 보는 도구이고 J6가 정확한지는 판정하지 못한다 (J6 영점과 브라켓 yaw는 구분되지 않음). 이전 실행이 3회뿐이라 "범위 폭의 몇 배"는 범위가 좁게 잡혀 과장될 수 있다. v1.2 손목 전용 (J6 축을 플랜지 z축으로 읽음). 10-02의 원인이 마커의 화면 내 위치라는 것은 이 보고서로도 여전히 추정이다
- **남은 일 (사용자 결정)**: 보고서 항목·배치 수정 요청, 앱에서 Step 1 뒤 자동으로 만들지 여부, 실행 기록을 자동으로 백업해 비교 대상을 쌓을지 여부
- **미커밋 파일**: `scratch/calibration_report.py`, `tests/test_calibration_report.py` (새 파일), `docs/progress.md`, 출력물 `reports/calib_check_1002_headless.png`·`calib_check_1001_run2.png`·`calib_check_0929_1500.png`와 같은 이름의 `.json`

---

## 2026-10-06 — 노출 변경 때 마커 모니터 창 비우기 + 사용자 결정 반영
> 커밋 기준 `a21bcb0` + 미커밋 변경 · 검증 수준 🟠 (오프라인 테스트) · 테스트 통과 259 / 실패 0 / 오류 0 / skip 0 · 실기 미확인
- **목표**: 앞 항목의 남은 한계 해결 (사용자 지시). 노출을 바꾼 직후 화면에 보이는 마커가 이전 노출값 샘플로 다시 "확인됨"이 되는 것을 막는다
- **원인 → 변경**
	- core 모니터의 3초 창(30샘플)이 노출 변경 때 비워지지 않았다 → `MarkerMonitor.restart_window()`가 창 번호를 올리고, 모니터 루프가 번호가 바뀐 것을 보면 샘플을 버린다. 요약 이벤트에 `window` 번호를 싣는다 (`core/calibration/sequences/marker_monitor.py`)
	- `CalibrationCore.restart_marker_monitor_window()` 추가, `main_ui.reset_marker_monitor_confirmation()`이 이를 부르고 새 창 번호를 패널에 넘긴다
	- 노출을 바꾸는 순간 이미 전달 중이던 요약이 뒤늦게 도착해 다시 확인시키는 경우 → `MarkerMonitorPanel`이 이전 창 번호의 요약으로는 확인 처리를 하지 않는다 (`ui/marker_monitor_panel.py`)
- **테스트**: `tests/test_wizard_slides.py`에 3개 추가 (모니터가 샘플을 버리는지, 패널이 이전 창 요약을 무시하는지, 노출 적용이 창을 다시 시작하는지)
- **검증**: G2 전체 259/0/0/0 (`test-runner`, 로그 `backups/g2_20261006_monitor_window.log`), 실행 중 `config/`·`result/` 변경 없음. 그 뒤 새 테스트에서 타이밍에 기댈 수 있는 중복 검사 한 줄을 지우고 `tests.test_wizard_slides`만 다시 실행: 37개 통과
- **검증 경계**: 한 줄을 지운 뒤에는 전체 스위트를 다시 돌리지 않았다 (해당 파일만). 실제 카메라에서 노출을 바꿨을 때의 동작, 노출 적용 직후 카메라가 내놓는 한두 프레임이 새 창에 섞이는지는 확인하지 않았다
- **사용자 결정 (2026-10-06)**
	- J0 공통모드 댐핑(2000)은 유지한다
	- `rby1_sdk`·`core.robot`을 여러 곳에서 직접 import하는 것은 허용한다 → `.claude/rules/architecture.md`의 알려진 위반에서 빼고 예외로 적음. 남은 위반은 `marker_detection.py` → `calibration_optimizer` (시뮬레이터용 `SimulationModel`의 `bracket_transform`·`marker_pose`에서만 사용), 처리 방법은 미정
	- 손목 다양화 자세는 보류 (확인할 로봇 없음)
	- v1.2 시뮬 앱 재초기화는 해결됐던 것으로 보고, 다시 생기기 전까지 무시
	- `setting.yaml` 브라켓 작업값은 다음 보정에서 다시 정해지므로 상관없음. 버전 전환 때 공칭값으로 돌아가는 것은 의도된 동작
	- 옵시디언 볼트는 이 세션에서 더 수정하지 않는다 (사용자가 나중에 한꺼번에 반영)
- **문서에 아직 없는 것**: 위 결정들과 이 수정은 노션·볼트에 반영하지 않았다 (요청 없음). 볼트·노션의 process 4.7 "결정 대기"에는 J0 댐핑, 구조 위반, 모니터 창 항목이 그대로 남아 있다
- **미커밋 파일**: `core/calibration/sequences/marker_monitor.py`, `core/calibration/calibration_core.py`, `main_ui.py`, `ui/marker_monitor_panel.py`, `tests/test_wizard_slides.py`, `.claude/rules/architecture.md`, `docs/progress.md`, 로그 `backups/g2_20261006_monitor_window.log`

---

## 2026-10-06 — G2 실행: 256개 중 2개 실패 (마커 모니터 "확인됨"이 노출 변경 뒤에도 남음)
> 커밋 기준 `a21bcb0` + 미커밋 변경 · 검증 수준 🟠 (오프라인 테스트 1회, `test-runner` 에이전트) · 테스트 통과 254 / 실패 2 / 오류 0 / skip 0 · 코드 수정 없음
- **배경**: 10-01 UI 변경(노출 슬라이드, 마커 모니터 패널, 그리퍼 전원 끄기)과 그때 작성만 한 테스트가 한 번도 실행되지 않았다. 직전 G2는 10-01의 241/0/0/0 (이 변경 전)
- **실패 2건** (둘 다 `tests/test_wizard_slides.py`)
	- `TestMarkerMonitorPanel.test_every_exposure_change_clears_the_confirmation` (482행): `apply_camera_exposure` 뒤에도 패널의 `confirmed`가 True
	- `TestWizardSlides.test_marker_monitor_runs_only_on_the_exposure_slide` (289행): 오른쪽 마커가 흔들리는데 전체 판정 라벨이 초록(양쪽 확인됨)
- **원인 (코드로 확인)**: `ui/marker_monitor_panel.py`의 `reset_confirmation()`이 `confirmed`를 지운 직후 `update_monitor(self.last_info)`를 다시 부른다. `last_info`가 "양쪽 안정" 상태이면 76~77행에서 곧바로 다시 확인됨이 된다. 즉 모니터가 안정 상태를 보고 있는 중에 노출을 바꾸면 이전 노출값의 확인이 지워지지 않는다
- **추정 (미확인)**: 두 번째 실패는 같은 원인으로 앞 테스트의 확인 상태가 `setUp`의 리셋을 넘어온 것. 고친 뒤 다시 돌려야 확인된다
- **검증 경계**: 1회 실행이다. 실기·실제 화면은 여전히 미확인. 나머지 254개 통과가 10-01 변경의 실기 동작을 뜻하지는 않는다
- **증거**: `backups/g2_20261006.log` (실패 traceback 683~706행). 실행 전후 `config/`, `result/` 변경 없음 (수정 시각으로 확인)
- **수정 (사용자 지시 "고쳐야 할 듯")**: `reset_confirmation()`이 마지막 정보를 다시 적용하지 않고 버리도록 한 줄 변경 (`update_monitor(None)`). 재실행 G2: 통과 256 / 실패 0 / 오류 0 / skip 0 (🟠, `backups/g2_20261006_after_fix.log`). 두 번째 실패도 사라져 같은 원인이라는 추정과 맞는다
- **남은 한계 (미수정, 사용자 결정 필요)**: core 모니터의 3초 창(30샘플)은 노출이 바뀌어도 비워지지 않는다. 화면에 보이는 마커는 변경 직후 이전 노출값 샘플이 섞인 요약으로 다시 확인될 수 있다. 다른 팔(화면 밖) 마커의 확인은 이번 수정으로 제대로 지워진다
- **하네스 실동작 확인 (이 세션)**: `test-runner` 에이전트 호출됨, skills 5개 목록에 표시됨, `ui/`·`tests/` 파일을 열 때 `rules/ui.md`·`measurement-evidence.md`가 로드됨, `loop_guard` 차단 동작, .py 수정 시 `g1_after_edit`의 "G1 통과" 출력. 미확인: `robot_guard`의 질문, `comment_check` 출력
- **사용자 답 (2026-10-06)**: 로봇은 이전 보정값으로 롤백해 둔 상태. 왜곡 계수 검증은 하지 않음, 외부 기준은 도입하지 않음. Home Offset J3 잔차 문제(09-15)는 이미 처리됨. `setting.yaml` 브라켓 v1.3 값은 로봇 버전을 오가며 쓰기 때문이고 버전 전환 때 알아서 바뀌면 됨. J0 공통모드 댐핑은 커플링을 막아 준 것으로 기억 (존폐 확정 전)
- **확인**: 저장된 Step 2 데이터셋은 10-01 네 번 모두 64자세, 10-02는 54자세(헤드 없음)이고 `step2.wrist_diversity_poses`는 false → 손목 다양화 자세로 수집한 실기 기록은 없다
- **문서 반영**: 볼트(정본)에 먼저 적고 노션에 같은 내용을 옮김 — process에 10-01~10-06 기록 6건과 4.7 현황·결정, cps에 C18·C19, report 상단 목록, harness 저장소 적용 내용. 다음 문서 정리는 이 항목 뒤부터

---

## 2026-10-06 — 새 공통 하네스 적용 (CLAUDE.md 틀, rules·skills·agents, 권한과 hook 7개)
> 커밋 기준 `a21bcb0` · 검증 수준 🟠 (hook에 JSON 입력을 넣은 판정 32/32, 실제 세션에서는 `loop_guard` 차단 1건만 확인) · 테스트 스위트 미실행 (프로덕션 코드 변경 없음)
- **목표**: 노션 `AI 코딩 하네스 & 가이드북`(2026-10-05 개정, 하위 6개 페이지)을 `harness_overview`의 적용 순서대로 이 저장소에 넣는다 (사용자 요청)
- **변경**
	- `CLAUDE.md`: 가이드북 틀(행동 지침 4개 + 절대 규칙)로 교체. 기존의 🔴 규칙·안티패턴·래칫 기록·graft 안내는 "이 프로젝트의 규칙"으로 옮김. 명령 표는 가이드북 방침대로 빼고 harness 문서를 가리킴. 계층 대응 표는 `rules/architecture.md`로 옮김. 143줄
	- `.claude/rules/` 7개, `.claude/skills/` 6개, `.claude/agents/` 3개: 가이드북 원문 그대로 (코드 블록을 기계적으로 추출). 고친 곳은 `architecture.md`·`comments.md`의 `main.py` → `main_ui.py`와 `architecture.md` 끝의 이 프로젝트 대응 표뿐
	- `.claude/hooks/` 7개: 원문에서 프로젝트 값만 바꿈 — `robot_guard`(`main_ui.py`·`run_windows`·`run_linux`는 항상 묻기, 시뮬레이터 표시는 비움), `protect_originals`(`result_backup/`, `result_d435/`, `result (…)`), `g1_after_edit`(수정한 .py를 `py_compile`), `loop_guard`(PowerShell 조회 명령 3개를 반복 허용에 추가)
	- `.claude/settings.json`: 기존 allow·deny·git 이력 ask는 유지, 앱·시뮬·테스트 ask는 삭제(테스트와 시뮬은 허용, 실기 실행은 `robot_guard`가 판정), hook 7개 연결. 가이드북 원문과 다른 점: Bash뿐 아니라 PowerShell 도구에도 연결, hook을 `.venv` 파이썬에 `-X utf8`로 실행 (PATH의 python은 3.14, 없으면 차단 메시지 한글이 깨짐)
	- `.claude/hooks/ask_tests.py` 삭제, `.gitignore`에 `.claude/state/` 추가
- **가이드북·노션 문서와 다른 점**: 노션 프로젝트 harness 1장 표의 `tests/run_connected_calibration.py --confirm-local-simulator`는 이 저장소에 없다 (파일도 옵션도 없음). 그래서 시뮬레이터 표시(`SIM_MARKERS`)를 비워 두었다
- **검증**: hook 7개에 Bash·PowerShell·Edit 모양의 JSON을 넣어 차단·질문·통과·루프 한도 판정 32건 확인. 적용 뒤 이 세션에서 같은 명령을 두 번 실행해 `loop_guard`가 실제로 막는 것 확인. `settings.json` JSON 검증, `CLAUDE.md` 6~91행이 가이드북 틀과 한 줄(harness 위치)만 다른 것 확인
- **검증 경계**: `robot_guard`·`block_patch`의 질문(ask) 판정과 `g1_after_edit`·`comment_check`의 PostToolUse 출력이 실제 세션에서 의도대로 동작하는지는 확인하지 않았다. rules의 `paths` 로드, skills·agents 인식도 새 세션에서 확인 필요
- **남은 일**
	- [ ] 새 세션에서 확인: `main_ui.py` 실행 요청 시 묻는지, .py 수정 시 G1 결과가 나오는지, skills·agents가 목록에 보이는지
	- [x] 노션 프로젝트 harness 갱신 (2026-10-06, 사용자 요청): 1장 반영 완료 문구, `run_connected_calibration` 정정 콜아웃, G1 hook 이름, 6장 저장소 적용 목록, 7장 래칫 행. report 상단의 "하네스 양식으로 바꿔야 한다" 항목도 완료로 표시
	- [x] 볼트(정본)의 프로젝트 `harness`·`report`에도 같은 수정 반영 (2026-10-06, 사용자: 옵시디언이 기준이고 노션이 거기에 맞춘다)
	- [x] 전역 `~/.claude/CLAUDE.md`를 행동 지침 4개 + 절대 규칙으로 교체 (2026-10-06, 사용자 요청). 이전 내용은 `~/.claude/CLAUDE.md.bak_20261006`. 하네스를 아직 적용하지 않은 다른 저장소에서는 옛 전역 요약의 세부 규칙(권한 표, 검증 표기 등)이 더는 로드되지 않는다
- **미커밋 파일**: `CLAUDE.md`, `.gitignore`, `docs/progress.md` (수정), `.claude/hooks/ask_tests.py` (삭제), `.claude/rules/`·`skills/`·`agents/`·`hooks/` (새 파일 23개). `.claude/settings.json`은 `.gitignore`의 `*.json`으로 git 밖이고, 이전 내용은 `backups/claude_settings_20261006_pre_harness.json`에 있다

---

## 2026-10-02 — 헤드 없는 모드 실행(11:49, 미적용) 결과가 나쁜 이유: 오늘 Step 1의 브라켓 pitch (사용자 요청: J6=0이면 어땠는지)
> 코드 변경 없음 · 검증 수준 🟠 (운영 `optimize_step2`로 오프라인 재계산, 로봇은 모델 읽기 전용 접속) · 실기 미확인
- **실행**: 11:35 리셋 → Step 1 → Step 2 (헤드 없는 모드, 54자세, `q_head` 없음). Step 1.5 없음. 결과 `result_20261002_114936_921873.json`, 위치 잔차 0.435 mm (헤드 있는 실행은 0.15 mm)
- **Step 1 값**: J6(store) 오른 +0.356°, 왼 +0.934°. 브라켓 pitch 오른 −0.519°, 왼 +2.227° (전날 0.056°), z −48.85 / −49.49 mm. 좌우 차가 커서 대칭 평균 미적용
- **결과 − 리셋 기준(= 10-01 19:5x 적용 영점)**: 오른 `[-0.89, -0.74, +1.30, -0.03, -0.13, -0.03, -0.35]`, 왼 `[-1.46, +2.12, -2.37, +0.03, +1.69, +0.02, -0.93]`°
- **재계산 (`scratch/step2_rerun_headless_20261002.py`, 재현 오차 0.000°)** — 값은 결과 − 기준
	- A. J6=0만: 오른 `[-0.87, -1.22, +1.26, -0.03, -0.82, -0.03, +0.01]`, 왼 `[-1.75, +1.68, -2.19, +0.02, +0.85, +0.01, 0.00]`, 잔차 0.431 mm → 거의 그대로 나쁨
	- B. J6=0 + 브라켓 pitch 0 (z는 오늘 값): 오른 `[+0.09, -0.28, +0.44, -0.03, -0.19, -0.03, +0.01]`, 왼 `[+0.09, -0.49, -0.37, +0.02, -0.22, +0.01, -0.02]`, 잔차 0.216 mm
	- C. J6=0 + 전날(10-01 19:49) 브라켓: 최대 0.50°, 잔차 0.239 mm
	- 브라켓 pitch만 0 (J6은 Step 1 값): 잔차 0.198 mm, 왼 J4 +0.51 / J6 −0.94 (J4·J6 맞바꿈)
	- 카메라 회전 잠금 해제(`lambda_cam_rot` 100, 1): 잔차 0.426 mm로 변화 없음 → 카메라 쪽 원인 아님
- **확인**: 나빠진 주원인은 오늘 Step 1이 낸 브라켓 pitch(특히 왼쪽 2.2°). J6은 그 짝(같은 J6 축 둘레 회전)이라 J6만 0으로 두면 해결되지 않음
- **추정 (미확인)**: 헤드 없는 모드는 헤드가 마커를 향하지 않아 마커가 화면 중심에서 30~35 mm 옆·20 mm 위로 치우친 채 스윕됨(전날 대비). 방향 측정 편향이 커져 pitch·J6이 크게 나왔을 가능성. 브라켓을 다시 달았는지는 사용자 확인 필요. 스윕 각도 범위는 모두 채워졌고 왼팔 2패스만 프레임이 적음(219/170/221)
- **산출물**: `result_backup/20261002_headless/` — 원본 결과·데이터셋·스윕·기준, `whatif_A_j6_zero.json`, `whatif_B_j6_zero_bracket_pitch_zero.json`, `whatif_C_j6_zero_brackets_1001.json` (미리보기용, `result/`에는 넣지 않음)
- **남은 일 (사용자 결정)**: 어느 what-if를 체크 포지션으로 미리볼지, 미리보기 방법(result_step2에 넣고 앱 재시작 / 파일 선택 기능 추가)

---

## 2026-10-01 — 로봇 전도(오른쪽) 후 재보정 17:43 결과 진단 (적용됨, 사용자: J0은 맞고 다른 축·J5가 어긋나 보임)
> 코드 변경 없음 · 검증 수준 🟠 (로그·결과 파일 오프라인 대조) · 실기 추가 확인 없음
- **배경**: 11:43 결과 적용 → 로봇이 오른쪽으로 넘어짐 → 17:26 Home Offset Reset → 17:43 Step 2 결과(`result_20261001_174254_601637.json`) 적용. 적용 후 `home_reset_baseline.json`은 0으로 바뀜 (리셋 당시 값은 GUI 로그에만 있음: R `[0.681, -1.272, 0.001, 2.658, 0.004, 0.698, 0.000]`, L `[0.079, -1.018, -0.003, 1.540, 0.001, 0.505, 0.018]`)
- **확인 1 (같은 리셋 안, 결과 − 기준 = 오전 보정 영점 대비 변화)**: 오른팔 J1 +1.09°, J5 +1.51°, J3 −0.44°, J4 +0.43°, J2 +0.19°, J6 −0.15°, J0 −0.09°. 왼팔 J1 0.00°, J5 −0.01°, J6 0.00°, J0 +0.14°, J4 −0.13°, J3 +0.44°. 왼팔은 J3만 빼고 재현, 오른팔만 크게 바뀜
- **확인 2 (Step 1 오른팔 J5, `joint_calib_debug_right_wrist_pitch.txt`)**: 수렴 뒤에도 J4 원과 J6 원이 겹치지 않음 — 반지름 차 1.50~1.56 mm, 중심 거리 1.59~1.61 mm (3회 반복 동일). 09-29 15:00 같은 팔: 0.05 mm / 0.04~0.12 mm. 오늘 왼팔: 0.09~0.14 mm / 0.14~0.32 mm. 즉 오른팔은 J5 오프셋만으로 J4·J6 축이 일직선이 되지 않는다
- **확인 3 (`scratchpad/wrist_axes_check2.py`, 도구의 `fit_circle_3d_and_6dof_misalignment`로 마커 자세 스윕 재계산)**: J5=90° 자세에서 J4·J5·J6 축 사이 간격은 09-29와 같은 수준(오른팔 0.70 / 0.33 / 1.17 mm, 이전 0.16~1.16 / 0.87~1.24 / 1.17~1.18). 축 사이 각은 오른팔 J4^J5만 88.77° → 88.09°. 손목 중심이 어긋난 흔적은 없고, 어긋남의 위치는 특정하지 못함
- **확인 4**: Step 2 방향 노이즈 0.386° (09-29·오전은 하한 0.30°), 위치 0.156 mm
- **추정**: 넘어지면서 오른팔에 기구적 변화(전완~손목, 어깨 J1). J5 −2.26°의 일부는 J5 영점이 아닌 변형을 대신 흡수 → 관절 오프셋만으로는 모든 자세에서 맞지 않음. 왼팔 J3 +0.44°는 설명하지 못함
- **검증 경계**: 기구 손상을 직접 본 것이 아님. 오전(11:43) Step 1 로그는 덮어써져 넘어지기 직전 상태와 직접 비교는 못 함(09-29와 비교). 보정 수학은 오늘 수정하지 않았고 왼팔 재현으로 보아 코드 회귀로 보이지 않으나, G2는 미실행
- **부수 경고**: 왼쪽 브라켓 z가 ±2 mm 한계(−50.00), J6 두 방식 차 양팔 0.5~1.05°
- **남은 일 (사용자 결정)**: 오른팔 기구 점검, 오른팔 Step 1만 재실행해 재현성 확인
- **2차 실행 (브라켓 재조립 후, 18:12 리셋 → 18:28 `result_20261001_182834_940993.json`, 미적용)**: 원시 기록은 `result_backup/20261001_after_fall_run2/`에 복사. 분석 스크립트 `scratch/wrist_axes_check.py` (도구의 원 피팅을 그대로 사용, 읽기 전용)
	- 재현성: 결과 − 기준(17:43 영점 대비) 오른팔 `[-0.07, +0.11, +0.16, +0.08, -0.23, -0.01, +0.22]°`, 왼팔 최대 0.07°. 브라켓을 다시 달아도 같은 값 → 오른팔은 새 상태로 안정, 브라켓 장착이 원인이 아님
	- 브라켓: 마커에서 본 J6 축(플랜지+브라켓+마커 판에만 의존)이 09-29 대비 오른팔 0.09° / 0.29 mm, 왼팔 0.07° / 0.04 mm (09-29 두 실행 간 차 0.06~0.11° / 0.15 mm). 반지름 r6 54.15 mm 동일 → 브라켓·마커 판이 휜 흔적 없음
	- 손목: J5=0 자세에서 마커~J4 축 52.50 mm, 마커~J6 축 54.20 mm (차 −1.71, 1차 −1.43, 왼팔 −0.09). J5 보정 뒤에도 반지름 차 1.21~1.52 mm 잔류. J5=90° 자세의 J4^J5 각은 88.77/88.77°(09-29) → 88.09/88.27°(전도 후), 왼팔 89.40~89.54° 그대로. J5^J6, 축 간격, r4·r5·r6는 그대로
	- 정합성: J4^J5 변화 0.5~0.7° × 손목 중심~마커 축방향 거리 167 mm = 1.5~2.0 mm ≈ 관측 1.5~1.7 mm
	- 팔꿈치 자세(J2|J4)는 수렴 뒤 0.18 mm로 정상 → 상완~전완은 곧음
	- **오프라인 검증 (`scratch/verify_j4_j5_misalignment.py`, 그림 `result_backup/20261001_after_fall_run2/j4_j5_axis_angles.png`)**: 도구의 피팅과 별개로, 마커 궤적 평면의 법선만으로 축 방향을 다시 계산 (관절 영점과 무관한 양)
		- 같은 조건(D405, 내부 파라미터 파일 09-29 11:35 그대로, 같은 준비 자세)끼리: 오른팔 J4^J5 88.86° (sd 0.12, n=4) → 88.22 / 88.29° = −0.60° (5.1 sd, 09-15·09-21 포함 이전 9회 범위 88.77~89.49 밖). 왼팔 90.42° → 90.31 / 90.62° (+0.04°)
		- 오른팔 J5^J6도 −0.60°이지만 이전 흩어짐이 0.48°라 1.3 sd (구분 불가). r5는 그대로(175.4 → 175.7 mm)
		- 방향·크기 정합: 넘어지기 전 데이터에 "J5 축부터 바깥쪽을 J4 축 대비 0.60° 기울임"만 넣어 J5=0 자세를 다시 계산하면 r4−r6가 −1.48~−1.59 mm 변함(J5 영점은 0.00° 변함). Step 1 실측은 0.05 → −1.2~−1.7 mm (r4 52.5 < r6 54.2). 부호와 크기 일치
		- 실패한 교차 검증(기록): 마커 자세 스윕만으로 J5=0 자세의 절대 반지름 차를 예측하면 모든 실행·양팔에서 −3~−5 mm로 나옴 → 축 방향에 공통 1~2° 편향이 있어 절대값은 못 쓰고 같은 조건 간 변화만 유효. 마커 회전(방향) 기반 축 추정은 흩어짐 0.5° 이상이라 판별력 없음
	- 추정 위치: J4 축과 J5 축 사이(손목 피치 관절이 달린 하우징·체결부)가 J5 축 방향으로 약 0.55° 틀어짐 + J5 영점 1.5° 이동. J5~J6 쪽은 측정 잡음(0.4°) 때문에 완전히 배제하지 못함. 어깨 J1(+1.09°)은 Step 1에 J0/J1 스윕이 없어 데이터로 확인 불가

---

## 2026-10-01 — 밝기 조정: 카메라 탭 마커 모니터, 미저장 경고, 위자드 노출 슬라이드 앞당김 + 준비자세·제로 버튼, 그리퍼 전원 끄기
> 커밋 기준 `a21bcb0` · 검증 수준 🟠 (offscreen 구동 확인 + 가짜 객체로 core 경로 확인, G2 미실행, 실기 미검증) · 사용자 결정: 순서 노출 → 제로 → 홈 리셋, 3번 슬라이드는 전원 켠 채 진행(티칭 버튼 사용), 가능하면 툴 플랜지 전원 끄기
- **공용 위젯**: `ui/marker_monitor_panel.py` `MarkerMonitorPanel` 신규 (위자드 노출 슬라이드에 있던 마커 인식·흔들림 표시를 옮김). 위자드와 카메라 탭 > 밝기 조정에 같은 위젯 2개. 모니터 실행 조건은 `main_ui.sync_marker_monitor`로 이동: 위자드 노출 슬라이드 또는 밝기 조정 하위 탭이 보일 때만 core 모니터 실행, 탭 전환·카메라 재연결·로봇 이동 종료 때 재평가
- **한 팔씩 확인**: 준비 자세는 한 팔만 화면에 들어오므로 패널이 마커별로 "이 노출값에서 안정 인식됨"을 기억한다 (`confirmed`). 두 마커 모두 기억되면 성공 표시. 노출이 바뀌면(적용·취소·자동 전환·카메라 재연결) `reset_marker_monitor_confirmation`으로 지운다. 판정 기준(RMS 0.2 mm, `exposure_check.max_marker_jitter_rms_mm`)은 그대로
- **미저장 경고**: `slide_exposure.not_saved_warn` 문구를 위자드 노출 슬라이드와 카메라 탭 밝기 상자에 표시. 근거: 적용 버튼은 메모리(`saved_camera_*`)에만 두고 setting.yaml을 쓰지 않으며, 시작·재연결 때 `init_camera_exposure_state`가 setting.yaml(`auto_exposure: true`)로 되돌린다
- **위자드 순서**: 3 그리퍼·브라켓 → 3-1 노출·마커 인식 → 3-2 제로 포즈 → 3-3 홈 오프셋 리셋 → 4 → 5 (`SLIDE_EXPOSURE` 4, `SLIDE_ZERO_POSE` 5, `SLIDE_HOME_OFFSET` 6, 9장 유지)
- **노출 슬라이드 버튼**: `오른팔 → J5 준비 자세`, `왼팔 → J5 준비 자세`, 그 아래 `제로 포즈로 이동` + `모션 정지`. 준비 자세는 `MoveToReadyWorker(…, mode, head_zero=True)` (v1.2 `wrist_pitch`, v1.3 `wrist_pitch_v13`), 제로는 `Step2ZeroPoseCheckWorker`. 이 제로 버튼은 3-2 슬라이드 완료로 치지 않는다
- **확인된 사실 → 추가 변경**: `perform_move_to_ready_pose`는 반대쪽 팔과 몸통만 영점으로 보내고 헤드는 그대로 둔다 (`head=None`). 노출 슬라이드가 제로 포즈보다 앞이라 헤드가 임의 위치일 수 있어, core "ready" 시퀀스에 `head_zero` 옵션 추가(헤드 사용 시 헤드를 먼저 0으로, 3 s). 기본값 False라 Step 1 동작은 그대로
- **그리퍼 전원**: 3번 슬라이드에 `그리퍼 전원 끄기` 버튼. `RobotOperations.power_off_tool_flanges` (`set_tool_flange_output_voltage(side, 0)` 양쪽 → `state.tool_flange_*.output_voltage`가 0이 될 때까지 최대 2 s 확인, 거부·잔류 시 오류) → `CalibrationCore.power_off_tool_flanges` (로봇 없음·시퀀스 중 거부) → `main_ui.power_off_tool_flanges`. 다시 켜는 기능은 넣지 않았다
- **구동 확인**: offscreen 창 생성, 제목 번호 1-1/1-2/2/3/3-1/3-2/3-3/4/5, 한·영 문구, 카메라 탭 패널 표시와 실행 조건, 로봇 없을 때 버튼 3개가 실패 문구로 끝남. 노출 슬라이드 최소 높이 526 px (기존 최대 532 px 이하). 가짜 객체로 `head_zero` 순서(head → ready)와 전원 끄기 반환값 확인
- **화면 확인 후 수정 (사용자 스크린샷)**: 노출 슬라이드의 확인 체크박스 글자가 위아래로 잘림. 원인: 밝기 상자 안의 여러 줄 문구(미저장 경고 2줄 등)를 상자 최소 높이가 한 줄로만 계산해 체크박스가 눌림 (실제 Windows 글꼴로 화면 밖 렌더링해 재현: 체크박스 28 px / 필요 42 px). 변경: 미저장 문구를 "이 노출값은 저장되지 않습니다." 한 줄로 줄여 자동 노출 체크박스 옆으로 이동, "체크해야 다음으로 진행" 상태 문구(`status_waiting`) 삭제(상태 줄은 적용·취소·확인 때만 표시), 상단 안내문과 이동 주의 문구를 한 줄 길이로 축약, 체크박스 padding 10 → 6 px, 밝기 상자 제목을 `밝기 조정`으로(슬라이드 제목의 `&`가 단축키로 먹혀 사라지던 것). 재렌더링(1400×874, 한·영): 체크박스 42 px / 필요 42 px
- **테스트 (작성만, 미실행)**: `tests/test_wizard_slides.py` — 순서·번호, 패널로 옮긴 모니터, 카메라 탭 모니터, 미저장 문구, 버튼 동작, 버전별 준비 자세, 전원 버튼, `TestMarkerMonitorPanel`, `TestToolFlangePowerOff`, `TestReadyPoseHeadZero`
- **검증 안 된 것**: 실기에서 J5 준비 자세일 때 마커가 화면에 들어오는지와 헤드 0 이동, 실제 로봇의 `set_tool_flange_output_voltage` 응답과 전압 read-back 지연, 시뮬(모델 m)의 툴 플랜지 지원 여부, 실제 화면 배치
- **미커밋 파일**: `ui/marker_monitor_panel.py`(신규), `ui/wizard_widget.py`, `ui/core_bridge.py`, `main_ui.py`, `core/calibration/calibration_core.py`, `core/robot/robot_core.py`, `config/ui_config/i18n.yaml`, `tests/test_wizard_slides.py`, `docs/progress.md`

---

## 2026-09-30 — UI 개편: 상단 탭 위자드/카메라/고급, 카메라 시리얼별 내부 파라미터, 위자드 슬라이드 정리
> 코드 변경 · 검증 수준 🟠 (offscreen 구동 확인, G2 미실행) · 사용자 결정: 시리얼 파일은 저장 때 항상 덮어쓰기, 브라켓 사진은 슬라이드에서 v1.2/v1.3 선택(로봇 연결 시 자동), 카메라 두 하위 탭 모두 실시간 화면
- **규칙 개정 (사용자)**: 로봇별 상수·로봇 시리얼은 계속 yaml 금지, 카메라 시리얼별 내부 파라미터 파일은 허용 → CLAUDE.md, 노션 project_harness 요약 반영
- **상단 탭**: `left_tabs` = 위자드 / 카메라(하위: 내부 파라미터 보정, 밝기 조정 — 각자 실시간 화면, 위에 카메라 정보 줄 + 재연결 버튼) / 고급(하위: Step 1 = 기존 Main, Step 2). `TAB_*`, `CAMERA_SUB_*`, `ADV_SUB_*` 상수와 `is_camera_tab_active`·`is_intrinsics_subtab_active`·`is_step2_view_active`로 판별 (숫자 인덱스 6곳 제거). 공유 패널 이동은 고급 하위 탭 전환에 연결. C 키 캡처와 가이드 오버레이는 내부 파라미터 하위 탭에서만. 상단 탭 QSS `QTabWidget#mainTabs` (17px, 선택 시 파랑 배경)
- **시리얼 내부 파라미터**: `storage.camera_intrinsics_serial_path` → `config/camera_intrinsics/<S/N>.yaml`, `save_camera_intrinsics`(작업 파일 + 모델 파일 + 시리얼 파일, 시리얼 파일은 항상 덮어씀). `Marker_Transform._use_serial_intrinsics`: 시리얼 파일(또는 이 S/N이 찍힌 작업 파일)이 있으면 우선 사용·작업 파일에 복사, 다른 S/N의 작업 파일은 그 S/N 파일로 보관. `camera_serial`, `intrinsics_serial_matched`, `intrinsics_source`(serial/model/factory/file)를 `get_camera_info`로 노출. Save 버튼은 `save_camera_intrinsics` 사용
- **위자드**: 1-4(내부 파라미터 확인) 삭제, 1-2 그리퍼 분리 + 1-3 마커 브라켓 → 1-2 한 슬라이드, `SLIDE_COUNT` 11 → 9. 1-1 카메라 정보에 내부 파라미터 출처 줄 + S/N 불일치 시 빨간 주의 문구 + `내부 파라미터 보정` 버튼(보정 슬라이드로, 다음=1-2, 이전=1-1). 1-2에 v1.2/v1.3 선택(로봇 연결 시 그 버전), 사진 `img/marker_bracket_assemble_1_2.png`/`_1_3.png` (원본 복사, 사용자가 나중에 덮어씀)
- **i18n**: `main_tabs.tab_camera/tab_advanced/camera_intrinsics/camera_exposure/camera_info(_none)`, `slide_0.cam_source*/cam_warn_serial/btn_intrinsics`, `slide_marker_bracket.version_label`, 제목 1-2 변경. `slide_2`·`slide_1.title` 삭제
- **테스트**: `tests/test_tabs_and_camera_serial.py` 신규(시리얼 경로·우선순위·보관·저장 덮어쓰기, 탭 구조·판별·공유 패널), `test_wizard_slides.py`(순서, 보정 버튼 이동, S/N 경고, 브라켓 버전, 문구), `test_core_ui_optimizer.py`(탭 상수) 갱신
- **구동 확인**: offscreen으로 창 생성, 탭 3/2/2, 위자드 다음 경로 0→1→3→…→8, 카메라 없음 문구 확인. 설정 파일 변경 없음
- **발견 (확인 필요)**: `config/setting.yaml`이 09-30 18:39:53에 바뀌어 `Tf_to_marker_left/right`가 v1.3 값 `[0.067, 0, 0, 90, 0, -90]`. `_apply_marker_bracket_version`이 v1.3 연결 때 쓰는 동작과 일치 (원인 미확인). 15:13에 쓴 보정 브라켓 값(z -0.04966, pitch -0.0124)은 결과 JSON `ee_to_marker_used`와 `result_backup/20260929_d405/`에만 남음
- **G2 (2026-10-01)**: 241개 통과 241 / 실패 0 / 오류 0 / skip 0. 실행 후 `config/` 파일 변경 없음 확인
- **2026-10-01 수정 (사용자 화면 확인 후)**: 상단 탭이 너무 큼 → 높이 46 → 37 px (padding 8px 16px, min-width 제거, 좌우 여유 = 블록 높이 − 글자 높이). 브라켓 사진은 로봇 연결 전에는 버전을 모름 → 그리퍼 분리·브라켓 슬라이드를 로봇 연결 뒤로 이동, v1.2/v1.3 선택 버튼 제거(연결된 버전 자동). 번호: 1-1 카메라 → 1-2 내부 파라미터 보정(선택, 1-1 버튼으로만) → 2 로봇 연결 → 3 그리퍼·브라켓 → 3-1 → 3-2 → 3-3 → 4 → 5 (제목 3개만 변경, 9장 유지). offscreen 확인: 다음 경로 0→2→3→…→8, 이전 역순, 버전 1.3 사진 전환. G2 미실행
- **검증 안 된 것**: 실제 화면 크기·글꼴, 실제 카메라의 시리얼 매칭. 로봇 전원·서보가 켜진 상태에서 그리퍼 분리·브라켓 체결을 하게 되는 점 (사용자 확인 필요)

---

## 2026-09-30 — Step 2 손목 다양화 자세 추가 (헤드 있는 경우만, J6 고정 유지)
> 코드 변경 · 검증 수준 🟠 (오프라인 선형 해석 + 단위 테스트) · 시뮬(G4)·실기 미검증 · 스크립트 `study_wrist_list.py` (스크래치)
- **사용자 결정**: J6 고정 완화는 고려하지 않음. 손목 다양화 자세 추가 + 테스트 진행
- **자세 목록 (고정 24개)**: J0~J3는 Step 2 기준 자세 그대로, 팔별 J4 ±40 / J5 ±30 / J6 ±45° 안의 변위와 헤드 pan/tilt 변위(±10 / -10~+12.5°). 15:13 기준 자세에서 후보 1,500개 중 가시(화면 여유 60 px, 입사각 ≤ 40°, 거리 ≤ 330 mm, 마커 x 간격 ≥ 11 cm, 3D ≥ 12 cm, 관절 한계 5° 안) 258개 → 정보량 최대 순 24개. 12:23·14:32 기준 자세에서도 24/24 가시
- **오프라인 효과 (운영 고정 J3/J5/J6)**: 가장 약한 특이값 2.4 → 5.0, 확인 자세 손 1σ 최대 0.37 → 0.20 mm, Step 1 J6 1° 치우침의 확인 자세 전파 LJ6 L x+0.99 y-1.08 → x+0.39 y-0.44 mm, RJ6 R x-1.24 y+1.19 → x-0.69 y+0.53 mm
- **변경**
	- `config/ready_poses.yaml` v1.2 `step2_wrist_diversity` (24개, 근거 주석), `config/setting.yaml` `step2.wrist_diversity_poses: false` (시뮬 검증 후 켬)
	- `core/calibration/data.py` `load_step2_wrist_diversity` (스위치 꺼짐 → 빈 목록, 형식 오류 → ValueError, v1.3는 목록 없음)
	- `core/robot/motion.py`: 계획 끝에 선택(optional) 관절 단계로 1회 추가 + 기준 자세 복귀. `offsets_by_arm`(팔별 오프셋, 미러링 없음) 지원. 선택 단계는 관절 한계 5° 안이면 이동 없이 건너뜀(None 반환)
	- `sequences/collection.py`: 헤드 있을 때만 목록 로드, 선택 단계의 한계 건너뜀·마커 미검출은 연속 실패로 세지 않음(로그만)
	- `main_ui.py`: 수동 Auto Motion 계획에도 같은 목록 (`step2_wrist_diversity()`)
	- `scratch/validate_step1_2_sim.py`: `WRIST_DIVERSITY=1`이면 스위치와 무관하게 목록 추가, 선택 단계 건너뜀 처리
	- `tests/test_step2_wrist_diversity.py` 신규 (로더, 계획, 팔별 오프셋·헤드, 한계 건너뜀, 선택 실패 비중단, 일반 실패 중단 유지)
- **G2**: 228개 통과 228 / 실패 0 / 오류 0 / skip 0
- **범위 축소 (사용자: 너무 크게 돌면 충돌·인식 실패 우려, 마커 기울기 30° 이하)**: 입사각 상한 30°로 ±25/20/30, ±20/15/20, ±15/10/15 비교 → 노이즈 쪽(약한 특이값 4.9~5.4, 확인 자세 1σ 0.20~0.21 mm)은 비슷, J6 치우침 전파 감소는 범위에 비례(LJ6 1° → 왼손 x+0.62/+0.80/+0.92, 현재 +0.99 mm). **사용자 선택 ±25/20/30** → `ready_poses.yaml` 목록 교체(최대 변위 R 24.6/19.9/27.8, L 24.7/19.3/29.6°, 헤드 pan ±10 / tilt -10~+12.5°), 테스트 범위 검사 강화. G2 228개 통과 228 / 실패 0 / 오류 0 / skip 0
- **시뮬 검증(G4) 실패 (4회, 새 자세 도달 전)**: v1.2 시뮬(`rby1-sim:0.10.6-m_v1.2`, 모델 m) 앱이 준비자세 이동 중·직후 약 10~16초에 스스로 종료·재실행(`[App] App main`)돼 gRPC 끊김. 오류 로그 없음, 컨테이너 재시작 0, OOM 아님. 클라이언트 없이도 1회 재초기화. `wsl --shutdown` 후 동일. 뷰어 없이 실행은 `Could not initialize GLFW`로 불가. `LIBGL_ALWAYS_SOFTWARE=1`은 대기 90초 안정, 연결 후 동일 실패. 가설(미확인): WSLg 뷰어 문제, 시뮬 SDK 0.9.1 vs 클라이언트 0.10.0 차이. 새 하네스 `scratch/validate_wrist_diversity_sim.py` (연결·전원·준비자세 명령까지만 확인)
- **`start/stop_sim_wsl.bat` 고장 (사용자 발견)**: LF 줄바꿈 + 한글 + `chcp 65001`로 cmd가 `ker` 등 조각을 실행 → 두 파일 CRLF로 변환, `.gitattributes`(`*.bat`, `*.cmd` eol=crlf) 추가, CLAUDE.md 래칫 기록
- **충돌 근사 검사 (오프라인, `collision_check.py` 스크래치)**: 실기 URDF 캡슐 모델 + `detect_collisions_or_nearest_links`. 캡슐이 실제보다 커서(기준 자세에서 양손 캡슐 -30 mm 겹침, 실기에서 돈 64자세 최대 -48 mm) 기존 64자세 대비 상대 비교로 판정. 첫 목록은 손-몸통 최소 89.6 mm로 기존 최소 108.4 mm보다 가까운 자세 7개(해당 팔 J5 -11~-20°) → 조건(양손 ≥ -48.3 mm, 몸통·헤드 ≥ 108.4 mm) 추가해 재선정, 효과 동일(약한 특이값 4.8, 확인 자세 1σ 0.21 mm, LJ6 1° → L x+0.66 y-0.77) → `ready_poses.yaml` 교체. 재검사: 양손 최소 -19.0 mm, 몸통 최소 108.8 mm, 플레이트 간 최소 64.7 mm(기존 62.8) → 기존보다 가까운 자세 없음. 헤드와 가까운 쌍은 없음. 교체 후 G2 228개 통과 228 / 실패 0 / 오류 0 / skip 0
- **발견**: 기존 64자세의 오른쪽 마커 기울기가 D405 실행에서 중앙값 30.8°(최대 40°, 64개 중 40~42개가 30° 초과), 왼쪽 21.5°. D435(09-22)는 오른쪽 25.0°. 사용자 결정: 기존 자세 유지
- **검증 안 된 것**: 팔-몸통·팔-팔 충돌 (마커 간격만 봄), 실제 마커 검출 품질(입사각·거리), 렌즈 왜곡, Step 2 시간 증가(24자세 ≈ +2분 추정)
- **남은 일**: 시뮬 검증(G4, ASK) → 스위치 켜기 → 실기(사용자) → Step 2 로그의 observability 보고로 효과 확인

---

## 2026-09-29 — Step 2 자세 확대 설계 검토 (1번, 헤드 있는 경우만, 오프라인)
> 코드 변경 없음 · 검증 수준 ⚪~🟠 (저장 URDF 선형 해석, 핀홀·왜곡 무시, 충돌 검사 없음) · 스크립트 `study_step2_poses.py` (스크래치, SEED 1)
- **노션 확인**: design 5.1 "J6 ∥ head tilt 1.000 (Step 2 준비자세) → J3/J5/J6 하드 락", design 7 "(09-15) Step 2 자세 폭 확대(J1/J3/J4 ×2, 57샘플) → 개선 미확인·시야 이탈 우려로 원복", S3 "09-17 좌우 대칭 명령 자세 플랜지 차 앞뒤 +1.76 mm"(D435 시절에도 앞뒤 차 존재), harness DO "여러 약한 방향 동시 검증, 끝단 위치로 판정"
- **방법**: 15:13 결과 모델로 후보 자세 생성 → 헤드를 두 마커 중앙에 맞춤 → 가시성(화면 여유 40 px, 입사각 ≤ 실측 최대 40°, 거리 ≤ 450 mm, 마커 x 간격 ≥ 11 cm, 손 거리 ≥ 20 cm) → 정보량 최대 순으로 24개 선택. 노이즈 σ는 거리 비례(위치 d², 회전 d, 기준 실측 중앙 거리)
- **결과 (확인 자세 손 1σ, mm / 가장 약한 특이값)**
	- 현재 64자세: 전부 자유 R x0.90 L x0.62 / 1.9, 운영 고정(J3/J5/J6) R x0.11 L x0.08 (y·z ≤ 0.37) / 2.4
	- 확인 자세 주변 +24 (거리 351~450 mm, 헤드 tilt 15~60°): 전부 자유 ≤ 0.22 / 4.1, 운영 고정 ≤ 0.18 / 5.5
	- 현재 Step 2 영역에서 손목 크게(J4 ±40, J5 ±30, J6 ±45°) +24 (거리 188~380 mm, 헤드 tilt -10~20°): 전부 자유 ≤ 0.25 / 5.7, 운영 고정 ≤ 0.16 / 6.5
- **고정값 치우침 전파 (운영 고정, 고정값 1° 틀림 → 확인 자세 손 오차)**: LJ6 64자세 L x+0.99 y-1.08 → 확인 영역 +24 x+1.03 y-1.37 (개선 없음) / 손목 +24 x+0.49 y-0.84 (약 절반). RJ6 R x-1.24 y+1.19 → 손목 +24 x-0.59 y+0.54. J3·J5 치우침 전파는 어느 쪽도 거의 그대로
- **해석**: 운영 조건의 노이즈 불확도는 이미 0.1~0.4 mm → 관찰된 mm 단위 앞뒤 오차는 고정값(Step 1 J6 등) 치우침이 약한 방향으로 번진 것. 자세 추가만으로는 J3/J5 치우침이 남고 J6는 절반만 줄어듦. 치우침을 없애려면 추가 자세 + Step 2에서 J6(필요시 J5·J3) 고정 해제가 필요 (전부 자유 시 가장 약한 특이값 1.9 → 5.7, J6 ∥ tilt 축퇴가 깨짐)
- **남은 일 (사용자 결정)**: 손목 다양화 자세 추가 여부, J6 고정 완화 여부, 시뮬(G4) 검증 착수

---

## 2026-09-29 — Step 2 약한 방향 보고 (2번) 구현
> 코드 변경 · 검증 수준 🟠 (15:13 데이터로 실제 `optimize_step2` 경로 재실행, 로봇 모델 읽기만) · G2 미실행
- **변경**
	- 새 compute 모듈 `core/calibration/observability.py`: 가중 야코비안(optimizer 자체 잔차 `se3_log(inv(T_model)·T_meas)` × 1/σ)을 유한차분으로 구해 SVD, 가장 약한 조합 4개와 확인 자세 손 이동량(mm/단위) 계산. 약하고(잔차 RMS 상승 < 0.3σ/단위) 손을 1 mm/단위 이상 움직이는 조합이 있으면 `[WARN]`
	- `sequences/step2.py`: 결과 저장 직전에 호출, 로그 출력 + 결과 JSON `observability`에 기록. 실패해도 보정은 계속하고 `[WARN]`만 남김
	- `tests/test_module_boundaries.py`: compute 순수성 검사 대상에 추가. `tests/test_step2_observability.py` 신규
- **검증**: 15:13 재실행 재현 0.038°. 약한 조합 크기·방향이 오프라인 분석과 일치 (0.085/0.100/0.112/0.157σ, J2·J4·J6 조합 확인 자세 오른손 x -0.8 / 왼손 x +1.8 mm)
- **구현 중 잡은 버그 2건 (테스트로 고정)**: `get_ready_pose`가 이미 라디안인데 `np.radians`를 한 번 더 적용 → 확인 자세가 0 자세로 계산됨. optimizer 가중이 꺼져 있으면 `weights()`가 1이라 rad·m가 섞임 → 추정 σ로 직접 가중
- **주의 (T25·T26)**: 15:13 결과 적용으로 `setting.yaml` Step 1 값과 `home_reset_baseline.json` 팔 오프셋이 0이 됨. 이후 15:13 재계산은 `result_backup/20260929_d405/` 입력으로 함

---

## 2026-09-29 — 15:13 적용 후 앞뒤 어긋남 (사용자: 위아래 OK, 왼쪽이 더 앞, 마름모에서 앞뒤로 흔들림)
> 코드 변경 없음 · 검증 수준 🟠 (15:13 데이터 오프라인, 모델 읽기용 로봇 연결 또는 저장 URDF) · 스크립트 `step2_depth_diag.py`, `step2_zscale_diag.py`, `step2_foreaft_pan.py`, `step2_coverage.py`, `step2_weak_dirs.py`, `j6_tools/run_analysis_(z)scaled.py` (스크래치) · 15:13 결과는 사용자가 적용함 (이후 분석에서 오프셋 재적용 금지)
- **기각한 가설**
	- baseline이 정답: 확인 자세 예측은 왼손이 오른손보다 1.1 mm 뒤 → 관찰(왼쪽 앞)과 반대
	- 헤드 pan 느슨한 고정: pan을 Step 1.5(0.97°)에 고정하면 잔차 0.172 → 0.190 mm로 악화, J2 좌우 벌어짐 남음 (R -0.81 / L +0.36)
	- 깊이만 배율 오차(초점거리형): z만 0.985~1.0 배율 → 잔차 0.167~0.181 mm, 관절값 변화 없음
	- 균일 배율(마커 크기 64 mm 오차): 0.99배에서 잔차 0.172 → 0.133 mm, 브라켓 z → 설계 -48 mm로 돌아오지만 r6은 54 → 53.6 mm로 벗어남, J2 벌어짐 그대로 → 부차 요인 가능성만 남김 (마커 실측으로 확인 가능)
- **확인된 사실: Step 2 샘플 범위가 좁음**. 64샘플의 손 위치 토르소 기준 앞 270~330 mm, 높이 240~300 mm, 관절 범위 J0 -78~-66, J1 ±68~78, J2 ±22~33, J4 ±19~30, J5 -46~-36, J6 ±34~41°. 확인 자세는 높이 0 mm, J0 -25, J1 ±40, J4 ∓124, J5 62, J6 ±58° → 샘플 영역 밖으로 외삽
- **약한 방향 (가중 야코비안 SVD)**: 가장 약한 6개 조합은 1°(또는 1 mm) 움직여도 Step 2 잔차가 σ 단위로 2.3~5.7밖에 안 늘어남 (768개 잔차, 현재 제곱합 694에 비해 무시 수준). 그중 `R6 -0.51, R4 +0.44, L4 +0.33, L6 -0.32, L5 +0.27, R2 -0.21` 조합은 확인 자세에서 1단위당 오른손 앞 +0.8 mm, 왼손 앞 -1.9 mm → 좌우 앞뒤 비대칭. `tilt, R4, L0, L4, R2` 조합은 양손 앞뒤 -2.0 mm
- **해석 (추정)**: 적용 결과가 baseline과 다른 관절(J2·J4·J6)이 바로 Step 2가 구분하지 못하는 조합. 이 조합은 Step 1 J6 고정값(왼팔 방향 방식 약 -1° 치우침)과 가중 방식이 정하고, 샘플 영역 밖인 확인 자세·마름모에서 mm 단위 앞뒤 오차로 드러남. 처음 증상("Step 2 자세는 맞는데 정의한 자세에서 안 맞음")과도 일치
- **남은 일 (사용자 결정)**: Step 2 자세 범위 확대(보정 수학·동작 계획 변경 → 노션 design 먼저 확인), 매 실행 약한 방향·확인 자세 민감도 보고(센서), 마커 한 변 실측(64 mm)

---

## 2026-09-29 — D405 교체 후 커플링 원인 정리 + 15:13 결과 유효성 (사용자: 확인 자세 만족, baseline보다 약간 나음)
> 코드 변경 없음 · 이 날 분석 전체 요약 · 백업 `result_backup/20260929_d405/` (11:36 baseline, 15:07 setting, 15:09 Step 1.5, 12시·15시 스윕)
- **증상**: D435 → D405 교체 후 Step 2가 J0를 양팔 같은 방향(-1.5/-1.8°), J4(-1.6/+0.6°)로 끌고 감. Step 2 자세에선 맞는데 사용자 정의 자세에서 어긋남
- **확인된 원인**: Step 2 잔차 가중(1/σ, σ = 잔차 RMS)의 방향 하한 0.01°. D405의 마커 방향 측정이 약해 J4가 방향 치우침을 흡수 → 방향 잔차 감소 → 방향 가중 증가의 순환 → J0 공통모드로 번짐
	- 조치: `step2.min_rot_noise_deg: 0.3` (A). D435 기록 재계산 변화 ≤0.09°, D405 J0 -1.5/-1.8 → 실기 2회 -0.07/-0.48, -0.22/-0.42. 위치 잔차 0.305 → 0.17 mm. 두 실행 차이 ≤0.17°
- **원인 아님 (오프라인 확인)**: 엔코더-마커 타이밍, 카메라 위치, 오른쪽 브라켓 재장착. 내부 파라미터 가설은 약화 (Step 1.5 카메라 단독 잔차 D435와 동일)
- **별개 문제 (해결)**: Step 1 스윕 "유효 점 1개" = 로봇 컨트롤러가 이동 중 상태 응답을 멈춤 → 로봇 재부팅으로 해소
- **남는 것 (추정·미해결)**
	- J6: 방향 방식은 D405 방향 측정·브라켓 yaw에, 위치 방식은 브라켓 옆방향 x에 1:1로 묶임 (게이지, 노션 원칙 3). 왼쪽 플레이트 옆 0.4~0.5 mm 이동 의심 (미확인). B1으로 두 값과 0.5° 경고를 기록 중, 채택은 방향 방식 유지
	- baseline 대비 J2 대칭(R -0.5 / L +0.7), LJ4·LJ6 차이: 사용자 확인 자세 판단으로는 문제 없음 → baseline(D435 시절 보정)도 정답이 아니므로 baseline 대비 0.15°는 참고 지표로만 사용
	- Step 1.5 헤드 tilt -0.87~-0.91 (카메라 교체 후 새 약 1° pitch를 규약 B대로 tilt가 흡수)
- **15:13 결과 유효성**: 같은 리셋(11:36:40) ✓, 미적용 상태 ✓, Step 1 4개 수렴·패스2 수렴 ✓, Step 1.5 성공(300/300점, 지연 78.6 ms — 오전 63~67보다 큼, 참고) ✓, Step 2 64샘플·결측 없음·하한 0.3 적용·위치 잔차 0.172 mm·오프라인 재현 0.038° ✓, 14:32와 차이 ≤0.17° ✓. 앱의 적용 가능 표시(OPTIMIZED 적용 버튼)는 GUI에서만 확인 가능
- **적용 시 주의**: Apply가 `home_reset_baseline.json` 팔 오프셋을 0으로 덮음 → 11:36 값은 백업본으로만 남음. 적용 후 분석에서 오프셋을 다시 더하지 말 것 (이중 적용)

---

## 2026-09-29 — 왼팔 J6 원인 오프라인 분석
> 코드 변경 없음 · 검증 수준 🟠 (저장 스윕 오프라인, 로봇 연결 없음, 저장 URDF) · 스크립트 `j6_tools/run_analysis.py`, `j6_tools/j6_pos_diag.py` (스크래치) · 15시 스윕 사본 `j6_tools/run_20260929_1500` (14:2x 스윕은 15시 실행에 덮여 없음)
- **재현**: 오전·15시 두 실행 모두 패스별 J6 두 방식이 GUI 로그와 소수 셋째 자리까지 일치
- **J6 값 (패스1/패스2, baseline R 2.174 / L 2.446)**
	- 왼팔 방향: 오전 1.824/1.597, 14시 1.670/1.492, 15시 1.755/1.410 → 실행 간 안정, baseline -0.6~-1.0, 매번 패스2가 0.1~0.35 낮음 (오른팔도 2.65→2.60, 2.78→2.68, 2.85→2.51)
	- 왼팔 위치: 오전 2.472/2.531 → 14시 2.996/2.918, 15시 2.977/2.777 → 오전과 오후 사이 +0.3~0.5 이동
	- 오른팔 위치: 2.088/2.389, 2.171/2.196, 2.020/2.117 → 안정
- **같은 자세 비교**: 패스1은 스테이징 0이라 명령 자세가 매번 같음. 왼팔 패스1 방향 1.824 → 1.755 (-0.07)인데 위치 2.472 → 2.977 (+0.51). 회전은 그대로, 마커 위치만 J6 둘레로 약 0.5° (반지름 54 mm에서 약 0.45 mm) 옆으로 이동. 오른팔은 같은 비교에서 위치 -0.07
- **위치 방식 점검**: 스윕 양 끝 따로 계산해도 일치 (차이 대부분 ≤0.1°, 최대 0.3°) → 지연·원 중심 문제 아님. J5 축(n5)을 n4×n6로 바꾸면 0.3~1.2° 달라지지만 n4 쪽 원 법선도 약한 방향(시상 7~10 mm)이 있어 어느 쪽이 맞는지 판단 불가. J5 스윕 40°, J6 스윕 60°
- **해석 (추정)**: 위치 방식 J6 = 브라켓 옆방향 x(0 고정, 추정 안 함)에 1:1 (1 mm → 1.06°). 방향 방식 J6 = 브라켓 yaw(설계값 고정) + 마커 방향 측정 치우침. 왼팔 위치값만 오전→오후 이동, 방향값·오른팔은 그대로 → **왼쪽 마커 플레이트가 12:16~14:2x 사이 옆으로 약 0.4~0.5 mm 밀렸을 가능성** (미확인, 측정 쪽 원인도 배제 못 함). 왼쪽 브라켓 z는 조립 한계(-50.0 mm, 설계 -48 ±2)에 걸리는 경고도 반복
- **결론**: 이 스윕으로는 J6와 브라켓 장착(x·yaw)을 분리할 수 없음 (노션 원칙 3, J6 게이지). 0.15° 목표는 플레이트 옆방향 ±0.14 mm 재현을 뜻함
- **남은 일 (사용자)**: 왼쪽 플레이트 볼트·유격 확인, 12~14시 사이 왼팔·플레이트를 건드렸는지 확인

---

## 2026-09-29 — 실기 재실행 2회차 (15:09 Step 1.5 → 15:13 Step 2, 미적용)
> 코드 변경 없음 · 검증 수준 🟢 실기 실행 + 🟠 오프라인 재계산 (`step2_rerun_1513.py`, 스크래치, 재현 0.038°) · baseline 11:36:40 (같은 리셋, 허용 0.15°)
- **Step 1**: J3 R -2.263 / L -1.890, J5 R -0.393 / L -0.039 → baseline 대비 ≤0.08°. J6 방향 R 2.853/2.511, L 1.756/1.411 (baseline R 2.174, L 2.446 → R +0.34, L -1.03). 위치 R 2.019/2.118, L 2.978/2.777
- **Step 2 − baseline**: R -0.22 -0.03 -0.51 +0.04 -0.41 -0.09 -0.32 / L -0.42 +0.08 +0.73 -0.11 -0.70 -0.07 +0.99, 헤드 +0.35/-0.84, 0.172 mm. 14:32와의 차이 ≤0.17° → **반복성 양호, baseline과의 차이는 계통적**
- **J6 바꿔 재계산**: baseline J6 → RJ2 -0.34, RJ4 -0.35, LJ0 -0.17, LJ1 +0.19, LJ2 +0.41만 초과 (LJ4 -0.07, LJ6 -0.02로 해소). 위치 방식 J6 → RJ2 -0.27, RJ4 -0.34, LJ1 +0.24, LJ2 +0.31, LJ4 +0.17, LJ6 -0.34
- **해석**: 14:32와 같은 결론. 왼팔 손목(LJ4·LJ6) 초과는 Step 1 왼팔 J6(방향 방식, 3회 연속 baseline -0.85~-1.03)가 원인. J2 대칭·RJ4는 J6와 무관하게 남음
- 참고: `result_20260929_145653.json` (데이터셋 없음, 헤드 tilt -0.04)은 이번 비교에서 제외 (출처 미확인)

---

## 2026-09-29 — 하한 0.3° 적용 후 실기 재실행 (14:1x Step 1 → 14:28 Step 1.5 → 14:32 Step 2, 미적용)
> 코드 변경 없음 · 검증 수준 🟢 실기 실행 + 🟠 오프라인 재계산 (`step2_rerun_1432.py`, 스크래치, 재현 0.016°) · 기준 baseline 11:36:40 (같은 리셋, 허용 0.15°)
- **Step 1 반복성**: J3 R -2.162 / L -1.922, J5 R -0.365 / L -0.075 → baseline 대비 ≤0.07°. Step 1.5 헤드 pan 0.93 → 0.96, tilt -0.88 → -0.87 (오전과 0.03° 이내)
- **J6 두 방식 (B1 로그)**: 방향 R 2.775/2.678, L 1.670/1.492 (baseline R 2.174, L 2.446 → R +0.50, L -0.95). 위치 R 2.171/2.196 (baseline과 0.02°), L 2.996/2.918 (+0.47, 오전 2.47/2.53에서 이동) → **왼팔 위치 방식은 실행 간 0.45° 흔들림**
- **Step 2 결과 − baseline**: R -0.07 -0.13 -0.68 -0.05 -0.32 -0.11 -0.49 / L -0.48 +0.00 +0.63 -0.09 -0.62 -0.03 +0.91, 헤드 +0.27/-0.80, 위치 잔차 0.177 mm. 오전(12:23) J0 -1.52/-1.84 → -0.07/-0.48 (하한 효과 실기 확인). 0.15° 초과: RJ2·RJ4·RJ6·LJ0·LJ2·LJ4·LJ6
- **J6 바꿔 재계산**: baseline J6 → RJ2 -0.46, RJ4 -0.39, LJ0 -0.21, LJ2 +0.33만 초과, 0.136 mm. 위치 방식 J6 → RJ2 -0.38, RJ4 -0.31, LJ1 +0.16, LJ2 +0.21, LJ4 +0.26, LJ6 -0.49. 오른 위치 + 왼 baseline → baseline J6와 같음
- **해석**: 하한으로 J0 공통모드는 해결. J6가 맞으면 LJ4·LJ6·RJ6 초과는 사라짐. J2 대칭(R -, L +)과 RJ4는 J6와 무관하게 남음 (미해결). 왼팔 J6는 두 방식 모두 신뢰 부족
- **남은 일**: J2 대칭 잔차 원인 (공통 yaw·헤드 pan과의 결합 가설, 미확인), 왼팔 J6 위치 방식 흔들림 원인, 이 결과 적용 여부 (사용자)
- **미리보기 잠금 수정 (사용자 지적)**: 적용 불가 결과는 영점·확인 자세 미리보기까지 막혀 비교를 못 함 → 미리보기(`move_zero`/`move_check`)는 항상 허용하고 경고 로그(`[WARN] Preview only ...`)만 남김, 적용(`apply`)만 차단. 대화상자·위자드 버튼과 문구(i18n `opt_locked`) 수정, `tests/test_home_offset_apply_guard.py` 갱신. G1·import 통과, G2 미실행
- **롤백 미리보기 에러 원인 (확인)**: BASELINE 선택 상태에서 "cannot be applied" 에러. `_is_home_reset_baseline`이 최상위 `type`만 봤는데 실제 파일은 `metadata.type` → 기준값을 Step 2 결과로 오판. 테스트 가짜 파일이 최상위 `type`이라 못 잡음 → 두 위치 모두 확인하도록 수정, 테스트 가짜 파일을 실제 형식으로 바꾸고 `build_home_reset_baseline_data` 출력으로 검사하는 테스트 추가. 실제 `config/home_reset_baseline.json` 판정 True 확인
- **정정**: OPTIMIZED 버튼이 회색인 건 선택하지 않은 상태 표시였음 (사용자 확인). 14:32 결과 잠김 아님
- **G2**: 전체 212개 통과 212 / 실패 0 / 오류 0 / skip 0
- ~~미확인: 14:32 결과가 왜 적용 불가로 판정됐는지~~ → 해당 없음 (위 정정)

---

## 2026-09-29 — Step 2 방향 노이즈 하한 0.3° (A) + Step 1 J6 두 방식 동시 보고 (B1)
> 코드 변경 · 검증 수준 🟠 (저장 데이터로 실제 `optimize_step2` 오프라인 재계산, 로봇 읽기 전용 연결) · 스크립트 `step2_floor_validate.py`, `j6_tools/run_analysis.py` (스크래치)
- **사용자 결정**: 하한 0.3°로 A·B1 같이 진행. baseline(11:36) 신뢰, 허용 오차 0.15°
- **A 변경**: `config/setting.yaml` `step2.min_rot_noise_deg: 0.3` → `data.load_step2_min_rot_noise_deg` (없으면 0.3 + 경고, 0 이하·비유한값은 ValueError) → `step2.optimize_step2`가 세 optimizer(pass 1, pass 2, 한 팔)에 `min_rot_noise_std_rad` 전달 → `QPCalibrationOptimizer`가 `ResidualNoiseEstimator(min_rot_std_rad=...)`로 넘김. 결과 JSON `measurement_noise_min_rot_std_deg`에 기록
- **B1 변경**: `JointCalibrator.choose_wrist_yaw2_offset` 추가. J6 계산 때 방향·위치 두 추정을 항상 로그(`[INFO] wrist_yaw2 J6 estimates: ...`)와 결과(`wrist_yaw2_orientation_deg`, `wrist_yaw2_position_deg`, `wrist_yaw2_method`)에 남기고, 차이 0.5° 초과 시 경고. 채택 방식은 `WRIST_YAW2_METHOD`(기본 orientation, 동작 변화 없음)
- **A 검증** (하한 0.01°로 저장 결과 재현 후 0.3°와 비교)
	- D435 09-22 14:25 / 16:43: 모든 오프셋 변화 최대 0.09° / 0.00° → 허용 0.15° 이내 (회귀 없음)
	- D405 09-29 12:23: J0 +1.17/+1.16, J4 +1.04/-0.96 이동, 위치 잔차 0.305 → 0.168 mm
	- D405 결과 − baseline (허용 0.15°): J6 실행값(방향) → RJ2·RJ4·RJ6·LJ0·LJ2·LJ4·LJ6 초과. J6 위치 방식 → RJ2 -0.46, RJ4 -0.21, RJ6 -0.20, LJ2 +0.39 초과
	- 재현 조건 주의: J3/J5 앵커를 결과 파일 값(±0.05°)에서 가져와 재현 오차 0.35° → D405 데이터는 조건이 약함. 앞선 정확한 앵커 재계산값은 J0 -0.13/-0.48(실행 J6), -0.02/-0.21(위치 J6)
- **B1 검증**: 실제 경로가 분석값 그대로 보고 — 방향 R 2.652/2.604, L 1.824/1.597 / 위치 R 2.088/2.389, L 2.472/2.531. 0.5° 초과 경고 발생, 채택값 변화 없음
- **회귀 테스트**: `tests/test_step2_rot_noise_floor.py` (하한 추정기·optimizer 전달·설정 로더·optimize_step2 배선·J6 선택). G2 전체 210개: 통과 210 / 실패 0 / 오류 0 / skip 0 (첫 실행은 새 테스트의 가짜 설정에 `arm_idx` 누락으로 오류 1 → 테스트만 수정)
- **남은 일**: B2(J6 위치 방식 전환, 왼쪽 플레이트 옆방향 장착 먼저 확인 — D435 때 왼팔 위치 방식 1.6~1.9° 어긋남), J2 대칭 잔차 분석, C 실기 재실행·확인 자세 검증(사용자), D 내부 파라미터 재보정(최후)

---

## 2026-09-29 — D405 교체 후 Step 2 커플링 (J0 공통모드) 오프라인 분석
> 코드 변경 없음 · 검증 수준 🟠 (실제 데이터 오프라인 재계산, 로봇 읽기 전용 연결) · 스크립트 `step2_resolve_0929.py` (스크래치)
- **상황**: 로봇 재부팅 후 Full Auto 정상 수집. 11:36 Home Offset Reset(baseline, 사용자: 신뢰 가능) → Step 1 → 1.5(12:18) → Step 2(12:23, 미적용). Step 2 자세에선 잘 맞는데 사용자 정의 자세에서 어긋남. 카메라는 오늘 D435 → D405 교체
- **결과 − baseline**: J3·J5 ≤0.07° 일치. J0 R -1.52 / L -1.84 (같은 방향), J1 -1.07/+0.86, J4 -1.59/+0.62, J6 -0.42/+0.82, 헤드 pan +0.29 / tilt -0.83
- **재계산 (A 실행 재현 0.01° 이내)**
	- 헤드 tilt 기준 1.5(-0.88) → 0: J0 -0.40/-0.73, 잔차 0.344 → 0.313 mm (헤드 tilt와 J0 공통모드 커플링 확인)
	- J6 기준 Step 1 → baseline: J0 -1.16/-1.39, 잔차 0.275 mm. 둘 다: J0 -0.33/-0.55, 잔차 0.256 mm (최소)
	- 공장 내부 파라미터로 근사 재투영: 잔차 악화(0.313→0.390) → 내부 파라미터 불일치 가설 비지지 (근사)
	- Step 1.5 카메라 회전 유지 / 카메라 위치 1.5 vs 기준값: 영향 없음
	- J4 R -1.5 / L +1.0은 모든 조건에서 남음 (미해결)
- **Step 2 자세**: 오늘은 대칭 자동 준비 자세(간격 14.9 cm, 벌리기 창 없음). 09-22 16:43은 오른팔 티칭 자세(간격 11.3 cm)
- **Step 1.5 이력**: 09-22 (D435, cam y 33 mm) tilt -0.12/+0.16, pitch -0.03/-0.05 → 오늘 (D405, cam y 8 mm) tilt -0.88, pitch -0.48. 카메라 교체 후 약 1° 새 pitch
- **Step 1 J6**: 패스 간 반복 좋음(R 2.65/2.60, L 1.82/1.60), baseline 대비 R +0.43 / L -0.85 (브라켓 J6 게이지 영향 가능, 미확인)
- **사용자 결정**: 헤드 tilt는 현 규약(B, tilt가 카메라 x회전 흡수) 유지
- **J6 원인 분석** (`j6_tools/run_analysis.py`, 09-22 도구 재사용, 오프라인): 오늘 방향 방식 R 2.65/2.60, L 1.82/1.60 (실제 로그 재현) vs 위치 방식 R 2.09/2.39, L 2.47/2.53 vs baseline R 2.17, L 2.45 → 위치 방식이 baseline과 0.1~0.2° 일치. 방향으로 본 J5 축이 엔코더 축과 R 2.3~3.7°, L 0.6~2.2° 어긋남. 09-22 (D435)에선 왼팔 위치 방식이 1.6° 어긋났음(당시 눈 확인은 방향 방식과 일치)
- **노션 확인** (design 4.1/4.3/5.1, project_harness 원칙 3·T11·T23): J6 ∥ head tilt 정렬도 1.000(Step 2) → J6 오차가 tilt·J0로 번짐. design 4.1에 보정 파일 D405의 공장 cy 347.9 기록, 오늘 카메라 공장 cy 361.0 → **다른 D405 개체일 가능성 높음** (카메라 교체 워크플로 미완성, T11)
- **Step 1.5 재계산** (`j6_tools/step15_resolve_0929.py`, 헤드 스윕 기록 1,810점, 위치만 사용): 사용 중 내부 파라미터 tilt -0.90 (실행 재현), 이 카메라 공장 내부 파라미터 tilt -0.21, 카메라 위치 43.2/8.2/62.3 → 44.7/9.6/58.3 mm (CAD 47/9/57). RMSE 0.183 → 0.235 mm (왜곡 미반영 근사)
- **사용자 결정**: 내부 파라미터 재보정(0단계)은 최후. 현재 보정을 기준으로 분석
- **타이밍 점검** (`j6_tools/sync_check.py`, Step 1 스윕 36개 + Step 1.5): 엔코더 갱신 안 됨 0~0.7%, 마커-엔코더 밀림 -0.8~+2.9프레임(지난주 -1.6~+4.0), 밀림 보정해도 오차 불변 → 타이밍 원인 아님. Step 1.5 샘플 간격 63~70 ms, 지연 63 ms(지난주 67~68)
- **Step 2 화면 위치 잔차** (`j6_tools/step2_pixel_residuals.py`): 위치 잔차 D435 0.23/0.24 mm → D405 0.58 mm, y·z 잔차의 76~80%가 화면 세로 위치·거리로 설명(D435 40~55%)
- **Step 1.5 카메라 단독 잔차** (팔 정지, D435 백업 기록과 비교): D435 0.192/0.186 mm, D405 0.183 mm(사용 중 내부 파라미터), 0.235 mm(공장값) → **카메라 측정 일관성은 D435와 같고 사용 중 내부 파라미터가 공장값보다 잘 맞음. "다른 개체 보정값" 가설은 약화(정정)**
- **팔 쪽 입력 교체** (Step 2 재계산): 브라켓 z -49.81/pitch 0.003(오늘) vs -49.24/0.047(09-22). 09-22 브라켓 → 잔차 0.288 mm, + baseline J6 → **0.218 mm**, J0 -0.87/-1.14 → Step 2 악화의 주원인은 오늘 Step 1의 브라켓·J6 (마커 방향 기반 추정). 오른팔 J4 스윕의 방향 일관성 0.43~0.49° (지난주 0.17~0.25°)
- **브라켓 성분별 출처** (`MarkerCalibrator.compute_unified_bracket_calibration`): roll 90 고정, yaw 설계값 고정, pitch = 원 궤적 축을 마커 방향으로 옮겨 계산(방향 의존), x·y·z = 엔코더 기준 반지름 r4·r5·r6(위치+엔코더). 오늘 z -0.57 mm는 r5가 양팔 약 +1 mm(175.4~175.7 vs 174.1~174.9)인 결과, r6은 동일
- **성분 분리 재계산**: 브라켓 z만 09-22 값 → 0.219 mm (L과 같음), pitch만 → 0.274 mm (효과 없음) → **z(r5)가 원인, pitch 아님**
- **J4 고정 시험** (스크립트 안에서만 get_joint_offset_limits 감쌈, J4 = baseline ±0.05°): 09-22 브라켓 + baseline J6 + J4 고정 → 위치 0.129 mm / 방향 0.223° (지난주 D435 수준), J0 +0.22/-0.38, J1 +0.33/+0.22, J2 -0.57/+0.11. 실행 조건 + J4 고정 → 0.195 mm / 0.270°. J4 자유일 때 방향 잔차를 줄이려고 J4가 -1.1~-1.6 / +0.6 끌려감 → Step 2에서 J4는 마커 방향 측정이 주도
- **해석 (추정)**: 오늘 Step 2 악화 = ① 브라켓 z(r5 +1 mm, 원인 미상: 재장착 또는 깊이 배율) ② Step 1 J6 방향 방식 치우침 ③ Step 2 J4가 방향 측정에 끌림. ②③ 모두 마커 방향 측정 약화(D405)와 일치, 위치 측정은 정상
- **사용자 확인**: 오른쪽 브라켓만 오늘 재장착(볼트 더 조여짐, 약 0.5 mm 변화 가능). 왼쪽은 그대로인데 r5 +0.6~0.8 mm, 지난주 같은 날 두 실행 사이에도 반지름 0.4~0.8 mm 흔들림
- **팔별 브라켓 재계산** (J6 baseline, J4 고정): 양팔 지난주 0.129 mm / 0.223°, 오늘 팔별(대칭화 전) 0.133 / 0.333°, 오른 오늘+왼 지난주 0.129 / 0.277°, 오른 지난주+왼 오늘 0.132 / 0.278° → **J4·J6가 맞으면 브라켓 z 차이(0.4~0.8 mm)는 위치 잔차에 거의 영향 없음**. 앞선 "z가 원인"은 J4가 자유일 때 z와 J4가 결합(노션 5.1: 브라켓 위치 → J4 0.13°/mm)한 결과. 모든 조합에서 오른팔 J2 -0.52~-0.61 잔존
- **정리**: 주원인은 마커 방향 측정에 기대는 Step 1 J6(방향 방식)과 Step 2 J4. 브라켓 재장착·대칭화는 부차적
- **가중 기전** (`ResidualNoiseEstimator`): 가중 = 1/σ, σ는 잔차 RMS, 방향 하한 0.01°(사실상 없음). 오늘 σ_rot 0.114° / σ_pos 0.345 mm (지난주 0.17~0.30° / 0.12~0.14 mm) → 방향 가중 과대 → J4가 방향 치우침 흡수 → 방향 잔차 감소 → 가중 더 증가의 순환
- **방향 노이즈 하한 시험** (스크립트 안에서만 하한 변경, J4 자유): 실행 + 하한 0.3° → J0 -0.13/-0.48, 위치 0.173 mm. J6 위치 방식 + 0.3° → J0 -0.02/-0.21, J4 -0.26/+0.07, J6 -0.20/-0.11, 위치 0.140 mm. + 0.5° → J0 +0.19/-0.03, J4 -0.07/-0.15, 0.128 mm. J6 위치 방식만(하한 없음) → J0 -1.24/-1.36 (재발). 남는 것: J2 R -0.4~-0.6 / L +0.35~+0.6 (대칭 성분)
- **남은 일**: 전체 계획 제시 → 사용자 결정(하한 값, J6 방식 전환 절차). 내부 파라미터 재보정은 최후

---

## 2026-09-29 — Step 1 J4 스윕 "유효 점 1개" 원인 진단 (실기)
> 코드 변경 없음 · 검증 수준 🟢 (실기 192.168.1.40, 스크래치 스크립트) · 테스트 해당 없음
- **증상**: Full Auto Step 1 오른팔 wrist_pitch Joint A(J4 ±15°, 8 s) 스윕이 매번 1점 → 재조정 창 반복. 노출·마커 인식은 정상(위자드 모니터 오른쪽 100%, RMS 0.04 mm)
- **측정 1** (`step1_j4_sweep_diag.py`, 수동 노출 6000): 재현. 스윕 8 s 동안 마커 호출 1회(J4=105°, 끝 각도), 호출 자체는 16 ms
- **측정 2** (`movej_blocking_diag.py`, J4 +5°/4 s): 파이썬 심장박동 431/430 정상, 카메라 135프레임 정상, `robot.get_state()` 1회 4,500 ms → 모션 대기 중 상태 조회가 막힘
- **측정 3** (`state_stream_diag.py`, `state_while_moving_diag.py`): `start_state_update` 스트림 3개/4.5 s, 별도 SDK 연결 1개, `.get()` 없이 `is_done()` 폴링 1개 → 모두 막힘. 클라이언트 코드가 아니라 로봇이 명령 실행 중 상태를 내주지 않는 것으로 판단
- **확인된 사실**: 스윕 코드(`CalibratorBase.perform_single_joint_sweep`, `HeadCameraCalibrator.sweep_head_joint`)와 `movej`는 09-15 커밋 이후 동일, 클라이언트 SDK 0.10.0(09-02 설치) 동일, `start_state_update`는 현재 코드에서 미사용. 로봇 보고 버전: version 0.10.7, sdk_version 0.9.1 (commit a3274bb), 모델 A v1.2
- **추정 (미확인)**: 09-22 이후 로봇 쪽 소프트웨어·설정 변경. Step 1.5 헤드 스윕도 같은 구조라 같은 증상 예상
- **남은 일**: 로봇 소프트웨어 변경 여부 사용자 확인 → 로봇 재부팅 후 재측정 → 지속되면 우회안(정지-측정 반복 스윕 등) 설계 논의
- **로봇 상태**: 오른팔 wrist_pitch 준비 자세(J4 90°), 왼팔 영점에서 종료

---

## 2026-09-29 — 홈 오프셋 중복 적용 방지 (이번 실행·현재 영점에서 계산한 결과만 1회 적용, 나머지는 롤백만)
> 커밋 기준 `39c2083` · 검증 수준 🟠 (G2 오프라인 테스트) · 테스트 201/0/0/0 (신규 `test_home_offset_apply_guard` 11개)
- **원인** (코드 분석): 적용 = "현재 영점 + 결과 보정값" 자세로 이동 → 그 자세를 새 영점으로 리셋. 결과 JSON에 적용 표시가 없고 버튼도 다시 켜져, 같은 결과를 다시 적용하면 보정값이 2배. 재시작 후에도 최신 결과 파일을 다시 골라 동일
- **결정** (사용자, 2026-09-29): 실행 중에는 적용 플래그로 막고, 재시작 후에는 Calculate 전까지 최적화 적용 불가·롤백만. npz로 계산한 결과는 적용 불가(A안)
- **변경**
	- `calibration_core.py`: `home_epoch`(영점이 바뀔 때마다 +1: Home Offset Reset·Apply, 실패해도 +1), `applicable_result_path`, `result_apply_allowed()`. `optimize`는 `data_home_epoch`가 현재 세대와 같을 때만, `step2`는 샘플 태그가 모두 현재 세대일 때만 결과를 적용 가능으로 등록. `home` 작업의 move_zero·move_check·apply는 baseline JSON(`type: home_reset_baseline`)이 아니면 적용 가능 결과만 허용 (core에서 예외)
	- `sequences/collection.py`: 수집 샘플에 `home_epoch` 태그
	- `ui/core_bridge.py` `Step2CalculateWorker(data_home_epoch=)`. `main_ui.py`: 공유 샘플 세대 추적(`shared_sample_epochs`), live/sim 계산은 모든 샘플이 현재 세대일 때만 epoch 전달(아니면 경고 로그), npz는 None. 적용 창: 최적화 토글 잠금 + 사유 문구, 롤백은 baseline 있으면 항상 가능, 함수 단 이중 확인
	- `ui/wizard_widget.py`: 5번 슬라이드 최적화 버튼 3개 잠금 + 사유 문구, 롤백 버튼 유지, 함수 단 이중 확인
	- `i18n.yaml` `dialogs.apply_home_offset.opt_locked(_title)` en/ko
	- 테스트: `tests/test_home_offset_apply_guard.py` 신규. `test_system_regressions.py`의 위자드 미리보기 테스트는 전제에 "적용 가능 결과"를 추가 (최적화 버튼이 이제 잠김)
- **G2**: `unittest discover -s tests` 201개 통과/실패 0/오류 0/skip 0 (2026-09-29)
- **검증 경계**: 실기 미실행. 실제 적용 후 로봇 재연결 흐름에서 버튼 상태 미확인
- **미커밋 파일**: `core/calibration/calibration_core.py`, `core/calibration/sequences/collection.py`, `ui/core_bridge.py`, `ui/wizard_widget.py`, `main_ui.py`, `config/ui_config/i18n.yaml`, `tests/test_home_offset_apply_guard.py`, `tests/test_system_regressions.py`, `docs/progress.md`

---

## 2026-09-28 — 위자드 슬라이드 개편 (사진 교체, 마커 브라켓 슬라이드, 노출 슬라이드 이동 + 마커 인식·흔들림 모니터)
> 커밋 기준 `39c2083` · 검증 수준 🟠 (G2 오프라인 테스트 + 위자드 offscreen 점검) · 테스트 190/0/0/0
- **목표**: 사용자 요청 — 홈 오프셋 사진을 shoulder_roll/elbow 2열로, 카메라 브라켓 사진 head/nohead, 그리퍼 분리 뒤 마커 브라켓 부착 슬라이드 추가, 노출 슬라이드를 홈 오프셋 리셋 뒤·캘리브레이션 시작 앞으로 이동하고 티칭 버튼으로 마커 인식 밝기 확인 안내 + 마커 값 튀는 정도 모니터
- **변경**
	- `ui/wizard_widget.py`: 슬라이드 번호를 `SLIDE_*` 상수로 (main_ui도 사용). 새 순서 1-1 카메라 브라켓 → 1-2 그리퍼 분리 → 1-3 마커 브라켓(신규) → 1-4 → 2 → 3 → 3-1 → 3-2 → 3-3 노출·인식(이동) → 4 → 5. 슬라이드는 dict에 모아 SLIDE 순서로 스택에 추가, 누락 시 예외
	- 1-1: `camera_bracket_head/nohead` 2열 + 캡션. 1-2: `marker_connect` 사진 제거, 그리퍼 해체 링크만. 1-3: `marker_bracket_assemble` + `marker_bracket_overview`, 기존 1-2의 브라켓·마커 표면 문구 이동 + 좌우 표시 안내
	- 3-2: `offset_reset_pose_shoulder_roll` / `offset_reset_pose_elbow` 2열, 방향 문구는 기존 그대로 + 어깨 roll 주의(몸체에 닿을 만큼 넣지 말 것). 메인 화면 리셋 확인창은 `offset_reset_pose_direction.png` + 같은 주의 문구
	- 3-3: 티칭 버튼으로 두 마커를 화면에 넣고 확인하라는 안내 + 마커 인식 모니터 (마커별 인식 여부·인식률·흔들림 RMS/최대 mm, 최근 30회 원시 검출)
	- 흔들림 기준 (사용자 결정 2026-09-28): RMS 0.2 mm 이하 = 성공. `setting.yaml` `exposure_check.max_marker_jitter_rms_mm: 0.2` (없으면 `DEFAULT_MAX_MARKER_JITTER_MM` 0.2 + WARN). 마커별 초록(인식+기준 이하)/주황(인식, 기준 초과)/빨강(미인식), 두 마커 모두 초록이면 "노출 설정 성공" 줄. '적용 완료' 체크는 막지 않음
	- `core/calibration/sequences/marker_monitor.py` 신규 (`MarkerMonitor`, 별도 스레드, `use_filter=False`), `CalibrationCore.start/stop_marker_monitor` — 시퀀스 시작(`prepare_run`)·카메라 재연결·종료 시 자동 정지. `data.py` `marker_position_jitter_mm`, `marker_monitor_summary`. `CoreBridge.marker_monitor` 신호
	- `i18n.yaml`: 위 문구 en/ko. `slide_1.inst2/inst3` 삭제 (→ `slide_marker_bracket`)
	- 테스트: `tests/test_wizard_slides.py` 신규
- **검증**: offscreen 위자드 점검 — 11장, 제목 한/영, 다음 0→1→2→3→5→…→10 / 이전 역순, 건너뛰기 4·7번, 상태 라벨 위치, 이미지 누락 0, 모니터는 노출 슬라이드에서만 실행·이탈/시퀀스 시작 시 정지, 가짜 입력으로 인식률·흔들림 표시 확인
- **G2**: `unittest discover -s tests` 190개 통과/실패 0/오류 0/skip 0 (신규 `test_wizard_slides` 16개). 첫 실행에서 이미지 존재 테스트가 `main_ui.py`의 후보 목록(`warning_pose_check.png` → `warning_pose.png` 폴백)을 결함으로 잡음 → 실제 결함 아님, 테스트를 "후보 중 하나 존재"로 수정
- **검증 경계**: 실제 화면(한글 글꼴·크기), 실기 카메라에서 모니터 주기·실제 흔들림 값(0.2 mm 기준이 현장에서 달성 가능한지), 로봇 연결 상태의 위자드 전체 흐름 미확인
- **남은 일**: 실제 화면·실기 확인, 1-1·1-3 캡션 문구 사용자 확인 (사진에서 해석한 설명)
- **미커밋 파일**: `ui/wizard_widget.py`, `ui/core_bridge.py`, `main_ui.py`, `core/calibration/calibration_core.py`, `core/calibration/data.py`, `core/calibration/sequences/marker_monitor.py`, `config/setting.yaml`, `config/ui_config/i18n.yaml`, `tests/test_wizard_slides.py`, `docs/progress.md`

---

## 2026-09-28 — Step 2 Init Pose 마커 간격 확보 (카메라 x < 11 cm면 벌리기 창 + 실시간 모니터)
> 커밋 기준 `39c2083` · 검증 수준 🟠 (G2 오프라인 테스트 + 대화상자 offscreen 점검) · 테스트 174/0/0/0 (신규 18 포함)
- **목표** (사용자 시나리오): Init Pose → 마커 안 보이면 기존 재조정 → 완료 후 간격 확인 → 좁으면 안내 메시지 → 간격 모니터 창 (빨강 → 11 cm 이상 초록) → 완료 → 이후 간격 확인 없이 Step 2 진행
- **확인**: 마커 좌표는 실기·시뮬 모두 카메라 좌표계 (camera→marker, m). 카메라 x 사용 (사용자 결정)
- **변경**
	- `config/setting.yaml` `step2.min_marker_x_gap_m: 0.11`. 설치본에 키가 없으면 `data.DEFAULT_MIN_MARKER_X_GAP_M`(0.11)을 쓰고 WARN 로그
	- `core/calibration/data.py`: `marker_x_gap_m`, `load_min_marker_x_gap_m`, `MarkersTooCloseError`
	- `sequences/collection.py` `ensure_marker_spacing`: 가시성 확인(`verify_and_align_head_at_ready_pose`) 직후 1회 측정 → 좁으면 `core.prompt_marker_spacing` 호출 후 core가 0.2 s마다 재측정해 `marker_gap` 이벤트 송신 → 운영자 완료 시 재확인 없이 반환. 취소는 실패, Stop은 취소. 콜백이 없으면 `MarkersTooCloseError`
	- 호출 위치: `step2_ready`(GUI Init Pose), `run_collection(prepare=True)`(Step 2·Full Auto·Wizard). GUI Auto Motion(`prepare=False`)은 검사 없음
	- `ui/core_bridge.py` `marker_gap` 신호. `main_ui.py` `MarkerSpacingDialog`(실시간 카메라 화면, 간격 빨강/초록, 초록일 때만 완료 가능, core가 모니터링을 끝내면 자동 닫힘), `prompt_marker_spacing`(안내 메시지창 → 모니터 창)
	- `i18n.yaml` `dialogs.markers_too_close.*` en/ko 9개
	- 테스트: `tests/test_step2_marker_spacing.py` 신규 (18개)
- **오프라인 조사** (저장된 Step 2 데이터셋 27개, 스크래치 스크립트): 두 마커는 카메라 x로 벌어져 있음 (y 간격 중앙값 0~1.2 cm). 첫 포즈 x 간격 < 11 cm가 7개 (09-21~22, 최소 9.2 cm) → 이제는 벌리기 창이 떴을 실행들. 수집 중 간격은 시작보다 최대 약 1.7 cm 줄어듦
- **검증**: offscreen 대화상자 점검 — 9.3 cm·마커 없음·10.99 cm 빨강+완료 비활성, 11.0·12.4 cm 초록+활성, 빨강에서 완료 무시, core 종료 시 자동 닫힘. 10.99 cm가 "11.0"으로 보이던 반올림을 버림 표시로 수정
- **G2**: `unittest discover -s tests` 174개 통과/실패 0/오류 0/skip 0 (2026-09-28, 신규 `test_step2_marker_spacing` 18개 개별 확인)
- **검증 경계**: 시뮬·실기 미실행. 실제 화면에서 메시지창→모니터 창 순서, 실기 검출 속도(0.2 s 주기 유지 여부), 티칭 중 검출 흔들림 미확인
- **남은 일**: 시뮬(G4)·실기(G5) 확인, 간격이 좁았던 7개 실행에서 실제 충돌이 있었는지 사용자 확인 (기준값 판단은 사용자)
- **미커밋 파일**: `config/setting.yaml`, `config/ui_config/i18n.yaml`, `core/calibration/data.py`, `core/calibration/sequences/collection.py`, `core/calibration/calibration_core.py`, `ui/core_bridge.py`, `main_ui.py`, `tests/test_step2_marker_spacing.py`, `.claude/settings.json`, `CLAUDE.md`, `docs/progress.md`

---

## 2026-09-26 — UI 매뉴얼 확장 (PySide6 기본 + 웹 부록, CS 분석기 구조 반영)
> 코드 변경 없음 (노션 문서만) · 검증 수준 🟠 (골격 offscreen 13/13) · 테스트 미실행
- **목표**: 사용자 결정 "PySide6 메인, 웹은 드물게"에 맞춰 UI 매뉴얼을 PySide6 본문 + 웹 부록으로 개편
- **변경 (노션 UI 매뉴얼)**
	- 3.2 화면 공통 순서(상단 바 → 상태 바 → 빈 화면/탭 → QUIT), 3.3 작업 영역 틀 2종(조작형 3열 / 분석형 카드 + 목록·상세)
	- 3.5 패턴 표에 CS 분석기 출처 7개 추가 (상태 바, 빈 화면, 드래그앤드롭, 요약 카드 필터, 목록·상세, 접히는 경고, 보조 관리 창)
	- 8장 골격을 6파일로 분리: `theme.yaml`, `ui/theme.py`, `ui/worker.py`(진행률·협조식 취소), `ui/widgets.py`, `ui/workspaces.py`, `main_ui.py`
	- 9장 "따라 하지 말 것", 10장 체크리스트 갱신, 부록 A 웹 (`tools/theme_to_css.py`, Qt↔웹 역할 속성 대응)
- **검증**: offscreen 동작 점검 13/13 (빈 화면 시작, 열기→분석 탭, 카드 필터, 검색, 경고 영역, 실행 중 잠금·취소 버튼, 완료→success·반영 활성, 취소→warning·반영 비활성, CSS 생성). 라벨 뒤 어두운 띠를 발견해 `QLabel { background: transparent }` 추가 후 재확인
- **검증 경계**: 실제 화면·한글 글꼴·실제 파일 대화상자·드래그 입력·브라우저 렌더링은 확인 안 함. 골격 파일은 세션 스크래치패드에만 있음 (노션 코드 블록이 원본)
- **미커밋 파일**: `docs/progress.md`

---

## 2026-09-26 — UI 매뉴얼 (노션 가이드) 작성, 노션 정정 2건, 읽기 권한 허용
> 커밋 기준 `39c2083` · 검증 수준 ⚪ (UI 분석) / 🟠 (골격 offscreen 렌더링) · 테스트 미실행
- **목표**: `main_ui.py`·`ui/` 기준으로 UI 구조·라이브러리·버튼·색상 규칙과 코드 골격을 노션 가이드 카테고리 "UI 매뉴얼"로 정리
- **변경**
	- 노션 "UI 매뉴얼" 신규 (가이드): 창 구성, 반복 패턴, 스레드·잠금·로그 패턴, 색 토큰(surface/text/tone/status), 버튼 variant 8종·scale 4종, 현재 코드 불일치 목록, 골격 3개(`theme.yaml`, `ui/theme.py`, `main_ui.py`)
	- 노션 정정: 카메라를 "D435 시리즈(현재 사용), D405도 지원"으로 (루트·report·design 3.1), 진행 기록 41건 → 40건 (루트)
	- `.claude/settings.json` allow 추가: `cd`, 읽기·검색 도구(Read/Grep/Glob, grep·cat·head·tail·wc·ls·sort·uniq·echo, PowerShell Get-Content·Get-ChildItem·Select-String)
- **검증**: 골격을 camera_ws `.venv` PySide6로 offscreen 렌더링 → variant·disabled·status·scale 색과 크기 적용, START→success·APPLY 활성 확인 (캡처는 세션 스크래치패드, 보존 안 함)
- **검증 경계**: 실제 앱 화면·한글 글꼴에서 본 것 아님. camera_ws UI 코드는 바꾸지 않았다.
- **발견 → 처리 (T8)**: `main_ui.py`의 `os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)`은 git 이력상 한 번도 활성이 아니었다 (07-28 `1d526b8` 주석, 09-13 `1a30d1c` 삭제). 이후 Windows 실기(09-15, 09-22) 정상. 노션 project_harness T8을 🔴 ACTIVE → 🟡 OPEN으로 옮기고 "Linux 미확인, 오류 시 되살림"으로 정정 (사용자 지시). `Readme.md` 81~83행은 아직 옛 설명 (수정 대기)
- **미커밋 파일**: `.claude/settings.json`, `docs/progress.md`

---

## 2026-09-26 — 노션 4문서 개편 (isaac_ros2, whole_body_mirroring)
> 코드 변경 없음 · 검증 수준 ⚪ (문서 작업) · 테스트 미실행
- **변경**: 두 프로젝트를 report / design / process / project_harness로 이관. isaac 요구 39·CPS 38·기능 25·진행 31, wbm 요구 41·CPS 63·기능 42·진행 38. 공통 지식 노트 K10~K19 추가.
- **검증 경계**: 두 프로젝트 모두 저장소와 대조하지 않음 (isaac 저장소는 다른 PC, wbm은 E: 드라이브 없음). 기록 간 모순은 각 process 4.3에 확인 필요로 남김.

---

## 2026-09-26 — 노션 4문서 개편 (camera_marker_calibration)
> 커밋 기준 `39c2083` · 검증 수준 ⚪ (문서 작업, 코드 변경은 CLAUDE.md 문구만) · 테스트 미실행
- **변경**: 노션 원본을 Archive에 백업 → report 신설, Status→process, References→design, Harness→project_harness로 재작성. Development Log는 진행 기록 DB 41행으로 옮기고 원본은 Archive 중간본으로 이동. 요구사항 14, CPS 16, 기능 16행 DB.
- **코드 대조로 찾은 것**: 카메라 위치는 현재 추정함(Trap 1·15 정정), `rby1_sdk` 설치 0.10.0, `.venv` Python 3.11.9(문서는 3.10 고정).
- **CLAUDE.md**: 노션 문서 이름을 새 구조로 수정.

---

## 2026-09-26 — 하네스 적용 (권한·hook·구조 테스트·CLAUDE.md)
> 커밋 기준 `39c2083` · 검증 수준 ⚪ (hook 동작은 확인, 테스트 스위트는 사용자 확인 대기) · 테스트 미실행
- **변경**
	- `.claude/settings.json`: git commit/push·exe 빌드 DENY, 앱·시뮬·테스트·git 이력 변경 ASK, .py 수정 시 테스트 여부를 묻도록 하는 PostToolUse hook
	- `.claude/hooks/ask_tests.py`: 위 hook 스크립트
	- `tests/test_module_boundaries.py`: compute 순수성, comm의 로봇 제어 금지 검사 추가 (`all_imported_modules`로 `import x` 형태도 검사)
	- `CLAUDE.md`: 프로젝트 규칙·계층 대응·안티패턴·래칫 기록 추가 (Graft 안내는 유지)
	- `~/.claude/CLAUDE.md` (전역, 저장소 밖): 공통 하네스 요약
- **검증**: hook 입력 파이프 테스트 통과, 실제 Edit 시 hook 호출 확인. settings.json JSON 검증 통과.
- **검증 경계**: 새 테스트 2개는 아직 실행하지 않았다 (import 현황을 AST로 확인해 통과 예상일 뿐).
- **미커밋 파일**: 위 5개 (전역 파일 제외 4개가 저장소 안)
- **남은 일**
	- [ ] 테스트 스위트 실행해 새 구조 테스트 2개 통과 확인 (사용자 승인 후)
	- [ ] 구조 위반 처리 결정: `*Calibrator.py`·`calibration_core.py`의 `rby1_sdk`·`core.robot` 직접 import, `marker_detection.py` → `calibration_optimizer` import (리팩터링 vs 예외 인정)

---

## 항목 양식
```
## YYYY-MM-DD — 작업 제목
> 커밋 기준 `hash` · 검증 수준 🟢/🟡/🟠/⚪/❌ · 테스트 N/0/0/0
- **목표**
- **원인 → 변경**
- **검증** (명령, 수치, 증거 파일 경로)
- **검증 경계** (이번 결과가 의미하지 않는 것)
- **남은 일**
- **미커밋 파일**
```
