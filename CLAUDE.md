# camera_marker_calibration

RB-Y1 헤드 카메라(RealSense D405/D435) 외부 파라미터와 양팔·헤드 관절 홈 오프셋을 마커로 보정하는 PySide6 GUI. 저장소: `https://github.com/JeonSangMin-sw2/camera_ws` (브랜치 `compute_jsm`). 문서(report, detail, harness와 아카이브의 process, cps): 노션 허브의 `rby1_camera_marker_calibration` (https://app.notion.com/p/3ef0f921eed78156bdf4fb51081af1eb). 정본은 옵시디언 볼트 `2_Projects/100_Company_main/101_rby1_camera_marker_calibration/`이다.
우선순위: 사용자의 명시적 지시 > 이 프로젝트의 규칙(harness) > 공통 하네스.

AI의 흔한 코딩 실수를 줄이기 위한 행동 지침이다. 속도보다 신중함에 무게를 둔다. 사소한 작업은 판단해서 줄인다.

## 1. 코딩 전에 생각한다

**가정하지 않는다. 헷갈림을 숨기지 않는다. 트레이드오프를 드러낸다.**

구현 전에:
- 가정을 명시한다. 확신이 없으면 묻는다.
- 해석이 여러 개면 조용히 하나를 고르지 말고 모두 제시한다.
- 더 단순한 방법이 있으면 말한다. 필요하면 반대 의견을 낸다.
- 불분명하면 멈춘다. 무엇이 헷갈리는지 밝히고 묻는다.

보조: `rules/workflow.md` (계획과 승인, 새 기술 도입, 멈춰야 할 때), 스킬 `session-start`

## 2. 단순함이 먼저다

**문제를 푸는 최소한의 코드. 추측으로 넣는 것은 없다.**

- 요청받은 것 이상의 기능을 넣지 않는다.
- 한 번 쓰는 코드에 추상화를 만들지 않는다.
- 요청하지 않은 "유연성"이나 "설정 가능성"을 넣지 않는다.
- 일어날 수 없는 상황의 에러 처리를 넣지 않는다.
- 200줄로 썼는데 50줄로 되면 다시 쓴다.

스스로 묻는다: "시니어 엔지니어가 과하다고 할까?" 그렇다면 줄인다.

보조: `rules/architecture.md` (새 코드가 따를 계층과 의존 규칙. 이 범위를 넘는 추상화는 만들지 않는다), `rules/coding-principles.md`

## 3. 외과적 변경

**꼭 필요한 곳만 건드린다. 내가 만든 어지러움만 치운다.**

기존 코드를 고칠 때:
- 인접한 코드, 주석, 서식을 "개선"하지 않는다.
- 망가지지 않은 것을 리팩터링하지 않는다.
- 내가 다르게 했을 것이라도 기존 스타일에 맞춘다.
- 관련 없는 죽은 코드를 발견하면 언급만 하고 지우지 않는다.

내 변경이 고아를 만들면:
- 내 변경 때문에 안 쓰게 된 import, 변수, 함수는 지운다.
- 원래 있던 죽은 코드는 요청이 없으면 지우지 않는다.

기준: 바뀐 모든 줄이 사용자의 요청으로 곧바로 이어져야 한다.

보조: `rules/coding-principles.md` (원본 보존, 영향 범위 확인, 외부 코드), 스킬 `ratchet`

## 4. 목표 중심 실행

**성공 기준을 정하고, 검증될 때까지 반복한다.**

작업을 검증 가능한 목표로 바꾼다:
- "검증 추가" → "잘못된 입력에 대한 테스트를 쓰고, 통과시킨다"
- "버그 수정" → "버그를 재현하는 테스트를 쓰고, 통과시킨다"
- "X 리팩터링" → "전후에 테스트가 통과하는지 확인한다"

여러 단계 작업은 짧은 계획을 적는다:
```
1. [단계] → 검증: [확인 방법]
2. [단계] → 검증: [확인 방법]
3. [단계] → 검증: [확인 방법]
```
성공 기준이 강하면 혼자 반복할 수 있다. "되게 해줘" 같은 약한 기준은 계속 되묻게 된다.

보조: 스킬 `verify` (G1~G5), `report-result`, 규칙 `verification.md`·`measurement-evidence.md`, 에이전트 `test-runner`·`benchmark-runner`·`log-reader`, hook `g1_after_edit.py`

이 지침이 작동하는 징후: diff의 불필요한 변경이 줄고, 과하게 복잡해서 다시 쓰는 일이 줄고, 질문이 실수 뒤가 아니라 구현 전에 나온다.

---

## 이 프로젝트

- 폴더 구조와 일에 따라 먼저 볼 곳: `.claude/rules/architecture.md`
- 게이트별 실행 명령: harness 문서 1·2장 (노션 https://app.notion.com/p/3ef0f921eed781eb9c0de40bdf0fcb3c, 볼트의 같은 프로젝트 폴더 `harness`)

### 절대 규칙
- `git commit`, `git push`는 하지 않는다.
- 실물 로봇·장비를 움직이는 일은 매번 사용자 허락을 받는다. 명령이 시뮬레이터를 지정했으면 묻지 않는다.
- 원본 영상·인체 좌표·고객 로그를 노션·외부 서비스에 올리지 않는다. 로컬 처리 도구가 외부로 데이터를 보내는 코드를 쓰지 않는다.
- ASK 항목은 매번 사용자 승인 후에만 한다. 한 번 승인받은 것을 다음 행동으로 넓히지 않는다.
- AI는 노션을 스스로 갱신하지 않는다. 사용자가 요청할 때만 `record` 스킬로 기록한다.

### 규칙이 있는 곳
- 주제별 규칙: `.claude/rules/` (workflow, architecture, ui, coding-principles, comments, verification, measurement-evidence)
- 절차: `.claude/skills/` (session-start, report-result, verify, record, ratchet, harness-setup)
- 긴 출력 위임: `.claude/agents/`. 강제 차단: `.claude/settings.json`, `.claude/hooks/`

### 이 프로젝트의 규칙
- Python 3.11.9 (`.venv`). 실기 `192.168.1.40:50051` 모델 "a", 시뮬 `127.0.0.1:50051` 모델 "m".
- 문서와 코드가 충돌하면 **현재 코드가 기준**이며, 판단이 필요하면 사용자에게 묻는다.
- exe 빌드(`build_windows.*`, `build_linux.sh`, PyInstaller)는 사용자가 직접 한다 (2026-09-22).
- `main_ui.py`, `run_windows`, `run_linux`는 실기에 연결될 수 있어 실행할 때마다 hook이 묻는다. `scratch/`의 `*_sim.py`, `arm_sign_test.py`, `camera_swap_test.py`, `randomized_sweep_test.py`는 시뮬(`127.0.0.1`)에만 접속한다.
- 작업 중 상태는 `docs/progress.md`에 갱신한다.

**🔴 ACTIVE — 반드시 지킬 것** (전체 트랩은 harness 문서 4장)
- 보정 수학을 바꾸기 전에 detail 4장(모델 검증), harness 4장(트랩), 아카이브 process의 관련 진행 기록을 먼저 읽는다.
- Step 1 수렴 기준 0.06°를 올려서 통과시키지 않는다 (사용자가 거부한 결정).
- 카메라 내부 파라미터 모드(`calib_intrinsics_mode`)를 바꾸지 않는다. 로봇별 상수·로봇 시리얼을 yaml에 넣지 않는다. 예외: 카메라 시리얼별 내부 파라미터 파일은 허용 (2026-09-30 사용자 결정).
- 테스트·분석 스크립트는 결과 경로를 임시 폴더로 돌린다. `result/`를 덮어쓰지 않는다.
- `config/home_reset_baseline.json`은 Home Offset Reset마다 새로 써진다. `created_at`을 확인하고 같은 리셋 이후 결과끼리만 비교한다.
- Home Offset Apply는 로봇 홈에 실제로 기록된다. 이후 분석에서 오프셋을 다시 더하면 이중 적용이 된다.
- `main_ui.clear_old_plots`는 실행 시작 시 `result_txt`의 .txt를 모두 지운다. 보존할 기록은 먼저 복사한다.
- Step 2 오프라인 재계산 시 `joint_offsets_store`의 부호는 결과 JSON과 반대다.

**안티패턴 (실제 사례)**

| 날짜 | 실수 | 원인 → 지금 규칙 |
|---|---|---|
| 2026-09-09 | 실행 도중 소스를 수정한 증거 실행(`connected-02`) | 증거 실행 중 소스 수정 금지, 해당 실행은 증거 제외 |
| 2026-09-14~15 | AI가 `build_linux.sh`를 직접 재실행 | 빌드는 사용자가 한다 → settings.json DENY |
| 2026-09-15 | runaway 오탐 수정 중 폴백을 되살려 수렴을 통과시키는 안이 나옴 | 사용자와 상의해 엄격 기준 유지, 기준 완화 금지 |
| 2026-09-21 | J0 비교에서 서로 다른 리셋의 baseline을 섞음 | baseline `created_at` 확인 후 같은 리셋끼리만 비교 |
| 2026-10-15 | docker 시뮬 6대를 동시에 띄워 반복 실험이 모두 깨짐 | 시뮬은 한 대씩 순서대로 (harness T39) |
| 2026-10-15~16 | v1.3 브라켓 위치 문제를 "축 사전값"으로 오진, J4 반지름(보정 전 J5에 묶임)을 씀 | 가설은 실측으로 먼저 확인, v1.3 브라켓 위치는 J5·J6 반지름 (T35) |
| 2026-10-14~16 | 죽어 있던 리셋 검사를 고친 뒤 실제 자세로 확인하지 않아 v1.3·v1.2 리셋을 막음, 카메라 광축 회전을 설계값에 묶어 양팔 J1이 기욺 | 되살린 검사는 실제 기록으로 먼저 돌려 봄 (T38), 광축 회전은 Step 1.5가 추정 (T37) |

**래칫 기록**

| 날짜 | 실수·요구 | 반영 계층 | 위치 |
|---|---|---|---|
| 2026-09-26 | AI 직접 커밋 방지 | 권한 DENY + hook | `.claude/settings.json`, `.claude/hooks/block_git.py` |
| 2026-09-26 | exe 빌드 방지 | 권한 DENY | `.claude/settings.json` |
| 2026-09-26 | 테스트는 물어보고 실행, lint는 자동 (2026-10-06에 아래 행으로 대체) | 권한 ASK + PostToolUse hook | 삭제됨 (`ask_tests.py`) |
| 2026-09-26 | 계층 경계 (compute 순수성, comm의 로봇 제어 금지) | 센서(테스트) | `tests/test_module_boundaries.py` |
| 2026-09-28 | 파일 읽기에 권한 창이 뜸 (묶은 명령·sed) → sed 허용, 읽기는 단일 명령/Read·Grep | 권한 ALLOW | `.claude/settings.json` |
| 2026-09-30 | `start/stop_sim_wsl.bat`이 LF 줄바꿈이라 한글 줄과 `chcp 65001`에서 명령이 잘려 실행됨 (`ker`) | 저장소 설정 (체크아웃 시 CRLF 강제) | `.gitattributes` (`*.bat`, `*.cmd` eol=crlf) |
| 2026-10-06 | 공통 하네스 개정 적용: 테스트는 허용, 실기 실행은 hook이 묻고 시뮬은 묻지 않음, lint 자동과 루프 한도·원본 보호·주석 검사 | 권한 + hook 7개 | `.claude/settings.json`, `.claude/hooks/` |

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
