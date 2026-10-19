import json
import re
import sys

# GUI와 실행 스크립트는 실기에 연결하면 로봇이 움직인다. 주소를 화면에서 고르므로 명령으로 시뮬레이터를 지정할 수 없어 항상 묻는다
HARDWARE = [r"main_ui\.py", r"run_windows", r"run_linux"]
SIM_MARKERS = []

d = json.load(sys.stdin)
cmd = d.get("tool_input", {}).get("command", "")
if any(re.search(p, cmd) for p in HARDWARE):
    if any(m in cmd for m in SIM_MARKERS):
        decision, reason = "allow", "명령이 시뮬레이터를 지정했으므로 묻지 않는다."
    else:
        decision, reason = "ask", "실물 로봇·장비를 움직일 수 있다. 시뮬레이터가 확인되지 않았으므로 사용자 허락을 받는다."
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}))
