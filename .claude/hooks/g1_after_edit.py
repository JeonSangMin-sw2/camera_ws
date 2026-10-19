import hashlib
import json
import os
import subprocess
import sys

d = json.load(sys.stdin)
if not d.get("tool_input", {}).get("file_path", "").endswith(".py"):
    sys.exit(0)

G1 = [sys.executable, "-m", "py_compile", d["tool_input"]["file_path"]]  # 프로젝트 harness 2장의 G1: 수정한 .py 문법 검사
state_dir = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), ".claude", "state")
os.makedirs(state_dir, exist_ok=True)
state = os.path.join(state_dir, "g1_fail.json")

r = subprocess.run(G1, capture_output=True, text=True)
out = r.stdout + r.stderr
if r.returncode == 0:
    if os.path.exists(state):
        os.remove(state)
    msg = "G1 통과. 작업 단위가 끝나면 G2를 test-runner로 실행한다."
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
    sys.exit(0)

h = hashlib.sha1(out.encode("utf-8")).hexdigest()
s = json.load(open(state)) if os.path.exists(state) else {"count": 0, "last": ""}
same = s["last"] == h
count = s["count"] + 1
if count >= 3 or same:
    os.remove(state) if os.path.exists(state) else None
    why = "같은 문제를 3번 고쳤는데도 G1이 실패했다" if count >= 3 else "같은 에러가 2번 반복됐다"
    print(json.dumps({"continue": False, "stopReason": "루프 한도: " + why + ". 시도한 것 / 결과 / 남은 가설을 보고하고 사용자 판단을 받는다.\n" + out[-1500:]}))
else:
    json.dump({"count": count, "last": h}, open(state, "w"))
    print(json.dumps({"decision": "block", "reason": "G1 실패. 다른 작업 전에 먼저 고친다.\n" + out[-2000:]}))
