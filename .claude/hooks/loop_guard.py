import hashlib
import json
import os
import sys

IGNORE = ("git status", "git diff", "git log", "ls", "pwd", "cat ", "echo ", "Get-Content", "Get-ChildItem", "Select-String")  # 조회 명령은 반복해도 된다

d = json.load(sys.stdin)
state_dir = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), ".claude", "state")
os.makedirs(state_dir, exist_ok=True)
path = os.path.join(state_dir, "last_command.txt")

if d.get("tool_name") in ("Edit", "Write"):
    if os.path.exists(path):
        os.remove(path)  # 파일을 고쳤으면 같은 명령을 다시 돌려도 된다
    sys.exit(0)

cmd = d.get("tool_input", {}).get("command", "").strip()
if cmd.startswith(IGNORE):
    sys.exit(0)
h = hashlib.sha1(cmd.encode("utf-8")).hexdigest()
if os.path.exists(path) and open(path).read() == h:
    print("고친 것 없이 같은 명령을 다시 실행하려 한다. 멈추고 시도한 것 / 결과 / 남은 가설을 보고한 뒤 사용자 판단을 받는다.", file=sys.stderr)
    sys.exit(2)
open(path, "w").write(h)
