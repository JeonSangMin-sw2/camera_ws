import json
import re
import sys

d = json.load(sys.stdin)
cmd = d.get("tool_input", {}).get("command", "")
# git -C dir push, git -c k=v commit 처럼 옵션이 앞에 끼어도 하위 명령을 찾는다
m = re.search(r"\bgit\s+((?:-[Cc]\s+\S+\s+|--\S+\s+)*)(commit|push)\b", cmd)
if m:
    print("git %s는 사용자가 직접 한다." % m.group(2), file=sys.stderr)
    sys.exit(2)
