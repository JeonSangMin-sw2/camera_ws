import json
import re
import sys

d = json.load(sys.stdin)
cmd = d.get("tool_input", {}).get("command", "")
SOURCE = re.compile(r"\.(py|c|cc|cpp|h|hpp|js|ts|css|html|yaml|yml|toml)\b")
PATTERNS = [
    r"\bsed\s+(-[A-Za-z]*i\b|--in-place)",          # sed -i
    r"\bperl\s+-[A-Za-z]*i",                        # perl -pi
    r"\bawk\b.*-i\s+inplace",                       # awk -i inplace
    r"(\.replace\(|re\.sub\().*(write_text|\.write\()",  # 파이썬 문자열 치환 후 쓰기
    r"-replace.*(Set-Content|Out-File)",            # PowerShell
]
if SOURCE.search(cmd) and any(re.search(p, cmd, re.S) for p in PATTERNS):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": "소스를 문자열 치환 스크립트로 고치려 한다. 기능은 소스에 직접 쓴다. 일괄 변경이 정말 필요한지 사용자에게 확인한다.",
    }}))
