import json
import re
import sys

d = json.load(sys.stdin)
ti = d.get("tool_input", {})
if not ti.get("file_path", "").endswith(".py"):
    sys.exit(0)
text = ti.get("new_string") or ti.get("content") or ""  # 이번에 AI가 쓴 부분만 검사한다

EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
CHECKS = [
    ("구분선", re.compile(r"^\s*#\s*[=\-*#~_]{4,}")),
    ("단계 나레이션", re.compile(r"^\s*#\s*(step|단계)\s*\d+", re.I)),
    ("끝 표시", re.compile(r"^\s*#\s*end\s+(if|for|while|def|class|try|with)\b", re.I)),
    ("내용 없는 TODO", re.compile(r"^\s*#\s*TODO\b[\s:\-]*(improve|fix|clean ?up|refactor|개선|수정|정리)?\s*\.?$", re.I)),
]
found = []
for n, line in enumerate(text.splitlines(), 1):
    if not line.lstrip().startswith("#"):
        continue
    for name, rx in CHECKS:
        if rx.match(line):
            found.append("%d번째 줄: %s" % (n, name))
    if EMOJI.search(line):
        found.append("%d번째 줄: 장식 이모지" % n)
if found:
    msg = "방금 쓴 주석에 군더더기가 있다. 지우거나 의도·제약만 한 줄로 고친다 (rules/comments.md).\n" + "\n".join(found)
    print(json.dumps({"decision": "block", "reason": msg}))
