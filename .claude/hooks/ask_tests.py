"""PostToolUse hook for .py edits.

G1 lint runs automatically: the edited file is syntax-checked with py_compile.
G2 tests are never run automatically (user rule, 2026-09-26); this hook only
reminds Claude to ask the user first.
"""
import json
import py_compile
import sys


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return
    tool_input = payload.get("tool_input") or {}
    path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if not path.endswith(".py") or ".claude" in path.replace("\\", "/").split("/"):
        return
    try:
        py_compile.compile(path, doraise=True)
        lint = f"G1 lint(py_compile) 통과: {path}."
    except py_compile.PyCompileError as exc:
        lint = f"G1 lint(py_compile) 실패 — 먼저 고칠 것: {exc.msg}"
    message = (
        f"{lint} 테스트는 자동 실행하지 말 것. "
        "이번 작업 단위가 끝나면 사용자에게 테스트(G2) 실행 여부를 먼저 물어볼 것. "
        "명령: .\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests"
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": message}}))  # ASCII escapes: Windows stdout is cp949


if __name__ == "__main__":
    main()
