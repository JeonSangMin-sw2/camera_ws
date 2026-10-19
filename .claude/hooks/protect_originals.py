import json
import sys

# 실기 결과 백업과 비교용 증거 폴더. 2026-10-08에 archive/ 아래로 모았다 (옛 result_backup, result_d435,
# "result (260921-2차)" 등, result/ 안의 디버깅 폴더). archive/reports와 archive/test_logs는 새 출력이 쌓이는 곳이라 뺀다.
PROTECTED = ["archive/runs/", "archive/result_d435/", "archive/result_copies/", "archive/debug_2026-09/"]

d = json.load(sys.stdin)
path = d.get("tool_input", {}).get("file_path", "").replace("\\", "/")
if any(p in path for p in PROTECTED):
    print("원본 입력 파일은 수정하지 않는다. 변환 결과를 별도 경로에 출력한다: " + path, file=sys.stderr)
    sys.exit(2)
