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
