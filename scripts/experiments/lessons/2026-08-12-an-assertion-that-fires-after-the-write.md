# 2026-08-12 — an assertion that fires after the write is not a guard (#3129)

**What happened.** A patch script checked `assert "import os" in s` *after*
`p.write_text(s)`. The assertion was correct and caught a real bug — the helper
it inserted used `os.environ` in a module that never imports `os` — but it fired
against an already-modified file, so it reported the problem instead of
preventing it.

**Prevented?** *Advice only.* Validate every precondition before the first
mutation. Patch scripts that edit code in place should be idempotent *and*
fail-closed: check anchors, check imports, then write.
