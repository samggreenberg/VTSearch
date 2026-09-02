"""One reading of the PreToolUse payload contract, shared by every hook.

Two hooks used to disagree about how the harness delivers a tool call.
`require-issue-labels.py` read stdin and fell back to `$TOOL_INPUT`;
`.claude/settings.json` matched Bash commands with an inline
`echo "$TOOL_INPUT" | grep ...`. The harness delivers on **stdin** and sets no
`TOOL_INPUT` variable at all, so the second one grepped an empty string and
never fired -- silently, for its whole life, while looking like a working
safety net (issue #3440).

Two hooks holding two guesses is what let one of them be wrong unnoticed, so
the contract lives here once and both import it. A third hook that reads the
payload some other way is the bug coming back.

**Verified against a live session**, not inferred: a real `Bash` tool call whose
command contained `pytest` left `$TOOL_INPUT` empty and never reached the
installer, while an `mcp__github__issue_write` call in the same session was
parsed and blocked through the stdin path. The env fallback below is kept
anyway -- it costs two lines and covers a harness that changes its mind.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable


def read_payload() -> dict:
    """Parse the PreToolUse payload, preferring stdin and falling back to env."""
    raw = ""
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
    if not raw:
        raw = os.environ.get("TOOL_INPUT", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_arguments(payload: dict, tool_suffix: str, bare_keys: Iterable[str]) -> dict | None:
    """Return the call's arguments, or None if the payload is for another tool.

    Handles both the nested envelope (`{"tool_name", "tool_input"}`) and a bare
    arguments dict, since the two shapes have both been observed. A bare dict
    carries no tool name, so it is identified by its own shape: `bare_keys` are
    the argument names that only this tool's calls would all carry.
    """
    name = payload.get("tool_name") or payload.get("toolName") or ""
    args = payload.get("tool_input") or payload.get("toolInput")
    if isinstance(args, dict):
        return args if name.endswith(tool_suffix) else None
    if all(key in payload for key in bare_keys):
        return payload
    return None
