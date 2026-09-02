#!/usr/bin/env python3
"""PreToolUse gate: install the project's dependencies before a command needs them.

A remote container starts with none of VTSearch's stack installed, so a bare
`pytest ...` or `npm run build:prod` fails on an import error rather than on
anything real. `run-tests.sh` guards itself by invoking
`.claude/hooks/ensure-test-deps.sh` directly, and CLAUDE.md's command list
prepends the same call -- this hook is the safety net for every *other* way a
session reaches for the test stack.

It used to live inline in `.claude/settings.json` as::

    if echo "$TOOL_INPUT" | grep -qE 'pytest|...'; then bash ...ensure-test-deps.sh; fi

which never once fired: the harness delivers the tool call on **stdin** and
sets no `TOOL_INPUT` variable, so the grep read an empty string every time.
`run-tests.sh` masked it, which is why a dead safety net kept looking alive
for so long (issue #3440). Reading the payload is now `hook_payload.read_payload`,
shared with `require-issue-labels.py` so the two cannot drift apart again.

The gate never blocks. A failed install exits 1 (a non-blocking hook error the
user sees) rather than 2, so the command still runs and fails on its own terms
-- an install problem should surface as the tool's own error, not as a refusal
to let the tool run.

Note that the match is on the command *text*, so a command that merely mentions
`pytest` in an echo will also trigger the install. That is the intended trade:
a spurious install costs one cold-container wait, a missed one costs a
confusing `ModuleNotFoundError`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_payload import read_payload, tool_arguments  # noqa: E402

TOOL_SUFFIX = "Bash"

# Kept verbatim from the inline matcher this script replaces.
NEEDS_DEPS = re.compile(r"pytest|ng test|ng serve|npm run|npm test")

# Overridable so the tests can assert the gate's *decision* without paying for
# a real dependency install.
INSTALLER = Path(os.environ.get("VTSEARCH_DEPS_INSTALLER") or Path(__file__).resolve().parent / "ensure-test-deps.sh")


def main() -> int:
    args = tool_arguments(read_payload(), TOOL_SUFFIX, bare_keys=("command",))
    if args is None:
        return 0

    if not NEEDS_DEPS.search(str(args.get("command") or "")):
        return 0

    result = subprocess.run(["bash", str(INSTALLER)], check=False)  # noqa: S603,S607  # repo-local installer, no shell
    if result.returncode != 0:
        print(f"ensure-test-deps.sh failed (exit {result.returncode}); running the command anyway.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
