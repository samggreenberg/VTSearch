"""Check Dockerfiles for ordering bugs.

Currently checks: RUN steps that execute Python before vtsearch/ and vtscore/
are both available in the image layer (via COPY vtsearch/, COPY vtscore/, or
a full COPY . .).

Run from the repo root:
    python scripts/check-dockerfiles.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _check(path: Path) -> list[str]:
    errors: list[str] = []
    copied_full = False
    copied_vtsearch = False
    copied_vtscore = False
    in_continuation = False  # previous physical line ended with \

    lines = path.read_text().splitlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        is_continuation = in_continuation
        # Update continuation state for next iteration: this line continues
        # if it ends with \ (but not \\, which is a literal backslash).
        in_continuation = raw.rstrip().endswith("\\") and not raw.rstrip().endswith("\\\\")

        if not line or line.startswith("#"):
            continue

        # Skip lines that are continuations of a previous instruction
        # (e.g. second+ physical lines of a multi-line RUN "...").
        if is_continuation:
            continue

        upper = line.upper()

        # Each FROM starts a new stage; reset layer tracking.
        if upper.startswith("FROM "):
            copied_full = copied_vtsearch = copied_vtscore = False
            continue

        if upper.startswith("COPY "):
            # Skip multi-stage --from= copies (they pull from a prior stage,
            # not the build context, so they don't affect package availability).
            if "--from=" in line.lower():
                continue
            if re.search(r"COPY\s+\.\s+\.", line, re.IGNORECASE):
                copied_full = True
            elif re.search(r"COPY\s+vtsearch/", line, re.IGNORECASE):
                copied_vtsearch = True
            elif re.search(r"COPY\s+vtscore/", line, re.IGNORECASE):
                copied_vtscore = True
            continue

        if upper.startswith("RUN ") and re.search(r"\bpython\b", line, re.IGNORECASE):
            if not copied_full and not (copied_vtsearch and copied_vtscore):
                errors.append(
                    f"{path}:{lineno}: RUN executes Python before vtsearch/ and vtscore/ are both COPY'd\n  {line}"
                )

    return errors


def main() -> int:
    docker_dir = Path(__file__).parent.parent / "docker"
    dockerfiles = sorted(f for f in docker_dir.iterdir() if f.name.startswith("Dockerfile"))

    all_errors: list[str] = []
    for df in dockerfiles:
        all_errors.extend(_check(df))

    if all_errors:
        for err in all_errors:
            print(err)
        return 1

    print(f"Checked {len(dockerfiles)} Dockerfile(s): OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
