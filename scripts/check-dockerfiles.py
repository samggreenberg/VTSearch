"""Check Dockerfiles for ordering bugs.

Both checks ask whether vtsearch/ and vtscore/ are both in the image layer yet
(via COPY vtsearch/, COPY vtscore/, or a full COPY . .) when a RUN step needs
them:

- a RUN that executes Python, which would fail to import the missing package.
  Any ``python``, ``python3`` or ``python3.12`` command counts, by any path
  (``/opt/venv/bin/python3``) and inside ``sh -c``. Two command segments are
  exempt because they cannot need the project: a package-manager install that
  merely names ``python3`` packages, and ``python3 -m venv``;
- a RUN that pip-installs the local project ``.``, editable or not, directly or
  through a requirements file that forwards to it. setuptools maps only the
  top-level packages it finds at install time, so a package that first arrives
  with a later COPY . . imports only from the image's WORKDIR, where it sits on
  sys.path as a plain directory, and not from a script run anywhere else
  (#4894).

Run from the repo root:
    python scripts/check-dockerfiles.py
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# `pip install`, `pip3 install`, `python -m pip install`, ...
_PIP_INSTALL = re.compile(r"\bpip[0-9.]*\s+install\b", re.IGNORECASE)

# The local project as a pip target: `.`, `./`, `.[dev,agpl]`.
_PROJECT = re.compile(r"\./?(\[[^\]]*\])?")

# A Python interpreter as a command word: `python`, `python3`, `python3.12`,
# `/opt/venv/bin/python3`. `\b` would miss `python3` (no boundary between
# `n` and `3`); the lookarounds instead reject package and variable names such
# as `python3-venv`, `python3.12-venv`, `python-dotenv` and `PYTHONUNBUFFERED`.
_PYTHON = re.compile(r"(?<![\w.-])python[0-9.]*(?![\w.-])", re.IGNORECASE)

# Command segments that name Python without running project code: a system
# package install (`apt-get install -y python3 python3-venv`) and venv creation.
_PYTHON_EXEMPT = re.compile(
    r"\b(?:apt-get|apt|yum|dnf)\b.*\binstall\b|\bapk\b.*\badd\b|-m\s+venv\b",
    re.IGNORECASE,
)

# Shell command separators. Splitting ignores quoting, so `sh -c '...'` bodies
# are split too, which is what both checks want.
_SEPARATORS = re.compile(r"&&|\|\||;|\|")


def _instructions(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(lineno, first_line, text)`` for each instruction in *path*.

    *text* joins Docker's backslash continuations into one line, dropping the
    comment lines Docker itself drops from inside a continued instruction.
    """
    start = 0
    first = ""
    parts: list[str] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not parts and (not line or line.startswith("#")):
            continue
        if parts and line.startswith("#"):
            continue
        if not parts:
            start, first = lineno, line
        # A trailing \ continues the instruction (but not \\, a literal backslash).
        continued = line.endswith("\\") and not line.endswith("\\\\")
        parts.append(line[:-1].strip() if continued else line)
        if not continued:
            yield start, first, " ".join(p for p in parts if p)
            parts = []
    if parts:
        yield start, first, " ".join(p for p in parts if p)


def _expand(value: str, args: dict[str, str]) -> str:
    """Substitute ``$NAME`` / ``${NAME}`` with the stage's ARG defaults; leave unknowns."""
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: args.get(m.group(1) or m.group(2), m.group(0)), value)


def _requirements_install_project(pattern: str, base: Path, seen: set[Path]) -> bool:
    """True when a requirements file matched by *pattern* (relative to *base*) installs ``.``.

    A variable left unresolved (an ARG with no default) matches every file it
    could name, so the check errs toward flagging.
    """
    pattern = re.sub(r"\$\{\w+\}|\$\w+", "*", pattern)
    for req in sorted(base.glob(pattern)) if "*" in pattern else [base / pattern]:
        req = req.resolve()
        if req in seen or not req.is_file():
            continue
        seen.add(req)
        for line in req.read_text().splitlines():
            line = re.sub(r"(^|\s)#.*", "", line).strip()
            if line and _installs_project(line.split(), {}, req.parent, seen):
                return True
    return False


def _installs_project(tokens: list[str], args: dict[str, str], base: Path, seen: set[Path]) -> bool:
    """True when pip *tokens* (the arguments after ``install``) install the local project."""
    it = iter(tokens)
    for tok in it:
        if tok in ("-r", "--requirement"):
            if _requirements_install_project(_expand(next(it, ""), args), base, seen):
                return True
            continue
        if tok.startswith("--requirement=") or (tok.startswith("-r") and len(tok) > 2):
            value = tok.split("=", 1)[1] if tok.startswith("--") else tok[2:]
            if _requirements_install_project(_expand(value, args), base, seen):
                return True
            continue
        if tok in ("-e", "--editable"):
            tok = next(it, "")
        elif tok.startswith("--editable="):
            tok = tok.split("=", 1)[1]
        elif tok.startswith("-e") and len(tok) > 2:
            tok = tok[2:]
        if _PROJECT.fullmatch(tok):
            return True
    return False


def _run_executes_python(text: str) -> bool:
    """True when any command segment of the RUN *text* runs a Python interpreter."""
    return any(_PYTHON.search(command) and not _PYTHON_EXEMPT.search(command) for command in _SEPARATORS.split(text))


def _run_installs_project(text: str, args: dict[str, str], context: Path) -> bool:
    """True when any ``pip install`` in the RUN *text* installs the local project."""
    for command in _SEPARATORS.split(text):
        match = _PIP_INSTALL.search(command)
        if not match:
            continue
        rest = command[match.end() :]
        try:
            tokens = shlex.split(rest)
        except ValueError:
            tokens = rest.split()
        if _installs_project(tokens, args, context, set()):
            return True
    return False


def _check(path: Path, context: Path = _REPO) -> list[str]:
    """Ordering errors in the Dockerfile at *path*, whose build context is *context*."""
    errors: list[str] = []
    copied_full = False
    copied_vtsearch = False
    copied_vtscore = False
    args: dict[str, str] = {}

    for lineno, first, text in _instructions(path):
        upper = text.upper()

        # Each FROM starts a new stage; reset layer tracking (and the stage's ARGs).
        if upper.startswith("FROM "):
            copied_full = copied_vtsearch = copied_vtscore = False
            args = {}
            continue

        if upper.startswith("ARG "):
            match = re.fullmatch(r"ARG\s+(\w+)=(.*)", text, re.IGNORECASE)
            if match:
                args[match.group(1)] = match.group(2).strip().strip("\"'")
            continue

        if upper.startswith("COPY "):
            # Skip multi-stage --from= copies (they pull from a prior stage,
            # not the build context, so they don't affect package availability).
            if "--from=" in text.lower():
                continue
            if re.search(r"COPY\s+\.\s+\.", text, re.IGNORECASE):
                copied_full = True
            elif re.search(r"COPY\s+vtsearch/", text, re.IGNORECASE):
                copied_vtsearch = True
            elif re.search(r"COPY\s+vtscore/", text, re.IGNORECASE):
                copied_vtscore = True
            continue

        if not upper.startswith("RUN ") or copied_full or (copied_vtsearch and copied_vtscore):
            continue

        if _run_executes_python(text):
            errors.append(
                f"{path}:{lineno}: RUN executes Python before vtsearch/ and vtscore/ are both COPY'd\n  {first}"
            )
        if _run_installs_project(text, args, context):
            errors.append(
                f"{path}:{lineno}: RUN pip-installs the local project before vtsearch/ and vtscore/ are both "
                f"COPY'd; setuptools maps only the packages present at install time\n  {first}"
            )

    return errors


def main() -> int:
    docker_dir = _REPO / "docker"
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
