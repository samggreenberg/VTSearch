"""Static syntax checks for Python embedded in our Dockerfiles.

Catches the class of bug fixed in PR #1282: try/except (and other compound
statements) cannot be flattened into a single ``python -c "..."`` line with
semicolons + backslash continuations. Without this test, a SyntaxError in
embedded Python is invisible to ruff, pytest, and CI — it only surfaces
when somebody actually runs ``docker build``, often after minutes of layer
work.

Approach: read each ``Dockerfile*``, join Docker's backslash-continued
lines, pull out every ``python ... -c "..."`` invocation, and run
``compile()`` on the contents. Pure-text work; no Docker daemon needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# Match `python` (optionally with a digit like python3) followed by any
# short flags (-u, -E, etc.), then `-c`, then a double-quoted string.
# Our Dockerfiles always use double quotes outside and single quotes
# inside, so we don't need to handle the single-quoted-outside case.
_PYTHON_DASH_C = re.compile(
    r'\bpython\d?\b(?:\s+-\w+)*\s+-c\s+"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)


def _join_continued_lines(text: str) -> str:
    """Collapse Docker's backslash-newline continuations into single lines."""
    return re.sub(r"\\\n\s*", " ", text)


def _find_python_c_snippets(dockerfile_text: str) -> list[str]:
    """Return every Python source string embedded via ``python -c "..."``."""
    joined = _join_continued_lines(dockerfile_text)
    return [m.group(1) for m in _PYTHON_DASH_C.finditer(joined)]


def _dockerfiles() -> list[Path]:
    return sorted(REPO_ROOT.glob("Dockerfile*"))


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.name)
def test_python_c_snippets_compile(dockerfile: Path) -> None:
    """Every ``python -c`` block in every Dockerfile must parse as Python."""
    snippets = _find_python_c_snippets(dockerfile.read_text())
    for idx, src in enumerate(snippets):
        try:
            compile(src, f"{dockerfile.name}#python-c[{idx}]", "exec")
        except SyntaxError as exc:
            pytest.fail(
                f"{dockerfile.name}: python -c block #{idx} has a SyntaxError "
                f"({exc.msg} at line {exc.lineno}, col {exc.offset}). Compound "
                f"statements like try/except can't be flattened with semicolons "
                f"in a single -c line — use `cmd || echo ...` at the shell level "
                f"or move the logic into a script file.\n"
                f"--- offending snippet ---\n{src}\n---"
            )


# ---------------------------------------------------------------------------
# Self-tests for the parser. These guard against the parser quietly missing a
# block (which would make the parametrized test above silently no-op) and
# against false positives.
# ---------------------------------------------------------------------------


def test_parser_finds_simple_block() -> None:
    text = 'RUN python -c "import os; print(os.name)"\n'
    assert _find_python_c_snippets(text) == ["import os; print(os.name)"]


def test_parser_joins_backslash_continuations() -> None:
    text = (
        'RUN python -u -c "import os; \\\n'
        "from sys import path; \\\n"
        'print(path)"\n'
    )
    snippets = _find_python_c_snippets(text)
    assert len(snippets) == 1
    assert "import os" in snippets[0]
    assert "from sys import path" in snippets[0]
    assert "print(path)" in snippets[0]


def test_parser_catches_inline_try_except_syntax_error() -> None:
    """Regression for the bug fixed in PR #1282."""
    text = (
        'RUN python -u -c "import os; \\\n'
        "try: \\\n"
        "    do_thing(); \\\n"
        "except Exception: \\\n"
        '    pass"\n'
    )
    snippets = _find_python_c_snippets(text)
    assert len(snippets) == 1
    with pytest.raises(SyntaxError):
        compile(snippets[0], "<bad>", "exec")


def test_parser_skips_non_dash_c_python_invocations() -> None:
    """Real script files (``python scripts/foo.py``) are out of scope."""
    text = (
        "RUN python -u scripts/preload_siglip.py\n"
        'RUN python -c "print(1)"\n'
    )
    assert _find_python_c_snippets(text) == ["print(1)"]


def test_repo_actually_has_dockerfiles() -> None:
    """Parametrize collects at runtime — make sure we found anything at all."""
    assert _dockerfiles(), "expected at least one Dockerfile in the repo root"
