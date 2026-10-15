"""Tests for ``scripts/check-dockerfiles.py``, the Dockerfile ordering gate.

The gate used to look only for ``python`` on a RUN's first physical line, so a
``pip install`` of the local project slipped through: ``docker/Dockerfile`` and
``docker/Dockerfile.gpu`` installed ``-e .`` (through ``requirements/*.txt``)
before ``vtscore/`` was copied, and the image's editable finder never mapped
``vtscore`` (#4894). Each case below builds a throwaway build context in
``tmp_path`` so the requirements-file lookups resolve against it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("check_dockerfiles", REPO_ROOT / "scripts" / "check-dockerfiles.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_dockerfiles = _load_module()


@pytest.fixture
def context(tmp_path):
    req = tmp_path / "requirements"
    req.mkdir()
    (req / "base.txt").write_text("# forwards to pyproject.toml\n--extra-index-url https://example.invalid\n-e .[dev]\n")
    (req / "plain.txt").write_text("numpy\nrequests  # not the project\n")
    (req / "nested.txt").write_text("-r base.txt\n")
    return tmp_path


def _errors(context: Path, body: str) -> list[str]:
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nWORKDIR /app\n" + body)
    return check_dockerfiles._check(dockerfile, context)


# The dependency layer of docker/Dockerfile as it stood before #4894.
_ISSUE_LAYER = """\
COPY requirements/ requirements/
COPY pyproject.toml ./
COPY vtsearch/ vtsearch/
ARG REQUIREMENTS=base.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \\
    pip install --no-cache-dir \\
        --prefer-binary \\
        -r "requirements/$REQUIREMENTS"
"""


def test_requirements_forwarding_to_the_project_before_vtscore_is_flagged(context):
    errors = _errors(context, _ISSUE_LAYER)
    assert len(errors) == 1
    assert "pip-installs the local project" in errors[0]
    assert ":7:" in errors[0]  # the RUN's first line, not the continuation holding -r


def test_copying_vtscore_first_clears_it(context):
    assert _errors(context, _ISSUE_LAYER.replace("COPY vtsearch/ vtsearch/\n", "COPY vtsearch/ vtsearch/\nCOPY vtscore/ vtscore/\n")) == []


def test_a_full_copy_first_clears_it(context):
    assert _errors(context, "COPY . .\n" + _ISSUE_LAYER) == []


@pytest.mark.parametrize(
    "install",
    [
        "pip install --no-deps -e .",
        "pip install --no-deps --editable=.",
        'pip install -e ".[dev,agpl]"',
        "pip install .",
        "python -m pip install ./",
        "pip install -r requirements/nested.txt",
        "pip install --requirement=requirements/base.txt",
        "pip install -r requirements/${UNSET}",  # no default: matches every file it could name
    ],
)
def test_every_spelling_of_installing_the_project_is_flagged(context, install):
    errors = _errors(context, f"COPY vtsearch/ vtsearch/\nRUN {install}\n")
    assert any("pip-installs the local project" in e for e in errors), errors


@pytest.mark.parametrize(
    "install",
    [
        'pip install --no-cache-dir --no-deps "toponymy==0.5.2"',
        "pip install -r requirements/plain.txt",
        "pip install --upgrade pip setuptools",
        "pip install -r requirements/missing.txt",
    ],
)
def test_installs_that_do_not_touch_the_project_pass(context, install):
    assert _errors(context, f"RUN {install}\n") == []


def test_python_on_a_continuation_line_is_flagged(context):
    errors = _errors(context, "COPY vtsearch/ vtsearch/\nRUN set -e \\\n    && python scripts/preload.py\n")
    assert len(errors) == 1
    assert "executes Python" in errors[0]


def test_comment_lines_inside_a_continued_run_are_dropped(context):
    body = "RUN pip install numpy \\\n# -e .\n    requests\n"
    assert _errors(context, body) == []


def test_each_stage_starts_without_the_previous_stages_copies(context):
    body = "COPY . .\nFROM python:3.12-slim AS second\n" + _ISSUE_LAYER
    assert len(_errors(context, body)) == 1


def test_a_from_copy_does_not_count(context):
    body = "COPY --from=builder /src/vtscore/ vtscore/\n" + _ISSUE_LAYER
    assert len(_errors(context, body)) == 1


def test_the_repos_dockerfiles_pass():
    assert check_dockerfiles.main() == 0
