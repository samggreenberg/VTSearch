"""Tests for ``scripts/gates/check-dockerfiles.py``, the Dockerfile ordering gate.

The gate used to look only for ``python`` on a RUN's first physical line, so a
``pip install`` of the local project slipped through: ``docker/Dockerfile`` and
``docker/Dockerfile.gpu`` installed ``-e .`` (through ``requirements/*.txt``)
before ``vtscore/`` was copied, and the image's editable finder never mapped
``vtscore`` (#4894). Its Python test, ``\bpython\b``, also missed ``python3`` (#5008).
Each case below builds a throwaway build context in ``tmp_path`` so the
requirements-file lookups resolve against it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_dockerfiles", REPO_ROOT / "scripts" / "gates" / "check-dockerfiles.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_dockerfiles = _load_module()


@pytest.fixture
def context(tmp_path):
    req = tmp_path / "requirements"
    req.mkdir()
    (req / "base.txt").write_text(
        "# forwards to pyproject.toml\n--extra-index-url https://example.invalid\n-e .[dev]\n"
    )
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
    fixed = _ISSUE_LAYER.replace("COPY vtsearch/ vtsearch/\n", "COPY vtsearch/ vtsearch/\nCOPY vtscore/ vtscore/\n")
    assert _errors(context, fixed) == []


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


@pytest.mark.parametrize(
    "command",
    [
        "python3 script.py",
        "python3.12 script.py",
        "/opt/venv/bin/python3 x.py",
        "sh -c 'python3 x.py'",
        "python3 -m venv /opt/venv && python3 x.py",  # the exemption covers its own segment only
        "apt-get install -y python3; python3 x.py",
    ],
)
def test_python3_before_the_copies_is_flagged(context, command):
    # `\bpython\b` missed all of these: there is no word boundary between `n` and `3`.
    errors = _errors(context, f"COPY vtsearch/ vtsearch/\nRUN {command}\n")
    assert len(errors) == 1
    assert "executes Python" in errors[0]


@pytest.mark.parametrize(
    "command",
    [
        "python3 -m venv /opt/venv",
        "apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgl1",
        "apt-get install -y python3.12-venv",
        "apk add --no-cache python3",
        "pip install python-dotenv",  # a false positive of the old `\bpython\b`
        "echo $PYTHONUNBUFFERED",
    ],
)
def test_naming_python_without_running_it_passes(context, command):
    assert _errors(context, f"RUN {command}\n") == []


def test_python3_after_both_copies_passes(context):
    body = "COPY vtsearch/ vtsearch/\nCOPY vtscore/ vtscore/\nRUN python3 script.py\n"
    assert _errors(context, body) == []


def test_the_gpu_base_layer_passes(context):
    # The system-Python block both GPU Dockerfiles run before any COPY.
    body = (
        "RUN apt-get update && \\\n"
        "    apt-get install -y --no-install-recommends \\\n"
        "        python3 \\\n"
        "        python3-venv \\\n"
        "        libgl1 && \\\n"
        "    rm -rf /var/lib/apt/lists/* && \\\n"
        "    python3 -m venv /opt/venv && \\\n"
        "    userdel --remove ubuntu\n"
    )
    assert _errors(context, body) == []


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


def test_every_test_tree_is_kept_out_of_the_images():
    # Every Dockerfile ends in `COPY . .`, so a test tree missing from
    # .dockerignore is baked into each image (tests_lib/ was, at 6 MB).
    trees = sorted(p.name for p in REPO_ROOT.iterdir() if p.is_dir() and p.name.startswith("tests"))
    assert {"tests", "tests_lib", "tests_shared"} <= set(trees)
    ignored = {line.strip().strip("/") for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()}
    assert [t for t in trees if t not in ignored] == []
