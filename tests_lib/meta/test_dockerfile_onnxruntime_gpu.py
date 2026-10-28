"""The GPU Docker image swaps the CPU onnxruntime for onnxruntime-gpu (issue #5363).

``scripts/install.sh`` does the swap on a CUDA install (#5332), but Docker builds
never run it, so ``docker/Dockerfile.gpu`` repeats the step by hand. Pinned here:
the step is there, the CPU build goes out before the GPU build goes in (both own
the ``onnxruntime`` module), nothing later puts the CPU build back, and the pin
stays on a CUDA 12 build. The all-image-embedders GPU image installs no
onnxruntime at all, so it has nothing to swap until its requirements gain one.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"

# The first onnxruntime-gpu release on PyPI built for CUDA 13 (its ``cuda`` extra
# names the nvidia-*-cu13 libraries); every earlier one is a CUDA 12 build.
FIRST_CUDA13_ORT = (1, 27)


def _load_check_dockerfiles():
    spec = importlib.util.spec_from_file_location("check_dockerfiles", REPO_ROOT / "scripts" / "check-dockerfiles.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_dockerfiles = _load_check_dockerfiles()


def _final_stage_commands(dockerfile: Path) -> list[str]:
    """Every shell command (``&&``-separated segment) of the final stage's RUNs, in order."""
    commands: list[str] = []
    for _, _, text in check_dockerfiles._instructions(dockerfile):
        if text.upper().startswith("FROM "):
            commands = []
        elif text.upper().startswith("RUN "):
            commands += [c.strip() for c in check_dockerfiles._SEPARATORS.split(text[4:])]
    return commands


def _index(commands: list[str], pattern: str) -> int:
    hits = [i for i, c in enumerate(commands) if re.search(pattern, c)]
    assert hits, f"no RUN command matches {pattern!r}"
    return hits[0]


_REQUIREMENTS_PASS = r"\bpip install\b.*\s-r\s"
_UNINSTALL_CPU = r"\bpip uninstall\b.*\sonnxruntime(\s|$)"
_INSTALL_GPU = r"\bpip install\b.*onnxruntime-gpu"


def test_gpu_image_uninstalls_the_cpu_build_before_installing_the_gpu_build():
    commands = _final_stage_commands(DOCKER / "Dockerfile.gpu")
    requirements = _index(commands, _REQUIREMENTS_PASS)
    uninstall = _index(commands, _UNINSTALL_CPU)
    install = _index(commands, _INSTALL_GPU)
    assert requirements < uninstall < install


def test_no_later_layer_puts_the_cpu_build_back():
    commands = _final_stage_commands(DOCKER / "Dockerfile.gpu")
    after = commands[_index(commands, _INSTALL_GPU) + 1 :]
    for command in after:
        if not re.search(r"\bpip install\b", command):
            continue
        assert not re.search(r"(?<![\w-])onnxruntime(?![\w-])", command), command
        # The project, or a requirements file forwarding to it, with its
        # dependencies would reinstall pyproject.toml's CPU onnxruntime.
        assert "--no-deps" in command or not re.search(r"\s(-r|-e)\s|\s\.(\[|\s|$)", command), command


def test_gpu_image_pins_a_cuda12_build():
    """The image is CUDA 12.9; an unpinned or CUDA 13 onnxruntime-gpu cannot load against its libraries."""
    text = (DOCKER / "Dockerfile.gpu").read_text()
    assert re.search(r"^FROM nvidia/cuda:12\.", text, re.MULTILINE), "the base moved off CUDA 12: re-check this pin"
    pins = re.findall(r"onnxruntime-gpu==(\d+)\.(\d+)\.\d+", text)
    assert pins, "onnxruntime-gpu must be pinned with =="
    for major, minor in pins:
        assert (int(major), int(minor)) < FIRST_CUDA13_ORT, f"onnxruntime-gpu {major}.{minor} is a CUDA 13 build"


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        line = re.sub(r"(^|\s)#.*", "", line).strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[\s\[<>=~!;]", line, maxsplit=1)[0].lower())
    return names


def test_image_embedders_gpu_image_has_no_onnxruntime_to_swap():
    """If image-embedders.txt gains onnxruntime (Face in those images), the GPU twin needs the swap too."""
    requirements = _requirement_names(REPO_ROOT / "requirements" / "image-embedders.txt")
    assert not {"onnxruntime", "onnxruntime-gpu"} & requirements
    commands = _final_stage_commands(DOCKER / "Dockerfile.image-embedders.gpu")
    project = [c for c in commands if re.search(r"\bpip install\b.*\s-e\s+\.(\s|$)", c)]
    assert project, "the image no longer installs the project with -e ."
    assert all("--no-deps" in c for c in project), "installing the project with its deps brings the CPU onnxruntime"
