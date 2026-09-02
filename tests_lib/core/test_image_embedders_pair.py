"""The two all-image-embedders Dockerfiles are one image with two bases.

``docker/Dockerfile.image-embedders`` and ``docker/Dockerfile.image-embedders.gpu``
build the same six-embedder image; the only things that legitimately differ are
the base image (plus the apt layer the CUDA base needs), which PyTorch wheel
index pip resolves against, and the NVIDIA runtime env. Everything else -- the
import canary, the six weight-baking RUN blocks, the settings seed, the non-root
user -- is a verbatim copy, because Docker has no include directive.

That copy had already drifted once: the import canary added in PR #1282 (the
fail-in-seconds guard against a missing ``torchvision``) existed only in the CPU
file, so the GPU build kept discovering the same class of breakage minutes into
a weights download. Nothing compared the two files, so nothing noticed.

These tests make the copy checkable instead of aspirational:

* the region between the ``SHARED BODY`` markers must be byte-identical, and
* both must install the same requirements file, each naming its own wheel index
  (that index is the whole reason the requirements pair was collapsed into one
  file -- see issue #3431 -- so a dropped ``--extra-index-url`` would silently
  put a 2 GB CUDA torch in the CPU image, or a CPU-only torch in the GPU one).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"

CPU = DOCKER / "Dockerfile.image-embedders"
GPU = DOCKER / "Dockerfile.image-embedders.gpu"

BEGIN = "# ===== BEGIN SHARED BODY: keep byte-identical with the CPU/GPU twin ====="
END = "# ===== END SHARED BODY ====="

REQUIREMENTS = "requirements/image-embedders.txt"

# The one axis the shared requirements file deliberately leaves to the caller.
WHEEL_INDEX = {
    CPU.name: "--extra-index-url https://download.pytorch.org/whl/cpu",
    GPU.name: "--extra-index-url https://download.pytorch.org/whl/cu121",
}


def _shared_body(path: Path) -> str:
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == BEGIN]
    ends = [i for i, line in enumerate(lines) if line.strip() == END]
    assert len(starts) == 1 and len(ends) == 1, (
        f"docker/{path.name} must carry exactly one {BEGIN!r} and one {END!r} line; "
        f"found {len(starts)} and {len(ends)}."
    )
    assert starts[0] < ends[0], f"docker/{path.name}: the SHARED BODY end marker precedes its begin marker."
    return "\n".join(lines[starts[0] + 1 : ends[0]])


def test_shared_body_is_byte_identical() -> None:
    cpu, gpu = _shared_body(CPU), _shared_body(GPU)
    assert cpu == gpu, (
        f"docker/{CPU.name} and docker/{GPU.name} have drifted inside their SHARED BODY "
        "region. Everything between the markers must be a verbatim copy: paste the edited "
        "region into the other file. If the change is genuinely GPU- or CPU-specific, it "
        "belongs outside the markers (base image, wheel index, NVIDIA env)."
    )


def test_shared_body_carries_the_import_canary() -> None:
    """The regression that motivated the pairing: a guard only one file had."""
    body = _shared_body(CPU)
    assert "image-embedders import canary OK" in body, (
        "The import-only canary (PR #1282) must live inside the SHARED BODY region so both "
        "images fail in seconds -- not minutes into a weights download -- when a transformers "
        "class used below would raise ImportError."
    )


@pytest.mark.parametrize("dockerfile", [CPU, GPU], ids=[CPU.name, GPU.name])
def test_both_install_the_single_shared_requirements_file(dockerfile: Path) -> None:
    text = dockerfile.read_text()
    assert f"COPY {REQUIREMENTS} requirements/" in text, f"docker/{dockerfile.name} must COPY {REQUIREMENTS}"
    assert f"-r {REQUIREMENTS}" in text, f"docker/{dockerfile.name} must install {REQUIREMENTS}"
    assert "image-embedders-gpu.txt" not in text, (
        f"docker/{dockerfile.name} names requirements/image-embedders-gpu.txt, which was "
        "removed in #3431: the CPU and GPU dependency sets are identical package-for-package, "
        f"so both images install {REQUIREMENTS} and pass their own --extra-index-url."
    )


@pytest.mark.parametrize("dockerfile", [CPU, GPU], ids=[CPU.name, GPU.name])
def test_each_dockerfile_names_its_own_wheel_index(dockerfile: Path) -> None:
    """The shared requirements file is index-agnostic; the index must be supplied here."""
    text = dockerfile.read_text()
    expected = WHEEL_INDEX[dockerfile.name]
    assert expected in text, (
        f"docker/{dockerfile.name} must pass `{expected}` to pip. {REQUIREMENTS} deliberately "
        "carries no --extra-index-url (that is the only thing the CPU and GPU dependency sets "
        "ever differed by), so without it pip resolves torch from PyPI and the image ships the "
        "wrong build."
    )


def test_the_collapsed_requirements_twin_is_gone() -> None:
    assert not (REPO_ROOT / "requirements" / "image-embedders-gpu.txt").exists(), (
        "requirements/image-embedders-gpu.txt is back. It differed from "
        f"{REQUIREMENTS} by exactly one line (the CPU --extra-index-url) while duplicating "
        "~25 package lines and their rationale comments; the index belongs on the pip "
        "invocation in each Dockerfile instead."
    )
