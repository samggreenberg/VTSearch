#!/usr/bin/env python3
"""Pick the right PyTorch CUDA wheel tag for the GPU(s) on this host.

``scripts/install-gpu.sh`` calls this when invoked without an explicit tag, so
users don't have to know their GPU's compute capability (or that the *newest*
CUDA tag is the wrong choice for an old GPU). It shells out to ``nvidia-smi``,
reads each GPU's compute capability and the driver's max CUDA version, and
prints a ``cuXYZ`` tag (e.g. ``cu124``) to **stdout**. A human-readable
explanation goes to **stderr** so the caller can capture just the tag.

Exit status is ``0`` with a tag on stdout when detection succeeds, ``1`` with
nothing on stdout when it can't tell (no ``nvidia-smi``, an unparseable
response, or a GPU fleet no single wheel can cover) so the shell falls back to
its built-in default.

Stdlib only, on purpose: this runs *before* dependencies are installed, so it
can't import torch or anything from the project.

The selection logic (:func:`select_cuda_tag`) is pure and unit-tested in
``tests_lib/core/test_detect_cuda_tag.py``; the ``nvidia-smi`` plumbing around
it is the thin, side-effecting shell.
"""

from __future__ import annotations

import re
import subprocess  # noqa: S404 - we invoke a fixed local binary (nvidia-smi), never user input
import sys

# Each PyTorch CUDA wheel ships kernel images for a fixed, bounded set of GPU
# architectures, expressed here as an inclusive ``(min_cc, max_cc)`` compute-
# capability range. There is a FLOOR and a CEILING: newer wheels add new
# architectures but DROP the oldest ones, which is exactly why "just use the
# newest tag" breaks old GPUs. cu128 dropped Volta (sm_70), so a V100 (cc 7.0)
# must use cu124 or older. The ``cuda_ver`` is the toolkit version the wheel
# needs the driver to support; we use it to step down on hosts with old drivers.
#
#               tag       cuda_ver  min_cc   max_cc
_CANDIDATES = [
    ("cu118", (11, 8), (3, 7), (9, 0)),
    ("cu121", (12, 1), (5, 0), (9, 0)),
    ("cu124", (12, 4), (5, 0), (9, 0)),
    ("cu128", (12, 8), (7, 5), (12, 0)),
]

# The anchor tag: a safe default that spans Volta through Hopper. We prefer it
# whenever it's valid and only deviate when the GPU is too new for it
# (Blackwell -> cu128) or the driver is too old for it (-> cu121/cu118).
DEFAULT_TAG = "cu124"


def select_cuda_tag(
    compute_caps: list[tuple[int, int]],
    driver_cuda: tuple[int, int] | None = None,
) -> str | None:
    """Choose a ``cuXYZ`` tag covering every GPU in *compute_caps*.

    *compute_caps* is the list of ``(major, minor)`` compute capabilities of
    the visible GPUs; *driver_cuda* is the max CUDA version the installed driver
    supports (``(major, minor)`` from ``nvidia-smi``), or ``None`` if unknown.

    A single wheel must cover the *whole* fleet, so its arch range has to
    include both the oldest and the newest GPU present. Among the wheels that
    qualify (and that the driver is new enough to run, when known), we prefer
    :data:`DEFAULT_TAG`; failing that, the newest qualifying wheel. Returns
    ``None`` when no single wheel can cover the fleet (e.g. a Volta card and a
    Blackwell card in the same box) so the caller can fall back and warn.
    """
    if not compute_caps:
        return None
    lo, hi = min(compute_caps), max(compute_caps)
    valid = [c for c in _CANDIDATES if c[2] <= lo and c[3] >= hi]
    if driver_cuda is not None:
        gated = [c for c in valid if c[1] <= driver_cuda]
        # Only apply the driver ceiling if it leaves something runnable; if even
        # the oldest covering wheel out-runs the driver, the driver is simply
        # too old and the caller surfaces that rather than picking nothing.
        if gated:
            valid = gated
    if not valid:
        return None
    for tag, *_ in valid:
        if tag == DEFAULT_TAG:
            return DEFAULT_TAG
    return max(valid, key=lambda c: c[1])[0]


def parse_compute_caps(query_output: str) -> list[tuple[int, int]]:
    """Parse ``nvidia-smi --query-gpu=compute_cap`` CSV output into ``(M, m)``.

    One GPU per line, e.g. ``7.0``. Lines that aren't a numeric ``X.Y`` (blank
    lines, ``[N/A]`` from drivers too old to report it) are skipped.
    """
    caps: list[tuple[int, int]] = []
    for line in query_output.splitlines():
        m = re.match(r"\s*(\d+)\.(\d+)\s*$", line)
        if m:
            caps.append((int(m.group(1)), int(m.group(2))))
    return caps


def parse_driver_cuda(smi_output: str) -> tuple[int, int] | None:
    """Pull the driver's max CUDA version from plain ``nvidia-smi`` output.

    The banner reads ``... CUDA Version: 12.4 ...``. Drivers older than ~R450
    omit it; returns ``None`` in that case.
    """
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_output)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _run(args: list[str]) -> str | None:
    """Run a command, returning stdout, or ``None`` if it can't be run."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            args, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def detect() -> tuple[str | None, str]:
    """Detect the recommended tag. Returns ``(tag_or_None, explanation)``."""
    caps_out = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if caps_out is None:
        return None, "nvidia-smi not found or failed; cannot auto-detect the GPU."
    caps = parse_compute_caps(caps_out)
    if not caps:
        return None, "nvidia-smi reported no usable compute capability for any GPU."

    driver_cuda = parse_driver_cuda(_run(["nvidia-smi"]) or "")
    tag = select_cuda_tag(caps, driver_cuda)

    cc_str = ", ".join(f"{a}.{b}" for a, b in caps)
    drv_str = f"{driver_cuda[0]}.{driver_cuda[1]}" if driver_cuda else "unknown"
    if tag is None:
        return None, (
            f"Detected GPU compute capabilities [{cc_str}] (driver CUDA {drv_str}), "
            "but no single PyTorch CUDA wheel covers them all. Install per-host "
            "manually with an explicit tag."
        )
    return tag, (
        f"Detected GPU compute capabilities [{cc_str}], driver CUDA {drv_str} -> selecting torch CUDA tag '{tag}'."
    )


def main() -> int:
    tag, explanation = detect()
    print(explanation, file=sys.stderr)
    if tag is None:
        return 1
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
