"""Shared environment setup for the per-study experiment runners (issue #3411).

Every study directory under ``scripts/experiments/`` keeps a ``common.py`` that
its stage scripts import as a bare sibling (``import common``) after putting
their own directory first on ``sys.path``.  Those files each used to carry a
copy of the same three things: the ``VTSEARCH_DATA_DIR`` / ``VTSEARCH_MODELS_DIR``
/ ``HF_HOME`` setup, the ``sys.path`` + meta-path surgery that makes ``import
vtscore`` resolve to the study's own worktree (the #2846 fix), and ``timed`` /
``log``.

Those copies are what this module exists to remove, and the worktree fix is the
reason it is worth removing them: when it is wrong it fails **silently**.  A
stage whose ``vtscore`` resolved to the venv's editable install embeds against
whatever code the main checkout happens to be on, with no error and no marker in
the results -- which is exactly how #2846 was found, late.  If the finder
heuristic below ever has to widen (a new packaging tool, a renamed ``.pth``), it
must not be a change that has to land in N files, one of which gets missed.

What stays per-study is everything a study actually decides for itself: which
env vars name its experiment root and results dir, where those default to, and
whether it neutralises the finder at all.  A study that runs from the repo
checkout on one CPU box (``inclusion_knob``) has no wrong worktree to guard
against; a grid study that runs from a dedicated worktree (``calibration``,
``max_patch``, ``mlp_vs_svm``) does.  So the flag is a parameter, not a default
someone has to remember to override.

Each ``common.py``'s public surface is unchanged -- ``REPO``, ``EXP``,
``RESULTS``, ``setup_env()``, ``timed()``, ``log()`` all still resolve through
the study's own module -- so no stage script moved for this.  That matters more
here than it looks: ``scripts/experiments/`` is archival cluster code with no
test and no type coverage (``pyrightconfig.json`` excludes ``scripts/``), and
#3409 declined a much larger reorganisation of the same tree for exactly that
reason.  Keeping the churn inside ``common.py`` is what makes this one cheap.

``umap_params/common.py`` shares nothing with these (it is a dataset roster and
a set of taxonomy builders that happens to carry the same filename), and
``fullmarks/sources/_common.py`` is unrelated.  Both are left alone.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import time
from collections.abc import Iterable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path


# The numeric kernels a closed-loop cell runs on, the same on every x86-64 node (#4110).
#
# The GRID mixes AVX-512 CPUs (Xeon Silver 4114 and Gold 5218R, EPYC 9534) with
# AVX2-only ones (EPYC 7713, Xeon E5-2698 v4), and four libraries on a cell's path
# pick their kernels by CPU: OpenBLAS (#4481), numpy (its AVX-512 float64 exp/log
# and argsort, the anchored EM and tie order), torch's bundled MKL (the head's
# sgemv, the text tower's sgemm) and torch's own ATen kernels (the text tower's
# LayerNorm, softmax and GELU).  The two kinds of node agree to the last bit only
# with all four pinned, and a last-bit difference is enough to move a threshold
# by 1e-6 and, some votes later, a pick (#3840: 3 of 114 cells).
#
# ``MKL_CBWR=COMPATIBLE`` makes CPU GEMMs 2.2-3.5x slower, so this is the cell
# runner's to set, not ``gridenv.sh``'s: a cell's only CPU GEMM is the forward
# pass of one typed query (+0.3 s), while a CPU embedding job would pay it on
# every image.  Disabling numpy's AVX-512 targets costs the anchored EM ~40% on
# an AVX-512 node and nothing on an AVX2 one.
def cpu_kernel_pins() -> dict[str, str]:
    """The four pins, numpy's chosen for the numpy installed here (:func:`numpy_kernel_pin`)."""
    return {
        "OPENBLAS_CORETYPE": "Haswell",
        "NPY_DISABLE_CPU_FEATURES": numpy_kernel_pin(),
        "MKL_CBWR": "COMPATIBLE",
        "ATEN_CPU_CAPABILITY": "avx2",
    }


def numpy_kernel_pin(version: str | None = None) -> str:
    """The ``NPY_DISABLE_CPU_FEATURES`` value that keeps numpy *version* (default: the installed one) on AVX2.

    The value names dispatch targets, and numpy 2.4 renamed its x86 ones (#4930).
    Up to 2.3 they are single extensions, and turning off ``AVX512F`` and
    ``AVX512_SKX`` takes every kernel off AVX-512.  From 2.4 they are
    microarchitecture levels plus two extensions, ``X86_V4`` holding what
    ``AVX512_SKX`` did.  numpy skips a name its build does not dispatch, with an
    ``ImportWarning`` that Python hides, so on 2.4 the old value disables nothing:
    an AVX-512 node keeps its AVX-512 kernels and nothing says so.  Probed on numpy
    2.2.6 to 2.5.4 across the GRID's five CPU models, each value takes its numpy's
    ufuncs to AVX2 and agrees on every node; on the other numpy it does nothing.
    One value naming both sets would work on both, but warn at every import, and
    that warning is the only sign of the next rename.

    Read from the package metadata, not by importing numpy: numpy reads the
    variable when it loads, so the value has to be known before then.
    """
    if version is None:
        version = importlib.metadata.version("numpy")
    major, minor = (int(part) for part in version.split(".")[:2])
    return "X86_V4 AVX512_ICL AVX512_SPR" if (major, minor) >= (2, 4) else "AVX512F AVX512_SKX"


def pin_cpu_kernels(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Default :func:`cpu_kernel_pins` into *environ* (``os.environ``); return what this call set.

    Each pin is a default, so an exported value wins, and ``VTS_KERNEL_PINS=0``
    skips all four (to rerun an arm as it ran before #4110).  x86-64 only: the
    names mean nothing on another architecture, and numpy warns at import about
    a feature its build cannot dispatch.  The libraries read these when they
    load, so a pin set after numpy or torch is imported would silently do
    nothing; on ``os.environ`` this raises instead.
    """
    env = os.environ if environ is None else environ
    if env.get("VTS_KERNEL_PINS", "1") == "0" or platform.machine() not in ("x86_64", "AMD64"):
        return {}
    loaded = [m for m in ("numpy", "torch") if m in sys.modules]
    if loaded and environ is None:
        raise RuntimeError(f"pin_cpu_kernels() must run before {' and '.join(loaded)} is imported")
    set_now = {k: v for k, v in cpu_kernel_pins().items() if k not in env}
    env.update(set_now)
    return set_now


def kernel_report() -> str:
    """One line naming this host's CPU, the kernel pins in force and the numpy kernels they left, for a cell's log."""
    model = "?"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            model = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
    except OSError:
        pass
    pins = " ".join(f"{k}={os.environ.get(k, '-')}" for k in cpu_kernel_pins())
    return f"host={platform.node().split('.')[0]} cpu={model!r} {pins}{_numpy_targets()}"


def _numpy_targets() -> str:
    """`` numpy=<version> numpy_targets=<...>``: the dispatch targets numpy's ufuncs run, once numpy is loaded.

    The pin's effect rather than its value (#4930): a pin numpy ignored still reads
    as set in the environment, but leaves an AVX-512 target (``AVX512_SKX``,
    ``X86_V4``) here on an AVX-512 node.  Empty before numpy loads, so a report
    never imports it.
    """
    np = sys.modules.get("numpy")
    if np is None:
        return ""
    try:
        from numpy.lib import introspect  # noqa: PLC0415  # numpy is loaded already

        current = {d["current"] for sigs in introspect.opt_func_info().values() for d in sigs.values()}
    except (ImportError, AttributeError):  # a numpy without the introspection API
        return f" numpy={np.__version__}"
    above = sorted(t for t in current if not t.startswith("baseline"))
    return f" numpy={np.__version__} numpy_targets={','.join(above) or 'baseline'}"


def neutralise_editable_finder() -> None:
    """Drop the main venv's ``__editable__.vtsearch`` meta-path finder, if present.

    That finder maps ``vtscore``/``vtsearch`` to the checkout the editable
    install was made from, regardless of ``sys.path``; removing it lets the
    study's own worktree (put first on ``sys.path`` by :func:`setup_env`) win.
    """
    keep = []
    for finder in sys.meta_path:
        mod = type(finder).__module__ or ""
        name = f"{mod}.{type(finder).__name__}".lower()
        if "editable" in name and ("vtsearch" in name or "vtscore" in name):
            continue
        keep.append(finder)
    sys.meta_path[:] = keep


def setup_env(
    *,
    repo: Path | str,
    datadir: Path | str,
    models_dir: Path | str,
    results: Path | str | None = None,
    hf_home: Path | str | None = None,
    extra_dirs: Iterable[Path | str] = (),
    neutralise: bool = True,
) -> None:
    """Point vtscore (and optionally HF) at a study's dirs and make *repo* importable.

    Every variable is set with ``setdefault``, so a launcher that exports one
    keeps control of it; the directories created are the ones the env vars
    actually resolved to, not the defaults passed here.

    :param repo: checkout to put first on ``sys.path``.
    :param datadir: value for ``VTSEARCH_DATA_DIR``.
    :param models_dir: value for ``VTSEARCH_MODELS_DIR``.
    :param results: durable output dir; created but not exported.  ``None``
        for a study whose outputs land under a dir it makes elsewhere.
    :param hf_home: value for ``HF_HOME``.  ``None`` leaves HF's own default
        alone, which is what a study wants when it runs on a box whose HF cache
        is already warm and shared.
    :param extra_dirs: further dirs to create.
    :param neutralise: run :func:`neutralise_editable_finder`.  Off for studies
        that run from the repo checkout itself, where there is no second
        checkout for the editable install to resolve to.
    """
    os.environ.setdefault("VTSEARCH_DATA_DIR", str(datadir))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(models_dir))
    if hf_home is not None:
        os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    exported = ["VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR"] + (["HF_HOME"] if hf_home is not None else [])
    for var in exported:
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    for d in ((results,) if results is not None else ()) + tuple(extra_dirs):
        Path(d).mkdir(parents=True, exist_ok=True)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if neutralise:
        neutralise_editable_finder()


@contextmanager
def timed(label: str, sink: dict | None = None) -> Iterator[None]:
    """Record wall-clock seconds for *label* (optionally into *sink*) and print it."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    if sink is not None:
        sink[label] = round(dt, 2)
    print(f"[timing] {label}: {dt:.1f}s", flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)
