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
``docmarks/sources/_common.py`` is unrelated.  Both are left alone.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


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
