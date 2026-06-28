"""GPU (cuML / RAPIDS) backend selection for UMAP and k-means.

cuML is an **optional, GPU-only** dependency (part of NVIDIA's RAPIDS suite).
When a usable CUDA device resolves (:func:`vtscore.config.resolve_device`) *and*
cuML is importable, the two heavyweight CPU clustering steps run on the GPU:

- UMAP projection (:mod:`vtscore.projection.umap_projection`) →
  ``cuml.manifold.UMAP`` instead of ``umap-learn``.
- the diversity tree's hierarchical k-means
  (:mod:`vtscore.state.diversity_tree`) → ``cuml.cluster.KMeans`` instead of
  ``sklearn.cluster.KMeans``.

cuML's UMAP/KMeans are deliberately API-compatible with their CPU counterparts,
so the call sites only swap which estimator they construct — not how they use
it.  Both estimators force ``output_type="numpy"`` so downstream code keeps
working with plain ``numpy`` arrays regardless of backend.

**Fit-time failures degrade to CPU, not just construction failures.**  cuML
compiles its cuVS/raft kernels with nvrtc lazily, on the first ``fit`` — so a
broken GPU toolchain (e.g. a CUDA-12 nvrtc compiling CUDA-13 fp8 headers it
can't parse) raises *during the fit*, long after the estimator constructed
cleanly.  The two public entry points here (:func:`umap_fit_transform`,
:func:`kmeans_fit_predict`) therefore wrap the whole construct-and-fit around a
single ``try``: any cuML hiccup, whenever it happens, logs one warning, flips a
process-global kill switch (:func:`cuml_enabled` returns ``False`` from then
on, so the rest of the run skips cuML entirely instead of re-failing per call),
and re-runs the operation on the CPU library.  The result is identical-shape
``numpy`` either way; only the speed differs.

**Output differs from the CPU path.**  cuML is a separate (CUDA-native)
implementation, so even with identical parameters it does not produce
byte-identical coordinates/labels.  That is safe for both consumers because each
computes its result exactly once and then freezes/persists it (the projection is
frozen per dataset; the diversity tree is cached in the dataset pickle), so the
non-reproducibility never surfaces.  The *structure* (neighbourhoods / cluster
topology) is preserved — that's the whole point of these algorithms.

``scripts/install.sh`` installs cuML **by default** on GPU hosts, but as a
separate *best-effort* step (``vts_install_cuml``): it is a multi-gigabyte
RAPIDS stack on a CUDA-major-pinned, separate index (``pypi.nvidia.com``), so a
slow/unreachable index or a torch resolver conflict must not abort an otherwise
good GPU install.  It is deliberately kept out of the main ``requirements/gpu.txt``
pass for the same reason; ``docker/Dockerfile.gpu`` installs it in its own
dedicated (fail-loud) layer instead.  Skip the host-script step with
``VTSEARCH_SKIP_CUML=1``.  Whenever cuML is absent — skipped, failed
to install, or an unsupported platform — everything falls back to the CPU
libraries automatically.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Process-global kill switch, flipped the first time a cuML op fails at runtime
# (typically an nvrtc compile error surfaced from ``fit``).  Once a GPU is known
# to be broken for cuML, there is no point paying the multi-second compile
# failure on every subsequent call — the diversity tree alone fits k-means
# dozens of times — so we disable cuML for the rest of the process and go
# straight to the CPU library.  It only resets on a fresh interpreter.
_cuml_runtime_failed = False


def cuml_enabled() -> bool:
    """True when a usable CUDA device resolves *and* cuML is importable.

    The device check goes through :func:`vtscore.config.resolve_device`, so it
    honours ``VTSEARCH_DEVICE`` and the CUDA kernel smoke-test (a GPU the
    installed wheels can't actually drive resolves to ``"cpu"`` and disables the
    cuML path).  A missing cuML install simply returns ``False`` — the CPU
    libraries handle everything in that case.  Returns ``False`` permanently
    (for this process) once a cuML op has failed at runtime; see
    :func:`_disable_cuml_after_failure`.
    """
    if _cuml_runtime_failed:
        return False
    from vtscore.config import resolve_device

    if not resolve_device().startswith("cuda"):
        return False
    try:
        import cuml  # noqa: F401, PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def _disable_cuml_after_failure(op: str, exc: Exception) -> None:
    """Log a one-time warning and disable cuML for the rest of the process.

    Called from the ``except`` of every cuML entry point so a fit-time blowup
    (e.g. an nvrtc compile error from a mismatched CUDA toolchain) degrades the
    whole run to the CPU library instead of crashing — and does so loudly, so a
    GPU box silently running the slow path is visible in the logs.
    """
    global _cuml_runtime_failed
    first = not _cuml_runtime_failed
    _cuml_runtime_failed = True
    # warning() once (the kill switch ensures we never reach here a second time
    # while emitting); debug-level detail on the off chance someone re-enables.
    log = logger.warning if first else logger.debug
    log(
        "cuML %s failed (%s: %s); falling back to the CPU library and disabling "
        "cuML for the rest of this process. This is usually a CUDA toolchain "
        "mismatch in the environment (e.g. nvrtc compiling fp8 headers it can't "
        "parse); the result is correct but slower.",
        op,
        type(exc).__name__,
        exc,
    )


def umap_fit_transform(
    mat: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int | None,
) -> np.ndarray:
    """Project *mat* to ``n_components``-D with UMAP, preferring cuML on a GPU.

    Constructs *and fits* the reducer here so a cuML failure at either step —
    construction, or the lazy nvrtc kernel compile that only happens inside
    ``fit_transform`` — degrades to the CPU ``umap-learn`` reducer (see
    :func:`_disable_cuml_after_failure`).  Returns a plain ``(N, n_components)``
    ``numpy`` array regardless of backend.  Imports are lazy so merely importing
    this module never pulls in numba's JIT (umap-learn) or the RAPIDS stack.
    """
    if cuml_enabled():
        try:
            from cuml.manifold import UMAP as CuUMAP  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

            reducer = CuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                output_type="numpy",
            )
            return np.asarray(reducer.fit_transform(mat))
        except Exception as exc:  # noqa: BLE001 — any cuML hiccup degrades to CPU
            _disable_cuml_after_failure("UMAP", exc)

    import umap  # noqa: PLC0415

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(mat))


def kmeans_fit_predict(
    vecs: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, float | None]:
    """Cluster *vecs* with k-means, preferring cuML on a GPU.

    Constructs *and fits* the estimator here so a cuML failure at either step
    (construction or the lazy nvrtc kernel compile inside ``fit_predict``)
    degrades to ``sklearn.cluster.KMeans`` (see
    :func:`_disable_cuml_after_failure`).  Returns ``(labels, inertia)`` where
    ``labels`` is a plain ``numpy`` array and ``inertia`` is the fitted
    ``inertia_`` (or ``None`` if the backend didn't report one) — the two things
    the diversity-tree builder needs.
    """
    if cuml_enabled():
        try:
            from cuml.cluster import KMeans as CuKMeans  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

            km: Any = CuKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                output_type="numpy",
            )
            labels = np.asarray(km.fit_predict(vecs))
            return labels, km.inertia_
        except Exception as exc:  # noqa: BLE001 — any cuML hiccup degrades to CPU
            _disable_cuml_after_failure("KMeans", exc)

    from sklearn.cluster import KMeans  # noqa: PLC0415

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,  # pyright: ignore[reportArgumentType]
    )
    labels = np.asarray(km.fit_predict(vecs))
    return labels, km.inertia_
