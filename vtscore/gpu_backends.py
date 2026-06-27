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

from typing import Any


def cuml_enabled() -> bool:
    """True when a usable CUDA device resolves *and* cuML is importable.

    The device check goes through :func:`vtscore.config.resolve_device`, so it
    honours ``VTSEARCH_DEVICE`` and the CUDA kernel smoke-test (a GPU the
    installed wheels can't actually drive resolves to ``"cpu"`` and disables the
    cuML path).  A missing cuML install simply returns ``False`` — the CPU
    libraries handle everything in that case.
    """
    from vtscore.config import resolve_device

    if not resolve_device().startswith("cuda"):
        return False
    try:
        import cuml  # noqa: F401, PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def make_umap(
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int | None,
) -> Any:
    """Construct a UMAP reducer, preferring cuML on a usable GPU.

    Falls back to ``umap-learn`` when cuML is unavailable or its construction
    raises.  Imports are lazy so merely importing this module never pulls in
    numba's JIT (umap-learn) or the RAPIDS stack.
    """
    if cuml_enabled():
        try:
            from cuml.manifold import UMAP as CuUMAP  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

            return CuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                output_type="numpy",
            )
        except Exception:  # noqa: BLE001 — any cuML hiccup degrades to CPU
            pass

    import umap  # noqa: PLC0415

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )


def make_kmeans(*, n_clusters: int, random_state: int, n_init: int) -> Any:
    """Construct a k-means estimator, preferring cuML on a usable GPU.

    Falls back to ``sklearn.cluster.KMeans`` when cuML is unavailable or its
    construction raises.  Both estimators expose ``fit_predict`` and an
    ``inertia_`` attribute, which is all the diversity-tree builder uses.
    """
    if cuml_enabled():
        try:
            from cuml.cluster import KMeans as CuKMeans  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

            return CuKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                output_type="numpy",
            )
        except Exception:  # noqa: BLE001 — any cuML hiccup degrades to CPU
            pass

    from sklearn.cluster import KMeans  # noqa: PLC0415

    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,  # pyright: ignore[reportArgumentType]
    )
