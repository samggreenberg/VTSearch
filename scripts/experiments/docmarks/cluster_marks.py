"""Turn located marks into identity classes — and be honest that it's a guess.

SPODS and StaVer both ship *where* every logo and stamp is, and neither ships
*which* one it is.  A previous VTSearch study reported "64 logo/stamp classes"
for SPODS with names like ``logo_14``; those identities were derived, not read
off the dataset, and nothing verified them.  Since class identity is the entire
ground truth of an instance-retrieval benchmark, a derived clustering that
nobody checked is not a benchmark — it is a hypothesis with error bars nobody
measured.

So this module does three things in order:

1. crop every mark and describe it (``phash`` by default, ``siglip`` when the
   pile's models are available);
2. single-linkage agglomerate under a distance threshold into candidate classes;
3. emit the material a human needs to confirm or correct the result, and mark
   every derived ``class_id`` with ``provenance="clustered"`` so downstream code
   can tell a verified identity from a guessed one.

Single linkage is chosen deliberately over a centroid method: instances of one
stamp vary continuously with ink coverage and scan quality, so a chain of near
neighbours is the right shape, and the failure mode it does have — two classes
bridged by one ambiguous crop — is exactly what a human reviewing the largest
clusters will spot immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from sources._common import Mark, Page

#: Side length the crop is normalised to before hashing.
PHASH_RESIZE = 32
#: Low-frequency DCT block kept.  64 coefficients -> a 64-bit hash.
PHASH_BLOCK = 8


@dataclass(frozen=True)
class MarkRef:
    """A pointer to one mark inside the page list, plus its crop descriptor."""

    page_index: int
    mark_index: int
    page_id: str
    kind: str
    box: tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------


def _dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via matrix multiply.

    Written out rather than pulled from scipy so the clustering pass has the
    same dependency footprint as the rest of the builder (numpy + Pillow) and
    runs identically on a login node and a compute node.
    """
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[0, :] = basis[0, :] / np.sqrt(2)
    return basis @ block @ basis.T


def phash(image: Any) -> np.ndarray:
    """A 64-bit perceptual hash of *image*, as a boolean vector.

    Scale-invariant by construction (everything is resized to a fixed square),
    which is what we want: the same stamp appears at whatever size the scanner
    and the page layout produced.  Aspect ratio is deliberately discarded here
    and reintroduced as a separate gate in :func:`distance_matrix`, because two
    marks with very different aspect ratios are not the same mark however
    similar their normalised pixels look.
    """
    grey = image.convert("L").resize((PHASH_RESIZE, PHASH_RESIZE))
    arr = np.asarray(grey, dtype=np.float64)
    coeffs = _dct2(arr)[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = coeffs.flatten()
    # Drop the DC term before taking the median: it encodes overall brightness,
    # which on a scan is the paper, not the mark.
    median = np.median(flat[1:])
    return flat > median


def crop_mark(page_image: Any, box: tuple[int, int, int, int], *, pad_frac: float = 0.04) -> Any:
    """Crop *box* out of *page_image* with a little context padding."""
    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(page_image.width, x + w + pad)
    bottom = min(page_image.height, y + h + pad)
    return page_image.crop((left, top, right, bottom))


def describe_marks(
    pages: Sequence[Page],
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
) -> np.ndarray:
    """Descriptor matrix for *refs*, one row per mark.

    ``phash`` returns a boolean matrix compared by normalised Hamming distance;
    ``siglip`` returns L2-normalised float vectors compared by cosine distance.
    """
    from PIL import Image

    if backend == "phash":
        rows = []
        cache: dict[str, Any] = {}
        for ref in refs:
            page = pages[ref.page_index]
            if page.path not in cache:
                cache.clear()  # one page open at a time; pages are large
                cache[page.path] = Image.open(page.path).convert("L")
            rows.append(phash(crop_mark(cache[page.path], ref.box)))
        return np.array(rows, dtype=bool)

    if backend == "siglip":
        from vtscore.media.image.embedder_siglip import SiglipEmbedder  # noqa: PLC0415

        embedder = SiglipEmbedder()
        crops = []
        cache_path: Optional[str] = None
        cache_img: Any = None
        for ref in refs:
            page = pages[ref.page_index]
            if page.path != cache_path:
                cache_path, cache_img = page.path, Image.open(page.path).convert("RGB")
            crops.append(crop_mark(cache_img, ref.box))
        vecs = np.asarray(embedder.embed_images(crops), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    raise ValueError(f"unknown cluster backend {backend!r} (expected 'phash' or 'siglip')")


def distance_matrix(
    desc: np.ndarray,
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
    max_aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Pairwise distances, with mismatched aspect ratios forced apart.

    The aspect gate is what stops a round rubber stamp and a wide letterhead
    banner from merging just because both are dark ink on white paper at 32x32.
    """
    if backend == "phash":
        bits = desc.astype(np.uint8)
        # Hamming distance via matrix algebra: |a XOR b| = a.(1-b) + (1-a).b
        inv = 1 - bits
        dist = (bits @ inv.T + inv @ bits.T).astype(np.float64) / desc.shape[1]
    else:
        dist = 1.0 - (desc @ desc.T)

    aspects = np.array([max(b.box[2], 1) / max(b.box[3], 1) for b in refs], dtype=np.float64)
    ratio = np.maximum(aspects[:, None] / aspects[None, :], aspects[None, :] / aspects[:, None])
    dist = np.where(ratio > max_aspect_ratio, 1.0, dist)

    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def single_linkage(dist: np.ndarray, threshold: float) -> list[int]:
    """Union-find single-linkage clustering.  Returns a label per row."""
    n = dist.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    # Relabel roots to dense 0..k-1 in first-appearance order, so the labelling
    # is a pure function of the distance matrix and not of dict iteration.
    remap: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels.append(remap[root])
    return labels


def assign_class_ids(
    pages: list[Page],
    refs: Sequence[MarkRef],
    labels: Sequence[int],
    *,
    source: str,
) -> dict[str, list[MarkRef]]:
    """Write clustered ``class_id``\\ s back onto the pages' marks.

    Class ids are derived from the cluster's *smallest page id*, not from its
    index, so adding pages to the corpus cannot silently renumber existing
    classes and invalidate a previous run's audit verdicts.
    """
    members: dict[int, list[MarkRef]] = {}
    for ref, label in zip(refs, labels):
        members.setdefault(label, []).append(ref)

    out: dict[str, list[MarkRef]] = {}
    for label, group in members.items():
        anchor = min(group, key=lambda r: (r.page_id, r.mark_index))
        kind = group[0].kind
        class_id = f"{source}/{kind}_{anchor.page_id.split('/')[-1]}_{anchor.mark_index}"
        out[class_id] = group
        for ref in group:
            mark = pages[ref.page_index].marks[ref.mark_index]
            pages[ref.page_index].marks[ref.mark_index] = Mark(
                kind=mark.kind,
                box=mark.box,
                class_id=class_id,
                provenance="clustered",
            )
    return out


def collect_refs(pages: Sequence[Page], *, kinds: Iterable[str], source: str) -> list[MarkRef]:
    """Every unlabelled mark of the given *kinds* from *source*'s pages."""
    wanted = set(kinds)
    refs = []
    for pi, page in enumerate(pages):
        if page.source != source:
            continue
        for mi, mark in enumerate(page.marks):
            if mark.kind in wanted and mark.class_id is None:
                refs.append(MarkRef(pi, mi, page.page_id, mark.kind, mark.box))
    return refs


def cluster_source(
    pages: list[Page],
    source: str,
    *,
    kinds: Iterable[str] = ("logo", "stamp"),
    backend: str = "phash",
    threshold: float = 0.18,
) -> dict[str, Any]:
    """Cluster one source's marks in place.  Returns a summary for the report."""
    refs = collect_refs(pages, kinds=kinds, source=source)
    if not refs:
        return {"source": source, "marks": 0, "classes": 0, "backend": backend, "threshold": threshold}

    desc = describe_marks(pages, refs, backend=backend)
    dist = distance_matrix(desc, refs, backend=backend)
    labels = single_linkage(dist, threshold)
    classes = assign_class_ids(pages, refs, labels, source=source)

    sizes = sorted((len(v) for v in classes.values()), reverse=True)
    return {
        "source": source,
        "marks": len(refs),
        "classes": len(classes),
        "backend": backend,
        "threshold": threshold,
        "largest_clusters": sizes[:10],
        "singletons": sum(1 for s in sizes if s == 1),
    }


def write_cluster_report(summaries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
