"""Embed-once stage of the UMAP sweep: build cached (ids, matrix, labels) npz.

For each dataset in the roster, produce one ``.npz`` per embedder holding the
sorted media ids, the ``(N, d)`` embedding matrix, and the per-item leaf label
(plus the biological lineage for iNaturalist). Embedding each (dataset ×
embedder) dominates the experiment's cost, so we snapshot it once here and the
sweep re-fits UMAP cheaply over these matrices (plan §Experiment mechanics:
"embed once, sweep many").

Run on a GPU node inside the VTSearch venv:

    python prepare_dataset.py <dataset_name> [<dataset_name> ...]
    python prepare_dataset.py --all-audio | --all-image | --all

Reuses existing clap pickles verbatim for audio (no GPU needed); re-embeds
every image dataset with clip / siglip / siglip_l.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import common as C


def _save(name: str, embedder: str, ids, matrix, leaf, lineage=None):
    C.MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    out = C.MATRIX_DIR / f"{name}__{embedder}.npz"
    payload = dict(
        ids=np.asarray(ids, dtype=np.int64),
        X=np.asarray(matrix, dtype=np.float32),
        leaf=np.asarray(leaf, dtype=object),
    )
    if lineage is not None:
        payload["lineage"] = np.asarray(lineage, dtype=object)
    np.savez(out, **payload)
    print(f"  saved {out.name}: N={len(ids)} d={matrix.shape[1]}")


def _embed_medias_matrix(medias: dict, embedder_name: str):
    """Run one embedder over a medias dict → (sorted_ids, matrix, kept_ids_set)."""
    from vtscore.media import get_embedder

    emb = get_embedder(embedder_name)
    vecs = emb.embed_medias(medias)  # {id: vector | None}
    ids = sorted(i for i, v in vecs.items() if v is not None)
    mat = np.stack([np.asarray(vecs[i], dtype=np.float32) for i in ids])
    # L2-normalize to match the ingest contract (euclidean == cosine-monotonic).
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    return ids, mat


# --- audio: reuse the existing clap pickle verbatim -------------------------
def prepare_audio_pkl(spec: C.DatasetSpec):
    from vtscore.datasets.loader import load_dataset_from_pickle
    from vtscore.embedding.matrix import get_embedding_matrix
    from vtscore.state.core import DatasetContext

    pkl = _find_pkl(spec.pkl)
    ctx = DatasetContext(spec.name)
    load_dataset_from_pickle(pkl, ctx.medias, thin=True)
    ids, mat = get_embedding_matrix(ctx, None)  # primary slot == clap
    leaf = [ctx.medias[i].get("category", "?") for i in ids]
    # normalize (defensive; ingest already L2-normalizes)
    mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    _save(spec.name, "clap", ids, mat, leaf)


# --- image: re-embed the file list from a pickle with each image embedder ----
def prepare_image_reembed(spec: C.DatasetSpec):
    from vtscore.datasets.loader import load_dataset_from_pickle
    from vtscore.state.core import DatasetContext

    pkl = _find_pkl(spec.pkl)
    ctx = DatasetContext(spec.name)
    load_dataset_from_pickle(pkl, ctx.medias, thin=False)  # need media_path / bytes
    leaf_by_id = {i: m.get("category", "?") for i, m in ctx.medias.items()}
    for embedder in spec.embedder_list():
        t = time.time()
        ids, mat = _embed_medias_matrix(ctx.medias, embedder)
        leaf = [leaf_by_id[i] for i in ids]
        print(f"  {embedder}: embedded {len(ids)} in {time.time()-t:.0f}s")
        _save(spec.name, embedder, ids, mat, leaf)


# --- image folder: Places365 (val.txt labels) --------------------------------
def prepare_places365(spec: C.DatasetSpec):
    from vtscore.datasets.loader import load_places365_metadata
    from vtscore.media.image._demo_sources import PLACES365_CATEGORIES

    target_n = {"places365_s": 5110, "places365_m": 10220, "places365_l": 21170}[spec.name]
    meta = load_places365_metadata(Path(spec.folder), PLACES365_CATEGORIES)  # {fname: {path, category}}
    items = sorted(meta.items())  # deterministic
    rng = np.random.default_rng(0)
    order = rng.permutation(len(items))[:target_n]
    chosen = [items[i] for i in sorted(order)]
    medias = {
        k: {"id": k, "media_type": "image", "media_path": m["path"], "media_bytes": None, "category": m["category"]}
        for k, (_fn, m) in enumerate(chosen)
    }
    leaf_by_id = {k: v["category"] for k, v in medias.items()}
    for embedder in spec.embedder_list():
        t = time.time()
        ids, mat = _embed_medias_matrix(medias, embedder)
        leaf = [leaf_by_id[i] for i in ids]
        print(f"  {embedder}: embedded {len(ids)} in {time.time()-t:.0f}s")
        _save(spec.name, embedder, ids, mat, leaf)


# --- image folder: iNaturalist subset (deep lineage) -------------------------
def prepare_inat(spec: C.DatasetSpec):
    lineage_map = C.load_inat_lineage()
    root = Path(spec.folder)
    files = []  # (path, species_dir)
    for sp_dir in sorted(root.iterdir()):
        if not sp_dir.is_dir():
            continue
        for img in sorted(sp_dir.glob("*.jpg")):
            files.append((str(img), sp_dir.name))
    medias = {
        k: {"id": k, "media_type": "image", "media_path": p, "media_bytes": None, "category": d}
        for k, (p, d) in enumerate(files)
    }
    leaf_by_id = {k: v["category"] for k, v in medias.items()}
    for embedder in spec.embedder_list():
        t = time.time()
        ids, mat = _embed_medias_matrix(medias, embedder)
        leaf = [leaf_by_id[i] for i in ids]
        lineage = [lineage_map.get(leaf_by_id[i], [""] * 7) for i in ids]
        print(f"  {embedder}: embedded {len(ids)} in {time.time()-t:.0f}s")
        _save(spec.name, embedder, ids, mat, leaf, lineage=lineage)


def _find_pkl(basename: str) -> Path:
    for base in [Path("/exp/sgreenberg/projects/VTSearch/data/embeddings"), C.DATA_ROOT]:
        p = base / basename
        if p.exists():
            return p
    raise FileNotFoundError(basename)


DISPATCH = {
    "pkl": prepare_audio_pkl,
    "reembed": prepare_image_reembed,
    "folder": None,  # resolved by taxonomy below
    "fsd50k": None,  # see prepare_fsd50k.py
}


def prepare(name: str, force: bool = False):
    spec = C.ROSTER_BY_NAME[name]
    expected = [C.MATRIX_DIR / f"{name}__{e}.npz" for e in spec.embedder_list()]
    if not force and all(p.exists() for p in expected):
        print(f"[prepare] {name}: all {len(expected)} matrices cached — skip")
        return
    print(f"[prepare] {name} ({spec.media_type}, source={spec.source})")
    if spec.source == "pkl":
        prepare_audio_pkl(spec)
    elif spec.source == "reembed":
        prepare_image_reembed(spec)
    elif spec.source == "folder" and spec.taxonomy == "places365":
        prepare_places365(spec)
    elif spec.source == "folder" and spec.taxonomy == "inat":
        prepare_inat(spec)
    elif spec.source == "fsd50k":
        from prepare_fsd50k import prepare_fsd50k

        prepare_fsd50k(spec, limit=5000)
    else:
        raise ValueError(f"no prepare path for {name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--all-audio" in args:
        names = [d.name for d in C.ROSTER if d.media_type == "audio"]
    elif "--all-image" in args:
        names = [d.name for d in C.ROSTER if d.media_type == "image"]
    elif "--all" in args:
        names = [d.name for d in C.ROSTER]
    else:
        names = args
    for nm in names:
        prepare(nm)
