"""FSD50K eval prep: embed with CLAP, build the multi-label AudioSet taxonomy.

FSD50K labels each clip against the hierarchical AudioSet ontology (multi-label:
a guitar clip is also Music). We keep two taxonomy tiers for the separability
metric — the 7 top-level AudioSet categories (Music, Human sounds, Animal, …)
and the 200 specific FSD50K classes — each scored one-vs-rest, which needs no
special multi-label handling.

The npz it writes carries a dense ``(N, K)`` label matrix plus which of the K
classes are ontology roots, so the sweep can rebuild the two tiers of masks.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

import common as C


def _load_ontology_roots(onto_path: Path) -> set[str]:
    """AudioSet top-level category mids (nodes that are nobody's child)."""
    nodes = json.load(open(onto_path))
    all_ids = {n["id"] for n in nodes}
    child_ids = {c for n in nodes for c in n.get("child_ids", [])}
    return all_ids - child_ids


def prepare_fsd50k(spec: C.DatasetSpec, limit: int | None = None):
    root = Path(spec.folder)
    gt = root / "FSD50K.ground_truth"
    audio_dir = root / "FSD50K.eval_audio"
    # some archives nest one level (FSD50K.eval_audio/FSD50K.eval_audio)
    if not any(audio_dir.glob("*.wav")) and (audio_dir / "FSD50K.eval_audio").exists():
        audio_dir = audio_dir / "FSD50K.eval_audio"

    # vocabulary: index,name,mid  → the 200 classes we score
    vocab = list(csv.reader(open(gt / "vocabulary.csv")))
    class_names = [r[1] for r in vocab]
    class_mids = [r[2] for r in vocab]
    mid_to_col = {m: i for i, m in enumerate(class_mids)}
    roots = _load_ontology_roots(root / "ontology.json")
    is_root = np.array([m in roots for m in class_mids], dtype=bool)

    # eval.csv: fname,labels,mids
    rows = list(csv.DictReader(open(gt / "eval.csv")))
    # Shuffle deterministically before capping so a size limit stays class-balanced
    # (eval.csv is grouped by label). Default cap keeps the CLAP embed near ~1h.
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    medias, labelsets = {}, {}
    k = 0
    for r in rows:
        wav = audio_dir / f"{r['fname']}.wav"
        if not wav.exists():
            continue
        mids = [m for m in r["mids"].split(",") if m in mid_to_col]
        if not mids:
            continue
        medias[k] = {"id": k, "media_type": "audio", "media_path": str(wav), "media_bytes": None}
        labelsets[k] = mids
        k += 1
    print(f"  fsd50k: {len(medias)} clips, {len(class_names)} vocab classes, {int(is_root.sum())} roots")

    from vtscore.media import get_embedder

    t = time.time()
    vecs = get_embedder("clap").embed_medias(medias)
    ids = sorted(i for i, v in vecs.items() if v is not None)
    mat = np.stack([np.asarray(vecs[i], dtype=np.float32) for i in ids])
    mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    print(f"  clap: embedded {len(ids)} in {time.time() - t:.0f}s")

    # dense (N, K) multi-label matrix aligned to `ids`
    ml = np.zeros((len(ids), len(class_names)), dtype=np.uint8)
    for row_i, mid_i in enumerate(ids):
        for m in labelsets[mid_i]:
            ml[row_i, mid_to_col[m]] = 1
    # a single "primary" label per clip for scatter coloring = its most specific
    # (rarest) class among the applicable ones
    freq = ml.sum(axis=0)
    leaf = []
    for row in ml:
        cols = np.where(row)[0]
        leaf.append(class_names[cols[np.argmin(freq[cols])]] if cols.size else "?")

    C.MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    out = C.MATRIX_DIR / f"{spec.name}__clap.npz"
    np.savez(
        out,
        ids=np.asarray(ids, dtype=np.int64),
        X=mat.astype(np.float32),
        leaf=np.asarray(leaf, dtype=object),
        ml_labels=ml,
        ml_names=np.asarray(class_names, dtype=object),
        ml_isroot=is_root,
    )
    print(f"  saved {out.name}: N={len(ids)} d={mat.shape[1]}")


if __name__ == "__main__":
    import sys

    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    prepare_fsd50k(C.ROSTER_BY_NAME["fsd50k_eval"], limit=lim)
