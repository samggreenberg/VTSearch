#!/usr/bin/env python
"""What a face embedder does with each person's photos, with no session in the way (#4780).

#4762's pronoun gap is already in the full-label ceiling, so part of it sits in the embedding of the
photos before any vote is cast. This reads a face cell and, for each of the review's people and each
of their subject photos, asks how the embedder ranks the person's other photos among every face in
the cell:

* ``loo_ap`` - the photo as the query, every other face ranked by cosine, AP of the person's own;
* ``genuine_mean`` / ``genuine_min`` - cosine to the person's other photos;
* ``impostor_top1`` / ``impostor_top10`` - the closest face that is not theirs, and the mean of the
  ten closest; ``impostor_top1_pronoun`` - whose that face is (``nobody`` for a face the loader kept
  as no one's);
* ``margin`` - ``genuine_mean - impostor_top1``; ``impostors_above`` - how many other people's faces
  score above the photo's median genuine cosine.

and, for matching photo pairs on how they were taken, the cosine of every **genuine pair** (two
subject photos of one person, ``static_pairs.csv``; ``id_a``/``id_b`` are photo ids, so they join
``face_gap_4780_covariates.py``'s ``pid``).

Each ``--cell NAME=PATH`` is read on its own, so two embedders on the same faces (FaceNet, ArcFace)
pair photo for photo. Writes ``static_photos.csv`` (one row per cell x subject photo; names people,
so owner-only), ``static_pairs.csv`` and ``static_people.csv`` (per person, the mean over their photos).

    python face_gap_4780_static.py --cell facenet=<cells>/fhibe_faces_roll_1024__face.pkl \\
        --cell arcface=<cells-4780>/fhibe_faces_roll_1024__arcface_r50.pkl \\
        --identities <ids.txt> --strata <person_strata.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_cell(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids, unit vectors, category)`` of a face cell, sorted by id."""
    medias = pickle.loads(path.read_bytes())  # noqa: S301  # our own pile cell
    ids = np.asarray(sorted(medias), dtype=np.int64)
    vecs = np.stack([np.asarray(medias[i]["embeddings"]["face"], dtype=np.float64) for i in ids])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cats = np.asarray([medias[i].get("category") or "" for i in ids])
    return ids, vecs, cats


def per_photo(name: str, ids, vecs, cats, people: set[str], pronoun: dict[str, str], chunk: int = 256) -> pd.DataFrame:
    queries = np.flatnonzero(np.isin(cats, list(people)))
    rows = []
    for s in range(0, len(queries), chunk):
        q = queries[s : s + chunk]
        sims = vecs[q] @ vecs.T
        sims[np.arange(len(q)), q] = -np.inf
        for r, qi in enumerate(q):
            row_s = sims[r]
            own = (cats == cats[qi]) & (np.arange(len(cats)) != qi)
            other = cats != cats[qi]
            order = np.argsort(-row_s, kind="stable")
            rel = own[order]
            hits = np.cumsum(rel)
            npos = int(rel.sum())
            ap = float((hits[rel] / (np.flatnonzero(rel) + 1)).sum() / npos) if npos else np.nan
            gen = row_s[own]
            imp = row_s[other]
            top = np.argsort(-imp)[:10]
            top1_cat = cats[other][top[0]]
            med = float(np.median(gen)) if len(gen) else np.nan
            rows.append(
                {
                    "cell": name,
                    "id": int(ids[qi]),
                    "category": cats[qi],
                    "n_own": npos,
                    "loo_ap": ap,
                    "genuine_mean": float(gen.mean()) if len(gen) else np.nan,
                    "genuine_min": float(gen.min()) if len(gen) else np.nan,
                    "impostor_top1": float(imp[top[0]]),
                    "impostor_top10": float(imp[top].mean()),
                    "impostor_top1_pronoun": pronoun.get(top1_cat, "nobody") if top1_cat else "nobody",
                    "margin": (float(gen.mean()) - float(imp[top[0]])) if len(gen) else np.nan,
                    "impostors_above": int((imp > med).sum()) if len(gen) else -1,
                }
            )
    return pd.DataFrame(rows)


def genuine_pairs(name: str, ids, vecs, cats, people: set[str]) -> pd.DataFrame:
    """Every pair of one person's subject photos, with its cosine."""
    rows = []
    for cat in sorted(people):
        idx = np.flatnonzero(cats == cat)
        sims = vecs[idx] @ vecs[idx].T
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                rows.append(
                    {
                        "cell": name,
                        "category": cat,
                        "id_a": int(ids[idx[a]]),
                        "id_b": int(ids[idx[b]]),
                        "cos": float(sims[a, b]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--identities", required=True)
    ap.add_argument("--strata", required=True, help="person_strata.csv (pronoun per person)")
    ap.add_argument("--covariates", default=None, help="photo_covariates.csv: every subject's pronoun, for impostors")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    people = {line.strip() for line in open(args.identities) if line.strip() and not line.startswith("#")}
    strata = pd.read_csv(args.strata, dtype={"skin_tone": str})
    pronoun = dict(zip(strata["category"], strata["pronoun"]))
    if args.covariates:
        cov = pd.read_csv(args.covariates, usecols=["subject", "pronoun"])
        pronoun = {**cov.groupby("subject")["pronoun"].agg(lambda s: s.mode().iloc[0]).to_dict(), **pronoun}
    frames, pairs = [], []
    for spec in args.cell:
        name, path = spec.split("=", 1)
        ids, vecs, cats = load_cell(Path(path))
        df = per_photo(name, ids, vecs, cats, people, pronoun)
        comp = pd.Series([pronoun.get(c, "nobody") if c else "nobody" for c in cats]).value_counts().to_dict()
        print(f"{name}: {len(ids):,} faces ({comp}); {len(df):,} subject photos of {df['category'].nunique()} people")
        frames.append(df)
        pairs.append(genuine_pairs(name, ids, vecs, cats, people))
    photos = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        photos.to_csv(args.out / "static_photos.csv", index=False)
        pd.concat(pairs, ignore_index=True).to_csv(args.out / "static_pairs.csv", index=False)
        num = ["loo_ap", "genuine_mean", "genuine_min", "impostor_top1", "impostor_top10", "margin", "impostors_above"]
        ppl = photos.groupby(["cell", "category"])[num].mean().reset_index().merge(strata, on="category", how="left")
        ppl.to_csv(args.out / "static_people.csv", index=False)
    finally:
        os.umask(old)
    for name, g in ppl.groupby("cell"):
        print(f"\n{name}: per-person means by pronoun")
        print(g.groupby("pronoun")[num].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
