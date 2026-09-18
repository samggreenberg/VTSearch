"""Why does structural search score near chance on DocMarks?  (#3904 diagnostic)

For each roster class: SIFT inliers between the query crop and (a) the page it
was cut from, (b) a sample of true positives, (c) a sample of eligible
negatives (same-source non-members) -- first with the features stored in the
tier-s cell, then with the page re-extracted at larger keypoint budgets.

    python diag_structural.py <out.json>  If the crop cannot verify
against its own page with stored features but can with more keypoints, the
per-page feature cap is starving the mark.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402

#: Keypoint budgets to verify at; ``stored`` means the features already in the cell.
#: Space-separated so ``sbatch --export`` (which splits on commas) can pass it.
BUDGETS = tuple(None if b == "stored" else int(b) for b in os.environ.get("DIAG_BUDGETS", "stored 4096 16384").split())
#: Only classes whose id starts with one of these (space-separated); all when unset.
PREFIXES = tuple(os.environ.get("DIAG_PREFIXES", "").split())
#: ``sift`` (the shipped ``sift_vlad`` matcher) or ``splg`` (SuperPoint +
#: LightGlue, ``vtscore.media.structural_splg`` -- the backend the 2026-07-13 study
#: found beats SigLIP on documents, never wired to an embedder; #3911).  ``splg``
#: has no stored features, so ``stored`` is not a valid budget for it.
BACKEND = os.environ.get("DIAG_BACKEND", "sift")
#: SuperPoint resizes a page's long side to this before extracting.
LONG_SIDE = int(os.environ.get("DIAG_LONG_SIDE", "1536"))
#: RANSAC model for ``splg``: ``similarity`` (production) or ``scale_translation``.
GEOMETRY = os.environ.get("DIAG_GEOMETRY", "similarity")
N_SAMPLE = 8


def main() -> int:
    from vtscore.media import get_embedder  # noqa: PLC0415
    from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415

    corpus = cfg.OUT
    classes = {
        c: m
        for c, m in json.loads((corpus / "classes.json").read_text()).items()
        if m.get("on_roster") and (not PREFIXES or c.startswith(PREFIXES))
    }
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(corpus, "s")}
    emb = get_embedder("sift_vlad")
    emb.load_models()
    if BACKEND == "splg":
        from vtscore.media.structural_splg import SplgMatcher  # noqa: PLC0415

        matcher = SplgMatcher(geometry_model=GEOMETRY, extract_long_side=LONG_SIDE)
        if None in BUDGETS:
            raise SystemExit("DIAG_BACKEND=splg has no stored features; give numeric DIAG_BUDGETS")
    else:
        matcher = emb.structural_matcher
    rng = random.Random(3904)

    wanted: set[str] = set()
    plan = {}
    for cid, meta in classes.items():
        pos = [p for p in meta["page_ids"] if p != meta["query_page_id"] and p in pages]
        neg = [p for p, pg in pages.items() if pg.source == meta["source"] and p not in meta["page_ids"]]
        plan[cid] = {
            "own": meta["query_page_id"],
            "pos": rng.sample(pos, min(N_SAMPLE, len(pos))),
            "neg": rng.sample(neg, min(N_SAMPLE, len(neg))),
        }
        wanted |= {plan[cid]["own"], *plan[cid]["pos"], *plan[cid]["neg"]}
    stored = ev.read_local_features(embed_corpus.cell_path("s", "sift_vlad"), wanted) if BACKEND == "sift" else {}
    print(f"backend {BACKEND}, budgets {BUDGETS}, long side {LONG_SIDE}, geometry {GEOMETRY}")
    if stored:
        print(
            f"{len(stored)} stored feature sets; keypoints per page: median {np.median([f.count for f in stored.values() if f is not None]):.0f}"
        )
        print(f"default max_features: {emb.max_features}")

    def feats_for(page_id, budget):
        if budget is None:
            return stored.get(page_id)
        img = _load_pil(Path(pages[page_id].path))
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        return matcher.detect_and_describe(gray, max_features=budget)

    print("\nclass | crop kp | budget | own inliers | pos median/max | neg median/max")
    out = []
    page_kp: dict[object, list[int]] = {}
    for cid, meta in sorted(classes.items()):
        if BACKEND == "splg":
            with Image.open(meta["query_crop"]) as img:
                crop = matcher.detect_and_describe(
                    np.asarray(img.convert("L"), dtype=np.uint8), max_features=max(BUDGETS)
                )
        else:
            _, crop = ev.embed_query(emb, Path(meta["query_crop"]))
        for budget in BUDGETS:

            def inl(p):
                f = feats_for(p, budget)
                if f is None or crop is None:
                    return 0
                page_kp.setdefault(budget, []).append(f.count)
                s = matcher.verify(crop, f)
                return s.inlier_count if s.model_ok else 0

            own = inl(plan[cid]["own"])
            pos = [inl(p) for p in plan[cid]["pos"]]
            neg = [inl(p) for p in plan[cid]["neg"]]
            row = dict(
                backend=BACKEND,
                class_id=cid,
                crop_kp=getattr(crop, "count", 0),
                budget=budget or "stored",
                own=own,
                pos_med=float(np.median(pos)) if pos else 0,
                pos_max=max(pos, default=0),
                neg_med=float(np.median(neg)) if neg else 0,
                neg_max=max(neg, default=0),
            )
            out.append(row)
            print(
                f"{cid} | {row['crop_kp']} | {row['budget']} | {own} | {row['pos_med']:.0f}/{row['pos_max']} | {row['neg_med']:.0f}/{row['neg_max']}",
                flush=True,
            )
    for budget, counts in page_kp.items():
        # What a stored page would cost: 2-d float32 keypoints plus descriptors
        # (128-d uint8 for SIFT; 256-d fp16 for SuperPoint, which uint8 destroys).
        per_kp = 8 + (128 if BACKEND == "sift" else 512)
        print(
            f"budget {budget}: median {np.median(counts):.0f} keypoints/page, ~{np.median(counts) * per_kp / 1024:.0f} KB/page stored"
        )
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
