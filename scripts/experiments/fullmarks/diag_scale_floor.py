"""Where should the similarity-scale floor sit? (#3912)

For every roster class, the query crop is verified against its own page, 8 true
positives and 8 same-source negatives, with pages re-extracted at a 16,384
keypoint budget.  Each pair is fit once and judged at several ``_MIN_SANE_SCALE``
floors, and the fitted scale is recorded so the floor can be read off the data.

    source scripts/experiments/pile/pile_env.sh
    python scripts/experiments/fullmarks/diag_scale_floor.py <out.json>

For every roster class: the query crop against its own page, 8 true positives and
8 same-source negatives, using the *stored* tier-s cell features (what production
verifies against).  Each pair is fit once and judged twice: with the shipped
``_MIN_SANE_SCALE`` and with the floor lowered.  The fitted scale is recorded so
the window can be read off the data rather than guessed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fullmarks_config as cfg  # noqa: E402
import embed_corpus  # noqa: E402
import eval_retrieval as ev  # noqa: E402

N = 8


def main(out: Path) -> int:
    import vtscore.media.structural as st
    from vtscore.media import get_embedder

    classes = {c: m for c, m in json.loads((cfg.OUT / "classes.json").read_text()).items() if m.get("on_roster")}
    pages = {p.page_id: p for p in embed_corpus.pages_for_tier(cfg.OUT, "s")}
    rng = random.Random(3904)  # the same draw diag_structural.py made
    plan, wanted = {}, set()
    for cid, meta in classes.items():
        pos = [p for p in meta["page_ids"] if p != meta["query_page_id"] and p in pages]
        neg = [p for p, pg in pages.items() if pg.source == meta["source"] and p not in meta["page_ids"]]
        plan[cid] = {
            "own": [meta["query_page_id"]],
            "pos": rng.sample(pos, min(N, len(pos))),
            "neg": rng.sample(neg, min(N, len(neg))),
        }
        wanted |= {*plan[cid]["own"], *plan[cid]["pos"], *plan[cid]["neg"]}
    from vtscore.media.image._image_bulk import _load_pil

    emb = get_embedder("sift_vlad")
    emb.load_models()
    m = emb.structural_matcher

    rows = []
    for cid, meta in sorted(classes.items()):
        _, crop = ev.embed_query(emb, Path(meta["query_crop"]))
        rec = {"class_id": cid, "crop_kp": getattr(crop, "count", 0)}
        for role in ("own", "pos", "neg"):
            fits = []
            for p in plan[cid][role]:
                if crop is None:
                    continue
                img = _load_pil(Path(pages[p].path))
                f = m.detect_and_describe(np.asarray(img.convert("L"), dtype=np.uint8), max_features=16384)
                s = m.verify(crop, f)
                fit = {
                    "page": p,
                    "inliers": s.inlier_count,
                    "scale": round(s.scale, 4),
                    "reflection": bool(s.reflection),
                }
                for floor in (0.1, 0.05, 0.03, 0.01):
                    fit[f"ok@{floor}"] = bool(
                        s.inlier_count >= st._MIN_MODEL_INLIERS
                        and floor <= s.scale <= st._MAX_SANE_SCALE
                        and not s.reflection
                    )
                fits.append(fit)
            rec[role] = fits
        rows.append(rec)

        def summ(role, key):
            xs = [f["inliers"] if f[key] else 0 for f in rec[role]]
            return f"{np.median(xs):.0f}/{max(xs, default=0)}" if xs else "-"

        own = rec["own"][0] if rec["own"] else {}
        print(
            f"{cid:40s} own {own.get('inliers')}@{own.get('scale')} | "
            + " | ".join(
                f"@{fl} pos {summ('pos', f'ok@{fl}')} neg {summ('neg', f'ok@{fl}')}" for fl in (0.1, 0.03, 0.01)
            ),
            flush=True,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
