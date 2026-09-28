#!/usr/bin/env python3
"""Size vs size (#4160): the zero-click text sort, scored at each TEST size.

    python baseline.py --exp <exp dir> --out <exp>/text_baseline.csv

The click-0 anchor of every curve: what typing the class's query alone gets,
before any training size exists. Scored exactly as the arms are: per seed, each
test size's cohort is the one every arm is tested on
(``scale_bands.band_cohorts``) and the negatives are the class's shared pool's
held-out half (``scale_bands.holdout_ids``, as ``paired_mix`` splits them). The
cut is the app's text-sort rule over the class's whole haystack, every size at
once, because a user who has only typed has not picked a size.

Test SML is the weighted average of the three bands, equal or natural, as in
``analyze.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import common  # noqa: E402

common.setup_env()

import experiment_config as cfg  # noqa: E402

LABEL = {"small": "S", "medium": "M", "large": "L"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedder", default="siglip")
    args = ap.parse_args()

    from _cells_io import load_medias
    from sklearn.metrics import roc_auc_score

    from vtscore.config import EMBEDDINGS_DIR
    from vtscore.embedding import embed_text_query
    from vtscore.eval import scale_bands as sb
    from vtscore.eval.labels import media_is_evaluable
    from vtscore.eval.patch_styles import resolve_style
    from vtscore.training.thresholds import inclusion_cost_weights, text_sort_threshold

    ds, emb = "coco_better", args.embedder
    wf, wn = inclusion_cost_weights(cfg.INCLUSION)
    shares = json.loads((args.exp / "mix_shares.json").read_text())
    medias = load_medias(EMBEDDINGS_DIR / cfg.text_pickle_name(ds, emb))
    style = resolve_style("whole_image")
    text_emb = cfg.text_embedder(emb)
    rows = []
    for cls in sorted(shares):
        cells = [c for c in sb.cells_of_class(medias, cls) if not sb.is_mix_band(sb.parse_cell(c)[1])]
        pool = {i: m for i, m in medias.items() if any(media_is_evaluable(m, c) for c in cells)}
        query = cfg.seed_query_text(ds, cells[0])
        tvec = embed_text_query(query, "image", enrich=cfg.SEED_ENRICH, embedder_name=text_emb)
        sims = style.exemplar_sims(pool, np.asarray(tvec, dtype=np.float32))
        cut = float(text_sort_threshold([float(sims[i]) for i in sorted(pool)]))
        negatives = sorted(i for i, m in pool.items() if sb.band_of(m, cls) is None)
        for seed in cfg.SEEDS:
            held_neg = sb.holdout_ids(negatives, cfg.SIM_FRACTION, seed)
            neg = np.asarray([sims[i] for i in held_neg], dtype=np.float64)
            fpr = float(np.mean(neg >= cut))
            cohorts = sb.band_cohorts(medias, cells[0], sim_fraction=cfg.SIM_FRACTION, seed=seed)
            per_band = {}
            for band, ids in cohorts.items():
                pos = np.asarray([sims[i] for i in ids], dtype=np.float64)
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                per_band[band] = {
                    "fnr": float(np.mean(pos < cut)),
                    "auroc": float(roc_auc_score(y, np.r_[pos, neg])),
                    "n_pos": len(pos),
                }
            tests = {LABEL[b]: {b: 1.0} for b in per_band}
            tests["SML="] = {b: 1.0 / len(per_band) for b in per_band}
            nat = {b: float(shares[cls].get(b, 0.0)) for b in per_band}
            tot = sum(nat.values())
            tests["SMLn"] = {b: w / tot for b, w in nat.items()}
            for test, w in tests.items():
                fnr = sum(w[b] * per_band[b]["fnr"] for b in w)
                auroc = sum(w[b] * per_band[b]["auroc"] for b in w)
                n_pos = sum(per_band[b]["n_pos"] for b in w)
                tp, fp = (1 - fnr) * n_pos, fpr * len(neg)
                rows.append(
                    {
                        "embedder": emb,
                        "cls": cls,
                        "seed": seed,
                        "test": test,
                        "query": query,
                        "text_cut": round(cut, 6),
                        "text_fpr": round(fpr, 6),
                        "text_fnr": round(fnr, 6),
                        "text_recall": round(1 - fnr, 6),
                        "text_cost": round(wf * fpr + wn * fnr, 6),
                        "text_auroc": round(auroc, 6),
                        "text_f1": round(2 * tp / (2 * tp + fp + (n_pos - tp)), 6) if n_pos else float("nan"),
                    }
                )
        common.log(f"{cls:32s} query={query!r} cut={cut:.3f} fpr={fpr:.3f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    common.log(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
