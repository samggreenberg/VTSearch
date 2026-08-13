"""Zero-click text-sort baseline, against the clicked detector's trajectory.

The question this answers: a user can type "car" and get a ranked haystack with a
GMM cut, for free. How many Good/Bad clicks does the trained detector need before
it beats that?

Scoring deliberately reuses the harness's own ``exemplar_sims`` path (the same
one the Autopilot seed phase uses for a cropped exemplar), with the text vector
substituted for the crop vector, so the text sort and the detector are scored in
the same geometry. The split is reproduced exactly: ``simulate_voting_iterations``
draws ``RandomState(seed)`` and, at natural prevalence, the media split is its
first draw.

Text is only available where the embedder has a text tower: ``dinov3_patch`` is
vision-only and ``embed_text`` returns None there, which is reported as n/a
rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402


def _split(clips: dict, sim_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Byte-identical to ``_split_media_ids`` under a fresh RandomState(seed)."""
    rng = np.random.RandomState(seed)
    all_ids = sorted(clips.keys())
    shuffled = rng.permutation(all_ids).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="experiment results dir (for prepare_info.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score

    from vtscore.embedding import embed_text_query
    from vtscore.eval.labels import media_is_positive
    from vtscore.eval.patch_styles import resolve_style
    from vtscore.training.thresholds import calculate_gmm_threshold

    from vtscore.datasets import loader as _loader  # isort: skip

    from _cells_io import load_medias  # noqa: PLC0415

    prepare = json.loads((Path(args.results) / "prepare_info.json").read_text())
    rows: list[dict] = []

    for ds, per_emb in prepare.get("datasets", {}).items():
        for emb, info in per_emb.items():
            cats = info.get("selected_categories") or []
            if not cats:
                continue
            pkl = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
            if not pkl.exists():
                common.log(f"SKIP {ds} x {emb}: no pickle")
                continue

            # Probe the text tower once before loading a multi-GB pickle.
            probe = embed_text_query("a photo", "image", embedder_name=emb)
            if probe is None:
                common.log(f"{ds} x {emb}: NO TEXT TOWER (vision-only) - recorded as n/a")
                for cat in cats:
                    rows.append({"dataset": ds, "embedder": emb, "category": cat, "supports_text": 0})
                continue

            medias = load_medias(pkl)
            common.log(f"\n=== {ds} x {emb} === {len(medias)} medias, {len(cats)} categories")
            style = resolve_style("whole_image")

            for cat in cats:
                tvec = embed_text_query(cat, "image", embedder_name=emb)
                if tvec is None:
                    rows.append({"dataset": ds, "embedder": emb, "category": cat, "supports_text": 0})
                    continue

                # Same scorer the seed phase uses, text vector in place of a crop.
                sims = style.exemplar_sims(medias, np.asarray(tvec, dtype=np.float32))
                ids = sorted(medias.keys())
                scores = np.asarray([float(sims[i]) for i in ids], dtype=np.float64)
                labels = np.asarray([1 if media_is_positive(medias[i], cat) else 0 for i in ids])

                # The app cuts the haystack it can see: every media, not a split.
                gmm_cut = float(calculate_gmm_threshold([float(s) for s in scores]))

                for seed in cfg.SEEDS:
                    _, test_ids = _split(medias, cfg.SIM_FRACTION, seed)
                    tset = set(test_ids)
                    mask = np.asarray([i in tset for i in ids])
                    y, s = labels[mask], scores[mask]
                    npos, nneg = int(y.sum()), int((1 - y).sum())
                    if npos == 0 or nneg == 0:
                        continue
                    pred = s >= gmm_cut
                    fpr = float(((pred == 1) & (y == 0)).sum() / nneg)
                    fnr = float(((pred == 0) & (y == 1)).sum() / npos)
                    # Oracle: best achievable cut on these same text scores.
                    order = np.argsort(-s)
                    ys = y[order]
                    tp = np.cumsum(ys)
                    fp = np.cumsum(1 - ys)
                    ocost = np.min(fp / max(nneg, 1) + (npos - tp) / max(npos, 1))
                    rows.append(
                        {
                            "dataset": ds,
                            "embedder": emb,
                            "category": cat,
                            "seed": seed,
                            "supports_text": 1,
                            "n_test": int(mask.sum()),
                            "n_test_pos": npos,
                            "prevalence": round(npos / max(npos + nneg, 1), 6),
                            "text_gmm_cut": round(gmm_cut, 6),
                            "text_cost": round(fpr + fnr, 6),
                            "text_fpr": round(fpr, 6),
                            "text_fnr": round(fnr, 6),
                            "text_oracle_cost": round(float(ocost), 6),
                            "text_AP": round(float(average_precision_score(y, s)), 6),
                            "text_auroc": round(float(roc_auc_score(y, s)), 6),
                        }
                    )
                common.log(
                    f"  {cat:22s} gmm_cut={gmm_cut:.4f} AP={rows[-1].get('text_AP')} cost={rows[-1].get('text_cost')}"
                )

    import pandas as pd

    pd.DataFrame(rows).to_csv(args.out, index=False)
    common.log(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
