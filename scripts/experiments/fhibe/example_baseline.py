#!/usr/bin/env python
"""Click 0 of a face review (#4762): the example sort, scored as ``text_baseline.py`` scores a typed query.

On FHIBE there is nothing to type. The session starts from K photos of the person (the example
opening, #4699), and until Autopilot shows a detector the screen holds the **example sort**: every
media ranked by cosine to the K photos' centroid, at its display line. Since #4732 that line is the
Goods' centroid's own (:func:`vtscore.detectors.centroid_head.centroid_cut`, the count line at the
balance), drawn over every media in the pool, as the app's example-sort route draws it
(:func:`vtscore.training.query_sort.cosine_sort_cuts`). This script writes that sort's scores in
``text_baseline.py``'s columns, one file per K, so the State of the App analyzer
(``state_of_app/analyze.py --baseline``) reads it as the click-0 anchor and as what a session shows
before its first detector.

The cell is reproduced with the harness's own pieces, so it is the sort the run opened on:
``evaluable_pool``, ``stratified_split`` on ``RandomState(seed)`` (the run's first draw),
``choose_examples`` and ``example_sort``. Each row is scored on the cell's withheld half. With
``--check`` it compares the withheld half's AP with the run's vote-K row, which Test cuts on the
same centroid.

Columns kept from ``text_baseline.py`` (``text_*``): ``text_AP``; the line at each preset
(``text_line_{cut,precision,recall,fpr}_b*``); the balance's top-K cap and best cut
(``text_{k,precision,recall,fbeta,oracle_fbeta,fb_share}_b*``,
``text_oracle_{precision,recall,fpr}_b*``); the precision-floor columns
(``text_*_x{10,50,90}``); and ``text_{precision,recall,fpr}`` at the two-Gaussian midpoint,
the example sort's line before #4732.

Usage (in a run dir's environment, e.g. ``sota_face.sh baseline``)::

    python example_baseline.py --results <run>/results --ks 1,4 --out-dir <dir> \\
        [--check '<runs>/<date>-sota-k{k}-b1/results/cells']
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
import common  # noqa: E402

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiment_config as cfg  # noqa: E402


def score_cell(pool: dict, cat: str, seed: int, k: int) -> dict | None:
    """One cell's example sort at K photos, scored on its withheld half, in ``text_baseline.py``'s columns."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from _rank_metrics import BETAS, FLOORS, balance_metrics, beta_tag, floor_tag, line_metrics, ranks_from_scores
    from vtscore.detectors.centroid_head import centroid_cut
    from vtscore.eval.calibration_metrics import detection_metrics, oracle_fbeta_cut
    from vtscore.eval.example_opening import choose_examples, example_sort, stratified_split
    from vtscore.eval.labels import media_is_positive

    rng = np.random.RandomState(seed)
    sim_ids, test_ids = stratified_split(pool, cat, cfg.SIM_FRACTION, rng)
    try:
        examples = choose_examples(sim_ids, pool, cat, k, seed)
    except ValueError:
        return None
    sims = example_sort(pool, examples)
    ids = sorted(pool)
    scores = np.asarray([float(sims[i]) for i in ids], dtype=np.float64)
    labels = np.asarray([1 if media_is_positive(pool[i], cat) else 0 for i in ids])
    all_scores = scores.tolist()
    # The display line over every media, as the app draws it; the midpoint is the pre-#4732 line.
    line_cut = {b: float(centroid_cut(all_scores, beta=b)) for b in BETAS}
    mid_cut = float(centroid_cut(all_scores, rule="midpoint"))
    tset = set(test_ids)
    mask = np.asarray([i in tset for i in ids])
    y, s = labels[mask], scores[mask]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    pred = s >= mid_cut
    det = detection_metrics(s, y, mid_cut)
    ranks = ranks_from_scores([i for i in ids if i in tset], s, y)
    row = {
        "dataset": None,
        "embedder": None,
        "category": cat,
        "seed": seed,
        "supports_text": 1,
        "query": f"{k} example photo{'s' if k > 1 else ''}",
        "examples": k,
        "n_test": int(mask.sum()),
        "n_test_pos": npos,
        "prevalence": round(npos / (npos + nneg), 6),
        "text_gmm_cut": round(mid_cut, 6),
        "text_fpr": round(float(((pred == 1) & (y == 0)).sum() / nneg), 6),
        "text_fnr": round(float(((pred == 0) & (y == 1)).sum() / npos), 6),
        "text_precision": round(det["precision"], 6),
        "text_recall": round(det["recall"], 6),
        "text_f1": round(det["f1"], 6),
        "text_AP": round(float(average_precision_score(y, s)), 6),
        "text_auroc": round(float(roc_auc_score(y, s)), 6),
    }
    for x in FLOORS:
        for name, v in line_metrics(ranks, int(mask.sum()), npos, x).items():
            row[f"text_{name}_{floor_tag(x)}"] = round(float(v), 6)
    for b in BETAS:
        tag = beta_tag(b)
        for name, v in balance_metrics(ranks, int(mask.sum()), npos, b, None).items():
            row[f"text_{name}_{tag}"] = round(float(v), 6)
        at = s >= line_cut[b]
        row[f"text_line_cut_{tag}"] = round(line_cut[b], 6)
        row[f"text_line_precision_{tag}"] = round(detection_metrics(s, y, line_cut[b])["precision"], 6)
        row[f"text_line_recall_{tag}"] = round(float((at & (y == 1)).sum() / npos), 6)
        row[f"text_line_fpr_{tag}"] = round(float((at & (y == 0)).sum() / nneg), 6)
        row[f"text_line_k_{tag}"] = int(at.sum())
        o_thr, _o_fb, o_fpr, o_fnr = oracle_fbeta_cut(s, y, b)
        row[f"text_oracle_precision_{tag}"] = round(detection_metrics(s, y, o_thr)["precision"], 6)
        row[f"text_oracle_recall_{tag}"] = round(1.0 - o_fnr, 6)
        row[f"text_oracle_fpr_{tag}"] = round(o_fpr, 6)
    return row


def check_against_runs(baseline: pd.DataFrame, cells: Path, k: int) -> str:
    """The withheld half's AP against each run's vote-K row (Test's centroid of the K photos)."""
    import _cells_io  # noqa: PLC0415

    got = []
    for f in _cells_io.main_frame_files(cells):
        try:
            df = pd.read_csv(f, usecols=["dataset", "category", "seed", "t", "average_precision", "gmm_variant"])
        except (pd.errors.EmptyDataError, ValueError, OSError):
            continue
        df = df[df["gmm_variant"].isna() & (df["t"] == k)]
        if len(df):
            got.append(df.iloc[-1])
    if not got:
        return "check: no run rows at vote K yet"
    runs = pd.DataFrame(got).merge(baseline, on=["dataset", "category", "seed"])
    d = (runs["average_precision"] - runs["text_AP"]).abs()
    return f"check (K={k}): {len(runs)} runs, |AP(run vote K) - AP(example sort)| median {d.median():.2e}, max {d.max():.2e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="a run's results dir (for prepare_info.json)")
    ap.add_argument("--ks", default="1,4", help="example counts, one output file each")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", default=None, help="the cells dirs to compare vote-K APs with, '{k}' standing for K")
    args = ap.parse_args()

    from vtscore.config import EMBEDDINGS_DIR  # isort: skip
    from vtscore.eval.labels import evaluable_pool

    from _cells_io import load_medias  # noqa: PLC0415

    ks = [int(k) for k in args.ks.split(",") if k]
    prepare = json.loads((Path(args.results) / "prepare_info.json").read_text())
    rows: dict[int, list[dict]] = {k: [] for k in ks}
    for ds in cfg.DATASETS:
        for emb, info in prepare.get("datasets", {}).get(ds, {}).items():
            cats = info.get("selected_categories") or []
            medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name(ds, emb), repair=True)
            common.log(f"{ds} x {emb}: {len(medias)} medias, {len(cats)} people")
            for cat in cats:
                pool = evaluable_pool(medias, cat)
                for seed in cfg.SEEDS:
                    for k in ks:
                        r = score_cell(pool, cat, seed, k)
                        if r is not None:
                            r["dataset"], r["embedder"] = ds, emb
                            rows[k].append(r)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        df = pd.DataFrame(rows[k])
        path = out / f"example_baseline_k{k}.csv"
        df.to_csv(path, index=False)
        common.log(f"K={k}: {len(df)} rows -> {path}")
        if args.check:
            common.log(check_against_runs(df, Path(args.check.format(k=k)), k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
