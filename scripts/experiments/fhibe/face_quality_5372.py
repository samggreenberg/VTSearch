#!/usr/bin/env python
"""Pin the Face Quality cut under which a starting face is a hard view (#5372).

Reads #5325's shipped arm (``2026-10-27-5325z55-ik``, buffalo_s, the face default since #5369), every set:
one photo (K = 1) and four (K = 4), 1024 px and 640 px, beta 1/4, 1 and 4. No new sessions. For each session it
joins the starting faces (the pick log's ``example`` rows) to the Face Quality their converter stamped
(``custom_metadata["Face Quality"]``, the raw embedding's L2 norm) and to the outcome: failed at #4777's bar
(final AP below 0.1 while the person's full-label ceiling reaches 0.5) and F-beta at 50 votes.

Writes to ``--out``:

* ``cuts.csv``: per K, size, beta, which start quality (min / max of the starts) and cut: the sessions flagged,
  the failures among them, F at 50 votes flagged and not, and the share of all the pile's faces under the cut;
* ``fails.csv``: every session that fails at every beta, with its start quality;
* ``auc.csv``: per K, size and beta, the AUC with which a low start quality ranks the failed sessions first.

Run on a compute node with the app venv (it reads two ~200 MB pickles and ~6,700 pick logs)::

    python scripts/experiments/fhibe/face_quality_5372.py --out docs/experiments/2026-10-28-face-quality-5372
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = "/expscratch/sgreenberg/fhibe/fhibe-full-resolution-674a7dcf-derived"
ARM = "2026-10-27-5325z55-ik"
BETAS = {"b025": 0.25, "b1": 1.0, "b4": 4.0}
CUTS = (12, 13, 14, 15, 16, 18)


def _starts(path: str) -> dict[tuple[str, str, str], list[int]]:
    """The starting media ids of one cell, from its pick log's ``example`` rows (they come first)."""
    out: dict[tuple[str, str, str], list[int]] = {}
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "example":
                break
            out.setdefault((row["dataset"], row["category"], row["seed"]), []).append(int(row["picked_id"]))
    return out


def sessions(derived: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per session with its start qualities and outcome, and the pile's qualities."""
    quality, pile = {}, []
    for res in ("1024", "640"):
        dataset = f"fhibe_faces_ik_{res}"
        with open(f"{derived}/cells/{dataset}__face.pkl", "rb") as f:
            medias = pickle.load(f)  # noqa: S301  # our own pile cell
        quality[dataset] = {i: (m["custom_metadata"]["Face Quality"], m["category"]) for i, m in medias.items()}
        pile += [(int(res), q) for q, _ in quality[dataset].values()]
    runs = f"{derived}/runs"
    rows = []
    for k in ("k1", "k4"):
        for b, beta in BETAS.items():
            starts: dict = {}
            with ProcessPoolExecutor(16) as ex:
                for part in ex.map(
                    _starts, glob.glob(f"{runs}/{ARM}-{k}-{b}/results/cells/task_*__picks.csv.gz"), chunksize=20
                ):
                    starts.update(part)
            for res in ("1024", "640"):
                dataset = f"fhibe_faces_ik_{res}"
                for r in csv.DictReader(open(f"{runs}/{ARM}-analysis/{k}-{b}-{res}/cells.csv")):
                    ids = starts[(dataset, r["category"], r["seed"])]
                    qs = [quality[dataset][i][0] for i in ids]
                    assert all(quality[dataset][i][1] == r["category"] for i in ids)
                    rows.append(
                        dict(
                            K=int(k[1:]),
                            beta=beta,
                            res=int(res),
                            category=r["category"],
                            q_min=min(qs),
                            q_max=max(qs),
                            final_ap=float(r["final_ap"]),
                            ceiling_ap=float(r["ceiling_ap"]),
                            f50=float(r["thr_fbeta_50"]),
                        )
                    )
    s = pd.DataFrame(rows)
    s["fail"] = (s.final_ap < 0.1) & (s.ceiling_ap >= 0.5)
    return s, pd.DataFrame(pile, columns=["res", "q"])


def _auc(low_first: np.ndarray, positive: np.ndarray) -> float:
    a, b = low_first[positive], low_first[~positive]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--derived", default=DERIVED)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s, pile = sessions(args.derived)
    every = s.groupby(["K", "res", "category"]).fail.transform("all")

    cuts, aucs = [], []
    for (k, res, beta), g in s.groupby(["K", "res", "beta"]):
        aucs.append(
            dict(
                K=k,
                res=res,
                beta=beta,
                sessions=len(g),
                failed=int(g.fail.sum()),
                auc_start_q=round(_auc(-g.q_min.to_numpy(), g.fail.to_numpy()), 3),
            )
        )
        for key in ("q_min", "q_max") if k > 1 else ("q_min",):
            for cut in CUTS:
                m = g[key] < cut
                cuts.append(
                    dict(
                        K=k,
                        res=res,
                        beta=beta,
                        start=("only" if k == 1 else key[2:]),
                        cut=cut,
                        flagged=int(m.sum()),
                        flagged_failed=int(g.fail[m].sum()),
                        failed=int(g.fail.sum()),
                        f50_flagged=round(g.f50[m].mean(), 3) if m.any() else None,
                        f50_rest=round(g.f50[~m].mean(), 3),
                        pile_share_below=round(float((pile[pile.res == res].q < cut).mean()), 4),
                    )
                )
    pd.DataFrame(cuts).to_csv(args.out / "cuts.csv", index=False)
    pd.DataFrame(aucs).to_csv(args.out / "auc.csv", index=False)
    fails = s[every & (s.beta == 1)][["K", "res", "category", "q_min", "q_max"]]
    fails.assign(category=fails.category.str[:8]).round(2).to_csv(args.out / "fails.csv", index=False)


if __name__ == "__main__":
    main()
