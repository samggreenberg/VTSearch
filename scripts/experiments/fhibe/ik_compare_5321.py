#!/usr/bin/env python
"""The imagekit face provider against FaceNet, session for session (#5321).

Both arms ran the face review's sessions on one commit: ``roll`` is the FaceNet path (MTCNN boxes,
``roll`` crops, FaceNet vectors) and ``ik`` is the imagekit provider's (its own detector's boxes and
its embedder's vectors from the full photo). The same people, starting photos and withheld halves:
a person's photos split by photo id, and a subject crop carries its photo's id in both arms. The
pools differ in the other faces each detector keeps as negatives, and in which subject photos each
finds, which is part of what is being priced.

Per K x size x beta, from ``align_compare_4777.session_set`` (the same columns the earlier face
studies read): session mean of the objective over clicks 1-150, click 0, after the check
(``thr_fbeta_final``), final AP, ceiling AP, failed sessions; each arm's mean, the paired difference
and its SE over people. Then the gaps the October face studies reported, per arm and their change:
she/her - he/him and 18-29 - 45+ (45-59 and 60+ pooled), on the session mean and after the check.

    python ik_compare_5321.py --roll <analysis>,<baseline dir> --ik <analysis>,<baseline dir> \\
        --strata <person_strata.csv> --out <dir>

Writes ``paired.csv`` and ``gaps.csv`` (aggregates, no subject ids) and ``runs.csv`` (per run, by
subject id: owner-only, never committed).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from align_compare_4777 import BETAS, KS, SIZES, failed, paired, session_set

METRICS = {"session_mean": "session mean", "thr_fbeta_final": "after the check"}
GROUPS = {
    "pronoun": ("she/her", "he/him"),
    "age": ("18-29", "45+"),
}


def _age_group(bracket: str) -> str:
    return "45+" if bracket in ("45-59", "60+") else bracket


def gap(j: pd.DataFrame, col: str, side: str, key: str) -> tuple[float, float, int, int]:
    """Mean of group a minus group b on ``<col><side>``, with the SE of a difference of two means."""
    a, b = GROUPS[key]
    x = j.loc[j[key] == a, f"{col}{side}"].dropna()
    y = j.loc[j[key] == b, f"{col}{side}"].dropna()
    se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
    return float(x.mean() - y.mean()), se, len(x), len(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roll", required=True, help="<analysis dir>,<baseline dir> of the FaceNet arm")
    ap.add_argument("--ik", required=True, help="<analysis dir>,<baseline dir> of the imagekit arm")
    ap.add_argument("--strata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fail", type=float, default=0.1)
    ap.add_argument("--fail-ceiling", type=float, default=0.5)
    args = ap.parse_args()
    roll_a, roll_b = (Path(p) for p in args.roll.split(",", 1))
    ik_a, ik_b = (Path(p) for p in args.ik.split(",", 1))
    strata = pd.read_csv(args.strata)
    sid = "category" if "category" in strata.columns else strata.columns[0]
    strata = strata.rename(columns={sid: "category"})[["category", "pronoun", "age_bracket"]]
    strata["age"] = strata["age_bracket"].map(_age_group)
    # FHIBE spells them "She/her/hers" and "He/him/his"; one person is "They/them/their".
    strata["pronoun"] = strata["pronoun"].str.lower().str.extract(r"^(she/her|he/him)", expand=False)

    rows, gaps, runs = [], [], []
    for k in KS:
        for size in SIZES:
            for beta in BETAS:
                fn = session_set(roll_a, roll_b, k, size, beta, "fhibe_faces_roll")
                ik = session_set(ik_a, ik_b, k, size, beta, "fhibe_faces_ik")
                j = fn.join(ik, lsuffix="_plain", rsuffix="_arm", how="inner").reset_index()
                row = {"K": k, "size": size, "beta": beta} | paired(fn, ik, args.fail, args.fail_ceiling)
                # paired() reports the control's ceiling; each detector has its own.
                row["ceiling_ap_arm"] = round(float(j["ceiling_ap_arm"].mean()), 4)
                rows.append(row)
                j = j.merge(strata, on="category", how="left")
                j["fail_plain"] = failed(j, args.fail, args.fail_ceiling, "_plain")
                j["fail_arm"] = failed(j, args.fail, args.fail_ceiling, "_arm")
                runs.append(j.assign(K=k, size=size, beta=beta))
                for col, label in METRICS.items():
                    for key in GROUPS:
                        g = {"K": k, "size": size, "beta": beta, "metric": label, "gap": "-".join(GROUPS[key])}
                        for arm, side in (("facenet", "_plain"), ("imagekit", "_arm")):
                            d, se, na, nb = gap(j, col, side, key)
                            g |= {f"{arm}": round(d, 4), f"{arm}_se": round(se, 4)}
                        # The change in the gap, paired by person: the gap of the per-person differences.
                        j["_d"] = j[f"{col}_arm"] - j[f"{col}_plain"]
                        d, se, na, nb = gap(j, "_d", "", key)
                        g |= {"change": round(d, 4), "change_se": round(se, 4), "n_a": na, "n_b": nb}
                        gaps.append(g)
    args.out.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        pd.DataFrame(rows).to_csv(args.out / "paired.csv", index=False)
        pd.DataFrame(gaps).to_csv(args.out / "gaps.csv", index=False)
        pd.concat(runs).to_csv(args.out / "runs.csv", index=False)
    finally:
        os.umask(old)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["K", "size", "beta", "runs", "unpaired", "session_mean_plain", "session_mean_arm", "d_session_mean",
                "se_session_mean", "thr_fbeta_final_plain", "thr_fbeta_final_arm", "text_thr_fbeta_plain",
                "text_thr_fbeta_arm", "fails_plain", "fails_arm", "ceiling_ap", "ceiling_ap_arm"]  # fmt: skip
        print(pd.DataFrame(rows)[cols].to_string(index=False))
        print(pd.DataFrame(gaps).to_string(index=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
