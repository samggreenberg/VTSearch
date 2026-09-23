#!/usr/bin/env python
"""Replay every captured text sort through every candidate line - the #3826 gate.

    python gate_3826.py --capture <corpus>/coco_val__siglip.npz --out <analysis>/parts

One CSV per capture, one row per (sort, rule, frame):

``full``
    the rule on the sort as captured.  With labels, the admitted set's
    confusion counts; for a mixture rule, the fit's diagnostics too.
``opt:<p>``
    the rule with its **optimiser** perturbed - a random start, or a different
    tolerance - on the same data.  This is the issue's own experiment
    (re-initialise, re-tolerance, count the verdicts that move), generalised to
    every rule that has an optimiser.  A closed-form rule has no such rows.
``boot:<b>``
    the rule on a bootstrap resample of the sort, applied back to the sort as
    captured.  This is the stability question every rule can be asked,
    including the ones with nothing to re-initialise: *how much of the line is
    the data, and how much is this particular draw of it?*  Every rule sees the
    same resamples, so the frame is paired.

``flip_frac`` is the fraction of the haystack whose verdict differs from the
``full`` row; ``jaccard_dist`` is ``1 - |A & B| / |A | B|`` on the admitted
sets, which does not flatter a rule for admitting little.

The first thing the gate does on every sort is assert that ``gmm_shipped`` *is*
``calculate_gmm_threshold`` to the bit - an arm that re-implements the shipped
rule and drifts from it would make every comparison against it meaningless.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "calibration"))
sys.path.insert(0, HERE)

import common  # noqa: E402

common.setup_env()

import rules_3826 as R  # noqa: E402

N_BOOT = int(os.environ.get("TEXTCUT_N_BOOT", "20"))

COLUMNS = (
    "capture",
    "dataset",
    "embedder",
    "category",
    "query",
    "curated",
    "n",
    "n_pos",
    "rule",
    "family",
    "frame",
    "cut",
    "n_adm",
    "tp",
    "fp",
    "flip_frac",
    "jaccard_dist",
    "seconds",
    "ashman_d",
    "w_hi",
    "loglik",
    "n_iter",
    "converged",
    "bulk_pred_fp",
)


def _sorts(z) -> "list[tuple[str, np.ndarray, np.ndarray | None, dict]]":
    """``(key, scores, labels-or-None, identity)`` for either capture format."""
    meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8")) if "_meta" in z.files else {}
    out = []
    for key in sorted(k for k in z.files if k.endswith("|scores")):
        base = key[: -len("|scores")]
        parts = base.split("|")
        if parts[0] == "sort":  # the #3585 capture: sort|ds|emb|cat
            ds, emb, cat = parts[1], parts[2], parts[3]
            ident = {"dataset": ds, "embedder": emb, "category": cat, "query": "", "curated": ""}
        else:
            ds, emb, cat = parts[0], parts[1], parts[2]
            q = meta.get("queries", {}).get(base, {})
            ident = {
                "dataset": ds,
                "embedder": emb,
                "category": q.get("category", cat),
                "query": q.get("query", ""),
                "curated": int(bool(q.get("curated", False))),
            }
        labels = z[f"{base}|labels"].astype(bool) if f"{base}|labels" in z.files else None
        out.append((base, np.asarray(z[key], dtype=np.float64), labels, ident))
    return out


def _reference_rules() -> "dict[str, R.Rule]":
    """The pre-#3585 sklearn fit and the issue's three-character control - reproduction only."""
    from vtscore.training.thresholds.gmm import fit_score_gmm_sklearn

    sys.path.insert(0, os.path.join(HERE, "..", "gmm_init"))
    import arms_3585 as A  # noqa: PLC0415

    kpp = A._sklearn_variant(init_params="k-means++")

    def mid(fit, x):
        return float(np.median(x)) if fit is None else fit.midpoint()

    return {
        "ref_sklearn": R.Rule(
            "ref_sklearn",
            "reference",
            lambda x: mid(fit_score_gmm_sklearn(x), x),
            (("kmeanspp", lambda x: mid(kpp(x), x)),),
        ),
        "ref_param1e-8": R.Rule(
            "ref_param1e-8",
            "reference",
            lambda x: mid(R.run_em(x, R.G._two_means_init(np.sort(x)), None, 200)[0], x),
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N sorts only (smoke)")
    ap.add_argument("--rules", default="", help="comma-separated subset (smoke)")
    ap.add_argument(
        "--chunk",
        default="",
        help="i/k: gate only the i-th of k interleaved slices of the sorts, into its own CSV. The slow "
        "captures (vg_scale x clip, where every EM crawls a flat ridge) do not fit one job's time limit.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.training.thresholds import calculate_gmm_threshold

    cap = Path(args.capture)
    z = np.load(cap)
    sorts = _sorts(z)
    if args.limit:
        sorts = sorts[: args.limit]
    suffix = ""
    if args.chunk:
        i, k = (int(v) for v in args.chunk.split("/"))
        sorts = sorts[i::k]
        suffix = f".chunk{i}of{k}"
    has_labels = any(lb is not None for _, _, lb, _ in sorts)
    rules = dict(R.RULES)
    if not has_labels:
        # A label-free capture is the #3585 corpus: it can only answer the
        # stability question, and it is where the issue's own numbers must
        # reproduce, so it also gets the two reference fits.
        rules.update(_reference_rules())
    if args.rules:
        keep = set(args.rules.split(","))
        rules = {k: v for k, v in rules.items() if k in keep}

    name = cap.stem if cap.parent.name == "corpus" and has_labels else f"old_{cap.stem}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = (out / f"{name}{suffix}_cuts.csv").open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
    w.writeheader()

    for si, (key, x, labels, ident) in enumerate(sorts, 1):
        t_sort = time.monotonic()
        shipped = calculate_gmm_threshold(x.tolist())
        mine = R.fit_shipped(x).cut
        if mine != shipped:
            raise AssertionError(f"{key}: gmm_shipped {mine!r} != calculate_gmm_threshold {shipped!r}")
        n = int(x.size)
        n_pos = int(labels.sum()) if labels is not None else ""
        base = {"capture": name, **ident, "n": n, "n_pos": n_pos}
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        boots = [np.random.default_rng(seed + b).integers(0, n, size=n) for b in range(N_BOOT)]

        def row(
            rule: R.Rule, frame: str, cut: float, ref_adm: "np.ndarray | None", seconds: float = float("nan"), **extra
        ):
            adm = x >= cut
            r = {
                **base,
                "rule": rule.name,
                "family": rule.family,
                "frame": frame,
                "cut": repr(float(cut)),
                "n_adm": int(adm.sum()),
            }
            if labels is not None:
                r["tp"] = int((adm & labels).sum())
                r["fp"] = int((adm & ~labels).sum())
            if ref_adm is not None:
                r["flip_frac"] = f"{float(np.mean(adm != ref_adm)):.6g}"
                union = int((adm | ref_adm).sum())
                r["jaccard_dist"] = f"{(1.0 - int((adm & ref_adm).sum()) / union) if union else 0.0:.6g}"
            if seconds == seconds:
                r["seconds"] = f"{seconds:.6f}"
            r.update(extra)
            w.writerow(r)
            return adm

        for rule in rules.values():
            t0 = time.perf_counter()
            cut = rule.cut(x)
            secs = time.perf_counter() - t0
            extra: dict = {}
            if rule.family in ("mixture", "guarded"):
                res = {
                    "gmm_shipped": R.fit_shipped,
                    "gmm_priorfree": R.fit_shipped,
                    "gmm_converged": R.fit_converged,
                    "gmm_multistart": R.fit_multistart,
                }.get(rule.name, R.fit_shipped)(x)
                extra = {
                    "ashman_d": f"{R.ashman_d(res.fit):.6g}",
                    "w_hi": "" if res.fit is None else f"{res.fit.w_hi:.6g}",
                    "loglik": f"{R.mean_loglik(res.fit, x):.10g}",
                    "n_iter": res.n_iter,
                    "converged": res.converged,
                }
            if rule.family == "tail":
                mu, s = R.bulk_location_scale(x)
                extra["bulk_pred_fp"] = f"{float(n * R._norm_sf(np.asarray([(cut - mu) / s]))[0]):.6g}"
            full = row(rule, "full", cut, None, secs, **extra)
            for label, fn in rule.perturb:
                row(rule, f"opt:{label}", fn(x), full)
            for b, idx in enumerate(boots):
                row(rule, f"boot:{b}", rule.cut(x[idx]), full)

        if labels is not None:
            # Oracle lines, read off the labels: the best any rule could do on
            # this sort under each of the two readings of "a good line".
            order = np.argsort(-x, kind="stable")
            xl, ll = x[order], labels[order]
            tp = np.cumsum(ll)
            fp = np.cumsum(~ll)
            P, N = int(ll.sum()), int((~ll).sum())
            cost = (fp / N) + (1 - tp / P)
            f1 = 2 * tp / (np.arange(1, n + 1) + P)
            oracle = R.Rule("oracle_cost", "oracle", lambda _x: 0.0)
            row(oracle, "full", float(xl[int(np.argmin(cost))]), None)
            oracle_f1 = R.Rule("oracle_f1", "oracle", lambda _x: 0.0)
            row(oracle_f1, "full", float(xl[int(np.argmax(f1))]), None)
        fh.flush()
        print(f"[{si}/{len(sorts)}] {key}: n={n} {time.monotonic() - t_sort:.1f}s", flush=True)

    fh.close()
    print(f"wrote {out / f'{name}{suffix}_cuts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
