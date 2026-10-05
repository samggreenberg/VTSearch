"""Does a larger shortlist pay at 200k pages? The paired arms of #4493's step 2.

Two ``sota_documents.py --stop-log`` arms over the same classes and replicates, differing only in
``--top-k``. Per (class, replicate) the mean over clicks 0..50 of test AP and of the returned set's
F1 at the line; B - A, with a 95% bootstrap interval over classes (both replicates of a resampled
class go together). Also, per arm: test positives beyond the shortlist at the last click, the
dry-run stop's positive-weighted unfound (#4488's bar, 5%), and the seconds per vote.

    python shortlist_arms_documents.py --a <rep1 dir>,<rep2 dir> --b <rep1 dir>,<rep2 dir> --out <dir>

Pre-registered on #4493: B ships if its F1 lower bound is above 0 and its AP lower bound above -0.01.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import stop_grid_documents as sg

N_BOOT = 10_000


def per_session(dirs: str, column: str) -> dict[tuple[str, int], float]:
    """Mean of *column* over each (class, replicate)'s steps."""
    out: dict[tuple[str, int], list[float]] = defaultdict(list)
    for i, d in enumerate(dirs.split(",")):
        for part in d.split("+"):
            with (Path(part) / "steps.csv").open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r[column] not in ("", "nan"):
                        out[(r["class_id"], i + 1)].append(float(r[column]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def retrain_seconds(dirs: str) -> np.ndarray:
    vals = []
    for d in dirs.split(","):
        for part in d.split("+"):
            with (Path(part) / "steps.csv").open(encoding="utf-8") as fh:
                vals += [float(r["retrain_s"]) for r in csv.DictReader(fh) if int(r["v"]) > 0]
    return np.array(vals)


def beyond_shortlist(dirs: str) -> tuple[int, int]:
    out = n = 0
    for d in dirs.split(","):
        for part in d.split("+"):
            with (Path(part) / "positives_final.csv").open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    n += 1
                    out += r["in_shortlist"] != "True"
    return out, n


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict[str, float]:
    keys = sorted(set(a) & set(b))
    diff = {k: b[k] - a[k] for k in keys}
    classes = sorted({c for c, _ in keys})
    by_class = {c: [diff[k] for k in keys if k[0] == c] for c in classes}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(classes), size=len(classes), replace=True)
        boots.append(np.mean([d for i in pick for d in by_class[classes[i]]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "a": float(np.mean([a[k] for k in keys])),
        "b": float(np.mean([b[k] for k in keys])),
        "diff": float(np.mean(list(diff.values()))),
        "lo": float(lo),
        "hi": float(hi),
        "sessions": len(keys),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="arm A: <rep1 dir>[+<part>],<rep2 dir>[+<part>]")
    ap.add_argument("--b", required=True, help="arm B, same layout")
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    la, lb = args.labels.split(",")
    rng = np.random.default_rng(4493)
    res: dict[str, Any] = {
        "ap": paired(per_session(args.a, "ap"), per_session(args.b, "ap"), rng),
        "f1": paired(per_session(args.a, "gate_f1"), per_session(args.b, "gate_f1"), rng),
    }
    lines = [
        f"| mean over clicks 0-50 | {la} | {lb} | {lb} - {la} [95%] | sessions |",
        "|---|---:|---:|---|---:|",
    ]
    for name, key in (("test AP", "ap"), ("returned-set F1 at the line", "f1")):
        r = res[key]
        lines.append(
            f"| {name} | {r['a']:.3f} | {r['b']:.3f} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['sessions']} |"
        )
    lines += ["", f"| per arm | {la} | {lb} |", "|---|---:|---:|"]
    out_a, n_a = beyond_shortlist(args.a)
    out_b, n_b = beyond_shortlist(args.b)
    lines.append(f"| test positives beyond the shortlist at the last click | {out_a / n_a:.1%} | {out_b / n_b:.1%} |")
    stop_a, stop_b = sg.cell_metrics(sg.load_cell(args.a)), sg.cell_metrics(sg.load_cell(args.b))
    lines.append(
        f"| dry-run stop: unfound at the stop (weighted), bar 5% | {stop_a['unfound_weighted']:.1%} | {stop_b['unfound_weighted']:.1%} |"
    )
    sa, sb = retrain_seconds(args.a), retrain_seconds(args.b)
    lines.append(
        f"| seconds per vote, p50 / p90 (this hardware) | {np.median(sa):.1f} / {np.percentile(sa, 90):.1f} | "
        f"{np.median(sb):.1f} / {np.percentile(sb, 90):.1f} |"
    )
    res.update(
        beyond={"a": out_a / n_a, "b": out_b / n_b},
        stop={"a": stop_a, "b": stop_b},
        seconds={
            "a": [float(np.median(sa)), float(np.percentile(sa, 90))],
            "b": [float(np.median(sb)), float(np.percentile(sb, 90))],
        },
        ships=bool(res["f1"]["lo"] > 0 and res["ap"]["lo"] > -0.01),
    )
    lines += [
        "",
        f"Pre-registered rule (F1 lower bound > 0, AP lower bound > -0.01): {'ships' if res['ships'] else 'does not ship'}.",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (args.out / "arms.md").write_text(text, encoding="utf-8")
    (args.out / "arms.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
