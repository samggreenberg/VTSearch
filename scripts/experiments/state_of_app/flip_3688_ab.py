#!/usr/bin/env python
"""#3688: how much does one wrong vote change a run's outcome, against the run's own seed-to-seed spread?

Pairs of session sets on the same COCO Better cells and seeds (``flip_3688.sh``): an arm with one wrong vote
(``CALIB_FLIP_VOTE=good:K`` / ``bad:K``) or with the owner's record put back (Arm 1, ``CALIB_HOLD_OUT``), and its
control. A pair's picks agree up to the wrong vote, so the paired difference is the effect of that vote alone.

The outcome is read as #5473 read it (``prior_4776.read_runs``): Test's untagged rows' ``fbeta`` at every vote
1..150, carried between rows, 0 before the first; the stop is the first vote Autopilot says Done, or 150. Per pair:

* **the paired effect**: arm minus control, F at each arm's own stop (what the user leaves with), F at votes 25, 50,
  100 and 150, the session mean over 1-150, and the stop's click; mean and SE clustered on the class;
* **the denominator**: per cell (class@band), the control's seed-to-seed SD of the same read over its seeds. The
  headline is the share of runs whose |effect| exceeds one (and two) of its cell's SD, and the ratio of the effect's
  spread to the seed spread. Comparing one flipped run with one clean run would measure the seed spread and call it
  sensitivity; this compares the flip with that spread instead;
* **where the vote landed**: the vote the wrong answer went in at, and its phase (arms with ``picked_vote``);
* **exposure** (``--record``, Arm 1): how many of the record's images the control serves by votes 10, 25, 50 and
  150, per run.

Splits by band (``@small`` / ``@medium`` / ``@large``) are in the summary; the per-class table is
``flip_3688_<stem>_by_class.csv``.

    python flip_3688_ab.py --pair NAME=ARM_CELLS::CONTROL_CELLS ... --out <dir> [--stem S] [--record JSON] [--docs D]
                           [--reseed CONTROL_CELLS]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fhibe"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cells_paths import side_frame_files  # noqa: E402
from prior_4776 import load  # noqa: E402
from skip_diversity_5473 import load_phases, stop_of  # noqa: E402
from weak_check_4797 import departure, se_cat  # noqa: E402

HORIZON = 150
AT = (25, 50, 100, 150)
EXPOSURE_AT = (10, 25, 50, 150)
BANDS = ("large", "medium", "small")


def read_votes(path: str) -> dict:
    """Per run in one pick log: its picks by vote, and the vote (and phase) of a wrong answer, if any."""
    try:
        pk = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}
    pk = pk[pk["t"] <= HORIZON]
    pk["phase"] = pk["phase"].fillna("").astype(str)
    out = {}
    for (ds, cat, seed), r in pk.groupby(["dataset", "category", "seed"], sort=False):
        r = r.assign(vote=r["t"] - r.groupby("t")["t"].transform("size") + r.groupby("t").cumcount() + 1)
        r = r.sort_values("vote", kind="stable")
        wrong = r[r["picked_vote"] != r["picked_label"]] if "picked_vote" in r else r.iloc[:0]
        out[(ds, cat, int(seed))] = {
            "picks": r["picked_id"].to_numpy(),
            "flip_vote": int(wrong["vote"].iloc[0]) if len(wrong) else None,
            "flip_phase": str(wrong["phase"].iloc[0]) if len(wrong) else "",
            "n_wrong": len(wrong),
        }
    return out


def load_votes(cells: str, workers: int) -> dict:
    files = [str(f) for f in side_frame_files(cells, "__picks")]
    with ProcessPoolExecutor(workers) as pool:
        return {k: v for got in pool.map(read_votes, files, chunksize=8) for k, v in got.items()}


def band_of(key: tuple) -> str:
    return key[1].rsplit("@", 1)[-1]


def reads(F: np.ndarray, stop: int) -> dict:
    """The outcome reads of one run: F at its stop, at fixed votes, and the session mean."""
    return {"F_stop": F[stop - 1], **{f"F{v}": F[v - 1] for v in AT}, "F_mean": float(F[:HORIZON].mean())}


def seed_sd(keys: list, C: dict, PC: dict) -> pd.DataFrame:
    """Per cell (dataset, class@band): the control's seed-to-seed SD of every read, over all its seeds."""
    rows = [{"cell": k[:2], **reads(C[k]["F"], stop_of(PC[k])), "stop": stop_of(PC[k])} for k in keys]
    return pd.DataFrame(rows).groupby("cell").std(ddof=1)


def effect(keys: list, A: dict, C: dict, PA: dict, PC: dict, sd: pd.DataFrame) -> dict:
    """Arm minus control over *keys*: the paired effect, and how it compares with the cell's seed spread."""
    row: dict = {"runs": len(keys)}
    if not keys:
        return row
    ra = pd.DataFrame([{**reads(A[k]["F"], stop_of(PA[k])), "stop": stop_of(PA[k])} for k in keys])
    rc = pd.DataFrame([{**reads(C[k]["F"], stop_of(PC[k])), "stop": stop_of(PC[k])} for k in keys])
    cell_sd = sd.loc[[k[:2] for k in keys]].reset_index(drop=True)
    for col in ("F_stop", "F25", "F50", "F100", "F150", "F_mean", "stop"):
        d = (ra[col] - rc[col]).to_numpy(float)
        s = cell_sd[col].to_numpy(float)
        ok = np.isfinite(s) & (s > 0)
        row[f"ctl_{col}"] = float(rc[col].mean())
        row[f"d_{col}"], row[f"d_{col}_se"] = float(d.mean()), se_cat(keys, d)
        row[f"seed_sd_{col}"] = float(np.sqrt(np.nanmean(s**2)))
        row[f"over_1sd_{col}"] = float(np.mean(np.abs(d[ok]) > s[ok])) if ok.any() else float("nan")
        row[f"over_2sd_{col}"] = float(np.mean(np.abs(d[ok]) > 2 * s[ok])) if ok.any() else float("nan")
        row[f"worse_1sd_{col}"] = float(np.mean(d[ok] < -s[ok])) if ok.any() else float("nan")
        # The spread of the effect against the seed spread: 1 means one wrong vote moves a run as far as a new seed.
        row[f"sd_ratio_{col}"] = float(np.std(d, ddof=1) / row[f"seed_sd_{col}"]) if len(d) > 1 else float("nan")
        row[f"unchanged_{col}"] = float(np.mean(np.abs(d) < 1e-9))
    return row


def curves(name: str, keys: list, A: dict, C: dict, out: dict) -> None:
    """The paired effect at every vote 1-150: mean and class-clustered SE."""
    D = np.array([A[k]["F"][:HORIZON] - C[k]["F"][:HORIZON] for k in keys], float)
    for v in range(1, HORIZON + 1):
        d = D[:, v - 1]
        out["curves"].append({"set": name, "vote": v, "d_F": float(d.mean()), "se": se_cat(keys, d)})


def reseed(name: str, ctl_dir: Path, workers: int, out: dict) -> None:
    """The yardstick: each control run against the same cell's next seed (cyclic), read exactly as a pair is.

    Two clean runs of one cell differ with SD sqrt(2) x the seed SD, so this row's ``sd_ratio`` is ~1.41 and its
    ``over_1sd`` ~48%: an arm that matches it moves a run as far as drawing a new seed does.
    """
    C, PC = load(str(ctl_dir), None, workers), load_phases(str(ctl_dir), workers)
    keys = sorted(set(C) & set(PC))
    sd = seed_sd(keys, C, PC)
    seeds: dict = {}
    for k in keys:
        seeds.setdefault(k[:2], []).append(k[2])
    nxt = {k: (*k[:2], (s := sorted(seeds[k[:2]]))[(s.index(k[2]) + 1) % len(s)]) for k in keys}
    pairs = [k for k in keys if nxt[k] != k]
    A, PA = {k: C[nxt[k]] for k in pairs}, {k: PC[nxt[k]] for k in pairs}
    out["summary"].append({"set": name, "band": "all", **effect(pairs, A, C, PA, PC, sd)})
    for b in BANDS:
        kb = [k for k in pairs if band_of(k) == b]
        out["summary"].append({"set": name, "band": b, **effect(kb, A, C, PA, PC, sd)})
    for k in pairs:
        ra, rc = reads(A[k]["F"], stop_of(PA[k])), reads(C[k]["F"], stop_of(PC[k]))
        out["per_run"].append(
            {"set": name, "category": k[1], "seed": k[2], "stop_ctl": stop_of(PC[k]), "stop_arm": stop_of(PA[k]),
             "F_stop_ctl": rc["F_stop"], "d_F_stop": ra["F_stop"] - rc["F_stop"], "d_F150": ra["F150"] - rc["F150"],
             "seed_sd_F_stop": sd.loc[[k[:2]]].iloc[0]["F_stop"]}
        )  # fmt: skip
    a = out["summary"][-1 - len(BANDS)]
    print(f"{name}: {a['runs']} seed pairs; |d|>1sd {a['over_1sd_F_stop']:.0%}, sd ratio {a['sd_ratio_F_stop']:.2f}")


def compare(name: str, arm_dir: Path, ctl_dir: Path, workers: int, record: dict | None, out: dict) -> None:
    A, C = load(str(arm_dir), None, workers), load(str(ctl_dir), None, workers)
    PA, PC = load_phases(str(arm_dir), workers), load_phases(str(ctl_dir), workers)
    VA, VC = load_votes(str(arm_dir), workers), load_votes(str(ctl_dir), workers)
    keys = sorted(set(A) & set(C) & set(PA) & set(PC) & set(VA) & set(VC))
    ctl_keys = sorted(set(C) & set(PC))
    if not keys:
        print(f"{name}: no paired runs", file=sys.stderr)
        return
    sd = seed_sd(ctl_keys, C, PC)
    # Pairing: a pair's picks may part only after the arm's wrong vote (a flip arm) -- never before it.
    early = [
        k
        for k in keys
        if VA[k]["flip_vote"] is not None
        and (dv := departure(VA[k]["picks"], VC[k]["picks"])) is not None
        and dv < VA[k]["flip_vote"]
    ]
    flipped = [k for k in keys if VA[k]["flip_vote"] is not None]
    fv = np.array([VA[k]["flip_vote"] for k in flipped], float)
    base = {
        "set": name,
        "control_only": len(set(C) - set(A)),
        "arm_only": len(set(A) - set(C)),
        "parted_before_flip": len(early),
        "flipped_share": len(flipped) / len(keys),
        "multi_wrong": int(sum(VA[k]["n_wrong"] > 1 for k in keys)),
        "flip_vote_median": float(np.median(fv)) if len(fv) else float("nan"),
        "flip_vote_p90": float(np.percentile(fv, 90)) if len(fv) else float("nan"),
        "flip_before_ctl_stop": float(np.mean([VA[k]["flip_vote"] <= stop_of(PC[k]) for k in flipped]))
        if flipped
        else float("nan"),
    }
    out["summary"].append({**base, "band": "all", **effect(keys, A, C, PA, PC, sd)})
    # Only the runs whose wrong vote went in before the clean run stopped: a vote after the stop cannot move F there.
    landed = [k for k in flipped if VA[k]["flip_vote"] <= stop_of(PC[k])]
    out["summary"].append({"set": name, "band": "landed", **effect(landed, A, C, PA, PC, sd)})
    for b in BANDS:
        kb = [k for k in keys if band_of(k) == b]
        out["summary"].append({"set": name, "band": b, **effect(kb, A, C, PA, PC, sd)})
    for cls in sorted({k[1].rsplit("@", 1)[0] for k in keys}):
        kc = [k for k in keys if k[1].rsplit("@", 1)[0] == cls]
        e = effect(kc, A, C, PA, PC, sd)
        out["by_class"].append(
            {"set": name, "class": cls, **{c: e[c] for c in e if c.endswith("F_stop") or c == "runs"}}
        )
    phases = pd.Series([VA[k]["flip_phase"] for k in flipped]).value_counts(normalize=True)
    for ph, share in phases.items():
        out["phases"].append({"set": name, "phase": ph, "share": float(share)})
    for k in keys:
        ra, rc = reads(A[k]["F"], stop_of(PA[k])), reads(C[k]["F"], stop_of(PC[k]))
        cs = sd.loc[[k[:2]]].iloc[0]
        out["per_run"].append(
            {
                "set": name,
                "category": k[1],
                "seed": k[2],
                "flip_vote": VA[k]["flip_vote"],
                "flip_phase": VA[k]["flip_phase"],
                "stop_ctl": stop_of(PC[k]),
                "stop_arm": stop_of(PA[k]),
                "F_stop_ctl": rc["F_stop"],
                "d_F_stop": ra["F_stop"] - rc["F_stop"],
                "d_F150": ra["F150"] - rc["F150"],
                "seed_sd_F_stop": cs["F_stop"],
            }
        )
    curves(name, keys, A, C, out)
    if record is not None:
        for arm_name, V in (("arm", VA), ("control", VC)):
            served = []
            for k in keys:
                held = set(record.get(k[1].rsplit("@", 1)[0], []))
                picks = V[k]["picks"]
                served.append([int(np.isin(picks[:v], list(held)).sum()) if held else 0 for v in EXPOSURE_AT])
            s = np.array(served, float)
            out["exposure"].append(
                {
                    "set": name,
                    "side": arm_name,
                    "runs": len(keys),
                    **{f"served_by_{v}": float(s[:, i].mean()) for i, v in enumerate(EXPOSURE_AT)},
                    **{f"any_by_{v}": float((s[:, i] > 0).mean()) for i, v in enumerate(EXPOSURE_AT)},
                }
            )
    a = out["summary"][-2 - len(BANDS)]
    print(
        f"{name}: {a['runs']} runs, flip vote median {base['flip_vote_median']:.0f}, parted early {len(early)}; "
        f"F@stop {a['ctl_F_stop']:.3f} {a['d_F_stop']:+.4f}±{a['d_F_stop_se']:.4f}, |d|>1sd {a['over_1sd_F_stop']:.0%} "
        f">2sd {a['over_2sd_F_stop']:.0%}, sd ratio {a['sd_ratio_F_stop']:.2f}; F150 {a['d_F150']:+.4f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", action="append", required=True, help="NAME=ARM_CELLS::CONTROL_CELLS")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stem", default="binary")
    ap.add_argument("--record", type=Path, help="{class: [media ids]}: exposure of the record's images")
    ap.add_argument("--docs", type=Path, help="also write the tables here")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reseed", type=Path, help="control cells: add the seed-vs-next-seed yardstick row")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record = json.loads(args.record.read_text()) if args.record else None
    out: dict[str, list] = {"summary": [], "by_class": [], "phases": [], "per_run": [], "exposure": [], "curves": []}
    for spec in args.pair:
        name, _, rest = spec.partition("=")
        arm_dir, _, ctl_dir = rest.partition("::")
        compare(name, Path(arm_dir), Path(ctl_dir), args.workers, record, out)
    if args.reseed:
        reseed("reseed", args.reseed, args.workers, out)
    for name, rows in out.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).round(4)
        df.to_csv(args.out / f"flip_3688_{args.stem}_{name}.csv", index=False)
        if args.docs and name != "per_run":
            args.docs.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.docs / f"flip_3688_{args.stem}_{name}.csv", index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
