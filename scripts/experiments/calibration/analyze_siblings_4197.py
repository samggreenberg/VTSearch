#!/usr/bin/env python3
"""#4197: where do a class's Bad clicks go - and are they its sibling class?

For each arm's pick logs, splits every Bad click by phase group (the opening's
text-sort walk, the learned Hard phase, the atlas's New phase) and reports the
share that hold the class's named sibling (``SIBLINGS``). "Hold" is the image's
largest annotated COCO object, as ``state_of_app/why.py`` reads it, so this is
the same measurement that named the siblings in the first place.

Descriptive, not part of #4197's decision rule: it says whether a knob moved
the clicks it was aimed at.

    python analyze_siblings_4197.py --arm label=RESULTS_DIR [...] --out OUT
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import common

common.setup_env()

import pandas as pd  # noqa: E402

from _cells_paths import side_frame_files  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "state_of_app"))
sys.path.insert(0, str(HERE.parent / "pile"))

#: Each sibling-confused class and the sibling its Bad clicks concentrate on
#: (#4197, from the 2026-09-27 State of the App run).
SIBLINGS = {"knife": "scissors", "laptop": "keyboard", "spoon": "knife", "tv": "remote", "bowl": "toilet"}
#: Phase names per group; ``s<i>`` are a schedule's opening rounds.
OPENING = {"good", "bad", "more", "s0", "s1", "s2", "s3"}


def main(argv: list[str] | None = None) -> int:
    import pile_config as pc  # noqa: PLC0415
    import why  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="label=results_dir (holding cells/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    dom = why._dominant(pc.COCO_ANCHOR_DIR)
    rows = []
    for spec in args.arm:
        label, _, d = spec.partition("=")
        for f in side_frame_files(Path(d) / "cells", "__picks"):
            p = pd.read_csv(f, usecols=["category", "seed", "phase", "picked_id", "picked_label"])
            if p.empty:
                continue
            cls = str(p["category"].iloc[0]).split("@")[0]
            if cls not in SIBLINGS:
                continue
            bads = p[p["picked_label"] == 0]
            for group, mask in (
                ("opening", bads["phase"].isin(OPENING)),
                ("hard", bads["phase"] == "hard"),
                ("new", bads["phase"] == "new"),
            ):
                held = collections.Counter(dom.get(int(i), "") for i in bads.loc[mask, "picked_id"])
                rows.append(
                    {
                        "arm": label,
                        "class": cls,
                        "category": p["category"].iloc[0],
                        "seed": int(p["seed"].iloc[0]),
                        "group": group,
                        "bad_clicks": int(mask.sum()),
                        "sibling_clicks": int(held.get(SIBLINGS[cls], 0)),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no pick logs for the sibling classes - is each --arm a results dir holding cells/?")
    df.to_csv(args.out / "sibling_rows.csv.gz", index=False)
    s = df.groupby(["class", "arm", "group"])[["bad_clicks", "sibling_clicks"]].sum()
    s["sibling_share"] = s["sibling_clicks"] / s["bad_clicks"].where(s["bad_clicks"] > 0)
    s["bad_clicks_per_session"] = df.groupby(["class", "arm", "group"])["bad_clicks"].mean()
    s.reset_index().to_csv(args.out / "sibling_share.csv", index=False, float_format="%.4g")
    print(s.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
