#!/usr/bin/env python3
"""Turn the banked per-class pass into verdicts `verdicts_to_corrections.py` can read (#3926).

The `vg_scale` build reads only `corrections.json`, and the per-class pass (#3720)
lives in ``human_record/LABELSETS__*``, which no converter input covered -- so
3,859 confirmed positives sat banked and outside the benchmark. This writes them as
``stratum: "flag"`` verdicts (every pass image was a current negative), which is
the recipe #3926 applied:

* **slate** and **below-cut** labelsets, only where the labelset's rule is the rule
  in force (a stale one answers a superseded question and is refused, not skipped);
* **a Good carries a box**: the reviewer's redraw if there is one, otherwise the
  OWLv2 box the reviewer was shown and confirmed (``slates.json``). Without it a
  boxless ``present`` becomes an *exclusion*, and the pass would remove images
  instead of adding positives -- "a loose box costs a stratum, not a label";
* **the prominence triage overrides the slate** for the pairs it re-asked
  (``LABELSETS__prominent__*``): a redraw replaces the stored box, a Good keeps it,
  a Bad makes the pair absent. Its boxes are already in photo coordinates.

**A stored box is clipped to the image.** OWLv2 predicts boxes that run past the
frame -- 358 of the applied ones, by at most 1.4% -- and the build refuses any
coordinate outside [0, 1] as pixel space (pilebuild.corrections). The render the
reviewer confirmed already clipped them (slate_render.inset_crop), so the clipped
box is the box they saw. The first application missed this and the rebuild's own
check caught it.

Rule stamps are carried from the labelset, never re-derived (#3814).

Usage: prefer ``regenerate_corrections.py``, which runs all three steps and
diffs the result against the committed copy -- this step alone, or steps 1 and 2
without 3, produces a plausible file that is quietly wrong (#4007). The chain it
composes, for the times one step is wanted by hand::

    python pass_verdicts.py --out pass.json
    python verdicts_to_corrections.py --verdicts "<the three defaults>,pass.json" --out scratch.json
    python apply_recheck.py --corrections scratch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pile_config as pc  # noqa: E402

HR = Path(__file__).resolve().parent / "human_record"


def _iid(el: dict) -> int:
    return int(str(el["filename"]).split(".")[0])


def _clipped(box: list[float] | None) -> list[float] | None:
    """*box* clipped to [0, 1]; refuses one with no area left inside the image."""
    if box is None:
        return None
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"stored box {box} has no area inside the image")
    return [x0, y0, x1, y1] if [x0, y0, x1, y1] != [float(v) for v in box] else box


def build(slates: dict) -> tuple[list[dict], Counter, Counter]:
    """``(verdicts, by_source, prominence_overrides)``; raises on a stale rule or a duplicate pair."""
    out: dict[tuple[int, str], dict] = {}
    tally: Counter = Counter()
    for kind in ("slate", "belowcut"):
        for f in sorted(HR.glob(f"LABELSETS__{kind}__*.json")):
            d = json.loads(f.read_text())
            cls = d["class"]
            if d.get("rule") != pc.review_name(cls):
                raise SystemExit(f"{f.name}: voted under {d.get('rule')!r}, rule in force is {pc.review_name(cls)!r}")
            owl = (
                {int(r["image_id"]): _clipped(r.get("box")) for r in slates.get(cls, {}).get("rows", [])}
                if kind == "slate"
                else {}
            )
            stamp = {k: d[k] for k in ("rule", "rule_digest") if d.get(k)}
            for side, human in (("good", "present"), ("bad", "absent")):
                for el in d.get(side, []):
                    iid, region = _iid(el), el.get("region_box")
                    if (iid, cls) in out:
                        raise SystemExit(f"duplicate pair {(iid, cls)} in {f.name}")
                    box = region or (owl.get(iid) if human == "present" else None)
                    out[(iid, cls)] = {
                        "image_id": iid,
                        "class": cls,
                        **stamp,
                        "stratum": "flag",
                        "human": human,
                        "export": f.name,
                        "box": box,
                    }
                    tally[(kind, human, "reviewer_box" if region else ("owl_box" if owl.get(iid) else "no_box"))] += 1

    over: Counter = Counter()
    for f in sorted(HR.glob("LABELSETS__prominent__*.json")):
        d = json.loads(f.read_text())
        cls = d["class"]
        if d.get("rule") != pc.review_name(cls):
            raise SystemExit(f"{f.name}: voted under a superseded rule")
        owl = {int(r["image_id"]): _clipped(r.get("box")) for r in slates[cls]["rows"]}
        for side, human in (("good", "present"), ("bad", "absent")):
            for el in d[side]:
                key = (_iid(el), cls)
                prev = out.get(key)
                # The triage only ever re-asked a stored-box Good; anything else means
                # the labelsets and this recipe have drifted apart.
                if not (prev and prev["human"] == "present" and prev["box"] == owl.get(key[0])):
                    raise SystemExit(f"{f.name}: {key} was not a stored-box Good on the slate")
                region = el.get("region_box")
                box = (region or owl[key[0]]) if human == "present" else None
                out[key] = {**prev, "human": human, "export": f.name, "box": box}
                over[(cls, human, "redrawn" if region else "stored_box")] += 1
    return list(out.values()), tally, over


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The record, not scratch: #4007 committed this file precisely because the
    # scratch copy was its only home and a tidy-up would have taken an input to
    # 3,837 corrections with it. A default pointing back at scratch would keep
    # the chain un-runnable from a fresh checkout.
    ap.add_argument("--slates", type=Path, default=HR / "VLM3720__slates.json")
    ap.add_argument("--out", required=True, help="never a live file: this feeds the converter, not the build")
    args = ap.parse_args()
    verdicts, tally, over = build(json.loads(Path(args.slates).read_text()))
    Path(args.out).write_text(json.dumps(verdicts) + "\n")
    print(f"{len(verdicts)} verdicts -> {args.out}")
    for k, n in sorted(tally.items()):
        print(f"  {k[0]:<9} {k[1]:<8} {k[2]:<13} {n:>6}")
    print("prominence overrides:")
    for k, n in sorted(over.items()):
        print(f"  {k[0]:<10} {k[1]:<8} {k[2]:<11} {n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
