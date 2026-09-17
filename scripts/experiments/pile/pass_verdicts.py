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

Rule stamps are carried from the labelset, never re-derived (#3814).

Usage (then ALWAYS ``apply_recheck.py`` on the same file -- see #3926)::

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
                {int(r["image_id"]): r.get("box") for r in slates.get(cls, {}).get("rows", [])}
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
        owl = {int(r["image_id"]): r.get("box") for r in slates[cls]["rows"]}
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
    ap.add_argument("--slates", default="/expscratch/sgreenberg/vlm-3720/slates/slates.json")
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
