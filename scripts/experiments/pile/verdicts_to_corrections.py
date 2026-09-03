"""Turn review verdicts into the corrections file the builder merges before banding.

Three sources feed one file, and they are *not* interchangeable:

* **human verdicts** (`ingest_slate.py`) -- the reviewer's Good/Bad on slate
  images, carrying a drawn box when they redrew one;
* **adjudications** -- a second opinion on the pairs where the reviewer and the
  reference disagree, recorded with a note so the reasoning survives;
* **triage flags** -- a model pass over the ranked negatives, which finds
  contaminated negatives efficiently but draws no boxes.

**Boxes are written NORMALISED, and say so.** A drawn box arrives as the app's
`region_box`, which is already in [0, 1], while VG's and COCO's are in pixels.
The builder normalises every box it merges, so an undeclared correction box was
normalised twice and landed on the frame origin -- 130 of them, taking their
band with them, invisible because the band is derived from the same box
(#3281). Each row therefore carries `box_space`, and `build_pile.py` refuses a
row whose boxes do not match what it declares.

**A correction without a box excludes rather than promotes.** "There is a bus in
this image" fixes a poisoned negative, but it cannot make the image a *positive*
for any band, because a band is a claim about size and no size was measured. The
builder therefore drops it from every cell of that class: not a positive, and no
longer a negative either. That is the whole point of the three-valued design --
the alternative is inventing a box to keep the arithmetic tidy.

**A rejection is not a deletion in the small band -- unless it is definitional.**
Boxed review confirms only ~2/3 of sub-patch positives even when the box is drawn
for the reviewer, and the same objects defeat the model, so "not confirmed" there
is recorded as exactly that and the label stands. Above one patch a rejection
backed by adjudication does remove the positive.

That guard reads a small-band rejection as *"I cannot tell at this size"*, which
is the right default and the wrong one when the adjudicator has named what the
object actually is. Three of the ten ``bicycle@small`` positives are bicycle
pictograms on road signs; they are not bicycles at any resolution, and the guard
made them uncorrectable by a human rejection and an adjudicated one alike
(#3614). An adjudication may therefore carry ``"reason": "definition"``
alongside ``"claude": "absent"``, which removes the positive regardless of band.
Use it only where the identity of the object is settled -- never to force through
a rejection that is really about confirmability, which is what the guard is for.

Usage::

    python verdicts_to_corrections.py --verdicts verdicts.json --triage tri_flags_all.json \
        --adjudication adjudication_ml.json --out corrections.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[corrections] {msg}", flush=True)


def _cell_band(cell: str) -> str:
    return cell.rsplit("@", 1)[1] if "@" in cell else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent / "vgscale-3156"
    ap.add_argument(
        "--verdicts",
        default=f"{base / 'verdicts_20260820b.json'},/exp/sgreenberg/vgscale-3156-labelsets/verdicts_audit_20260825.json",
        help="verdict files, comma-separated; later files win",
    )
    ap.add_argument("--triage", default=str(base / "tri_flags_all.json"))
    ap.add_argument("--adjudication", default=str(base / "adjudication_ml_20260820.json"))
    ap.add_argument("--sheets", default=str(base / "sheets_neg"))
    ap.add_argument("--slates", default=f"{base / 'slates'},{base / 'slates_pos2'}")
    ap.add_argument("--include-maybes", action="store_true", help="apply triage maybes too (default: no)")
    ap.add_argument("--out", default=str(pc.PILE / "corrections.json"))
    args = ap.parse_args()

    # (image, class) -> manifest row, for the band of a reviewed positive.
    cells: dict[tuple[int, str], str] = {}
    for root in args.slates.split(","):
        for man in sorted(Path(root).glob("*/manifest.csv")):
            for r in csv.DictReader(man.open()):
                if r.get("cell"):
                    cells[(int(r["image_id"]), r["class"])] = r["cell"]

    out: dict[tuple[int, str], dict] = {}
    stats: Counter = Counter()

    # --- adjudications first, so a later source cannot silently overrule them
    adj = {}
    if Path(args.adjudication).exists():
        for a in json.loads(Path(args.adjudication).read_text()):
            adj[(int(a["image_id"]), a["class"])] = a

    # --- human verdicts
    # `ruled` records every pair a human decided, in EITHER direction. A triage
    # flag must never overrule one: the audit measured the flags at 0.44
    # precision, so applying an unaudited flag over a human "absent" would
    # inject more error than it removes.
    ruled: set[tuple[int, str]] = set()
    verdicts = []
    for path in args.verdicts.split(","):
        if Path(path).exists():
            verdicts += json.loads(Path(path).read_text())
    for v in verdicts:
        key = (int(v["image_id"]), v["class"])
        ruled.add(key)
        band = _cell_band(cells.get(key, ""))
        # Any stratum that reviews a current NEGATIVE folds in here. Listing
        # them explicitly means a new stratum is ignored rather than
        # mishandled -- safe, but silent, so the list has to be updated when one
        # is added. `redef*` came from re-reviewing a class whose definition
        # changed (`make_definition_reslate.py`).
        if v["stratum"] in ("boundary", "random", "flag", "audit", "redef", "redef_fresh"):
            if v["human"] == "present":
                box = v.get("box")
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": True,
                    "boxes": [box] if box else [],
                    "box_space": pc.CORRECTION_BOX_SPACE,
                    "source": "human_review",
                }
                stats["negative_fixed" if box else "negative_excluded"] += 1
            continue
        if v["stratum"] == "positive_boxed":
            if v["human"] == "present":
                box = v.get("box")
                if box:  # reviewer redrew it: the box, hence the band, changes
                    out[key] = {
                        "image_id": key[0],
                        "class": key[1],
                        "present": True,
                        "boxes": [box],
                        "box_space": pc.CORRECTION_BOX_SPACE,
                        "source": "human_rebox",
                    }
                    stats["positive_reboxed"] += 1
                else:
                    stats["positive_confirmed"] += 1
                continue
            # Rejected. Small band: not confirmed is not absent.
            a = adj.get(key)
            # ...unless the rejection is DEFINITIONAL. The band guard exists
            # because a small object is hard to confirm, so a rejection there is
            # ambiguous between "absent" and "I cannot tell at 26 px". That
            # ambiguity does not arise when the adjudicator names *what the
            # object is*: a bicycle pictogram on a road sign is not a bicycle at
            # any size, and no amount of resolution would change the answer.
            # Without this branch such a positive is uncorrectable -- the guard
            # swallows the human rejection and the adjudicated one alike (#3614).
            if a and a.get("claude") == "absent" and a.get("reason") == "definition":
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": False,
                    "boxes": [],
                    "source": "human_reject+adjudicated_definition",
                    "note": a.get("note", ""),
                }
                stats["positive_removed_definitional"] += 1
            elif band == "small":
                stats["small_unconfirmed"] += 1
            elif a and a["claude"] == "absent":
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": False,
                    "boxes": [],
                    "source": "human_reject+adjudicated",
                    "note": a.get("note", ""),
                }
                stats["positive_removed"] += 1
            elif a and a["claude"] == "present":
                stats["rejection_overturned"] += 1
            else:
                stats["rejection_unadjudicated"] += 1
            continue
        stats[f"IGNORED_unknown_stratum:{v['stratum']}"] += 1

    # --- triage flags: contaminated negatives, no boxes, so they exclude
    if Path(args.triage).exists():
        flags = json.loads(Path(args.triage).read_text())
        for cls, kinds in flags.items():
            idx_path = Path(args.sheets) / cls.replace(" ", "_") / "index.json"
            if not idx_path.exists():
                log(f"  no sheet index for {cls}; skipping its flags")
                continue
            idx = {(r["sheet"], r["tile"]): r["image_id"] for r in json.loads(idx_path.read_text())}
            wanted = list(kinds["definite"]) + (list(kinds["maybe"]) if args.include_maybes else [])
            for sheet, tile in wanted:
                iid = idx.get((sheet, tile))
                if iid is None:
                    continue
                key = (iid, cls)
                if key in ruled:  # a human ruled on this pair, either way
                    stats["triage_deferred_to_human"] += 1
                    continue
                out[key] = {
                    "image_id": iid,
                    "class": cls,
                    "present": True,
                    "boxes": [],
                    "source": "claude_triage",
                }
                stats["negative_excluded_by_triage"] += 1

    rows = sorted(out.values(), key=lambda r: (r["class"], r["image_id"]))
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")

    print(f"\n{len(rows)} corrections written to {args.out}\n")
    for k, v in sorted(stats.items()):
        print(f"   {k:<32}{v:>6}")
    boxed = sum(1 for r in rows if r["boxes"])
    print(f"\n   {'of which carry a box':<32}{boxed:>6}  (can move an image between bands)")
    print(f"   {'excluded, no box':<32}{len(rows) - boxed:>6}  (dropped from every cell of that class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
