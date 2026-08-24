"""Turn review verdicts into the corrections file the builder merges before banding.

Three sources feed one file, and they are *not* interchangeable:

* **human verdicts** (`ingest_slate.py`) -- the reviewer's Good/Bad on slate
  images, carrying a drawn box when they redrew one;
* **adjudications** -- a second opinion on the pairs where the reviewer and the
  reference disagree, recorded with a note so the reasoning survives;
* **triage flags** -- a model pass over the ranked negatives, which finds
  contaminated negatives efficiently but draws no boxes.

**A correction without a box excludes rather than promotes.** "There is a bus in
this image" fixes a poisoned negative, but it cannot make the image a *positive*
for any band, because a band is a claim about size and no size was measured. The
builder therefore drops it from every cell of that class: not a positive, and no
longer a negative either. That is the whole point of the three-valued design --
the alternative is inventing a box to keep the arithmetic tidy.

**A rejection is not a deletion in the small band.** Boxed review confirms only
~2/3 of sub-patch positives even when the box is drawn for the reviewer, and the
same objects defeat the model, so "not confirmed" there is recorded as exactly
that and the label stands. Above one patch a rejection backed by adjudication
does remove the positive.

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
    ap.add_argument("--verdicts", default=str(base / "verdicts_20260820b.json"))
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
    for v in json.loads(Path(args.verdicts).read_text()):
        key = (int(v["image_id"]), v["class"])
        band = _cell_band(cells.get(key, ""))
        if v["stratum"] in ("boundary", "random"):
            if v["human"] == "present":
                box = v.get("box")
                out[key] = {
                    "image_id": key[0],
                    "class": key[1],
                    "present": True,
                    "boxes": [box] if box else [],
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
                        "source": "human_rebox",
                    }
                    stats["positive_reboxed"] += 1
                else:
                    stats["positive_confirmed"] += 1
                continue
            # Rejected. Small band: not confirmed is not absent.
            a = adj.get(key)
            if band == "small":
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
                if key in out:  # a human already ruled on this pair
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
