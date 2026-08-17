"""Turn a VTSearch label export back into verdict rows, and score the ground truth.

The other half of ``make_audit_slate.py``. A ``server_folder`` import → vote →
``server_json_file`` export round-trip already emits everything a verdict needs:
the file name identifies the VG image (the slate names files ``<image_id>.jpg``),
``label`` is the human's Good/Bad, and ``region_box`` carries the box drawn on a
Good vote, normalised.

**Verdicts, not corrections.** Every reviewed ``(image, class)`` pair becomes a
row whether or not it disagrees with COCO. Corrections are then derived from the
disagreements, and review *coverage* falls out for free — without it, "no bus
here" is indistinguishable from "nobody looked", and every rate computed
afterwards is biased by an unknown amount.

**The rate comes from the random stratum alone.** The boundary stratum is chosen
to find errors, so its error rate is not the pool's error rate and averaging the
two together produces a number that means nothing. Both are reported, labelled,
and never pooled.

Usage::

    python ingest_slate.py --export ~/exports/bus.json --slates /expscratch/$USER/vgscale-3156/slates
    python ingest_slate.py --export 'exports/*.json' --slates ... --out verdicts.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def load_manifests(slates: Path) -> dict[tuple[int, str], dict]:
    """``{(image_id, class): manifest row}`` over every slate on disk."""
    out: dict[tuple[int, str], dict] = {}
    for man in sorted(slates.glob("*/manifest.csv")):
        with man.open() as fh:
            for row in csv.DictReader(fh):
                out[(int(row["image_id"]), row["class"])] = row
    return out


def read_export(path: Path) -> list[dict]:
    """The labelled elements of one export, whichever shape it was written in."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "labels" in data:
        return list(data["labels"])
    if isinstance(data, list):
        return list(data)
    raise SystemExit(f"{path}: not a label export (no 'labels' key, not a list)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="exported JSON (glob allowed)")
    ap.add_argument("--slates", default=str(pc.PILE / "slates"), help="slate dir from make_audit_slate.py")
    ap.add_argument("--out", default="", help="write the verdict rows here")
    args = ap.parse_args()

    manifests = load_manifests(Path(args.slates))
    if not manifests:
        raise SystemExit(f"no manifests under {args.slates}; run make_audit_slate.py first")
    log(f"{len(manifests)} slate entries over {len({c for _, c in manifests})} classes")

    verdicts: list[dict] = []
    unmatched = 0
    for path in sorted(glob.glob(args.export)):
        elements = read_export(Path(path))
        for el in elements:
            name = Path(el.get("filename") or el.get("origin_name") or "").stem
            if not name.isdigit():
                unmatched += 1
                continue
            iid = int(name)
            # One export is one detector, i.e. one class; the slate says which.
            hits = [(i, c) for (i, c) in manifests if i == iid]
            if not hits:
                unmatched += 1
                continue
            for _, c in hits:
                row = manifests[(iid, c)]
                verdicts.append(
                    {
                        "image_id": iid,
                        "class": c,
                        "stratum": row["stratum"],
                        "human": "present" if el.get("label") == "good" else "absent",
                        "reference": row["reference"],
                        "exhaustive": row["exhaustive"],
                        "box": el.get("region_box"),
                        "text_score": float(row["text_score"]),
                        "export": Path(path).name,
                    }
                )
        log(f"  {Path(path).name}: {len(elements)} labelled elements")
    if unmatched:
        log(f"  WARNING {unmatched} exported elements matched no slate entry")

    # Per stratum, per direction. Never pooled across strata.
    by: dict[tuple[str, str], int] = defaultdict(int)
    for v in verdicts:
        by[(v["stratum"], f"{v['reference']}->{v['human']}")] += 1

    # The calibration: on images COCO already settled, a disagreement is the
    # reviewer's error, not a correction. Reported separately -- folding it into
    # the correction counts would let annotator noise masquerade as label noise.
    cal = [v for v in verdicts if v["exhaustive"] == "yes"]
    if cal:
        wrong = sum(1 for v in cal if v["human"] != v["reference"])
        print(
            f"\ncalibration: {len(cal)} pairs with an exhaustive reference, {wrong} disagreements "
            f"({wrong / len(cal):.3f}) -- reviewer error rate, not label noise"
        )

    print(f"\n{len(verdicts)} verdicts\n")
    hdr = f"{'stratum':<10}{'agree':>8}{'ref absent, human present':>28}{'ref present, human absent':>28}{'rate':>9}"
    print(hdr)
    print("-" * len(hdr))
    for stratum in ("random", "boundary", "positive"):
        agree = by[(stratum, "absent->absent")] + by[(stratum, "present->present")]
        miss = by[(stratum, "absent->present")]
        over = by[(stratum, "present->absent")]
        n = agree + miss + over
        if not n:
            continue
        rate = (miss + over) / n
        note = "" if stratum == "random" else "  (biased by design)"
        print(f"{stratum:<10}{agree:>8}{miss:>28}{over:>28}{rate:>9.3f}{note}")
    print(
        "\nThe residual error rate of the ground truth is the RANDOM row only."
        "\nThe boundary row is chosen to surface errors and says nothing about the pool."
    )

    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
