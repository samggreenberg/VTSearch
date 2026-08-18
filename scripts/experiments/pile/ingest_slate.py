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
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()


def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def load_manifests(slates: Path) -> tuple[dict[tuple[int, str], dict], dict[str, str]]:
    """``({(image_id, class): row}, {class: folder name})`` over every slate."""
    out: dict[tuple[int, str], dict] = {}
    folders: dict[str, str] = {}
    for man in sorted(slates.glob("*/manifest.csv")):
        with man.open() as fh:
            for row in csv.DictReader(fh):
                out[(int(row["image_id"]), row["class"])] = row
                folders[row["class"]] = man.parent.name
    return out, folders


def class_of(path: Path, elements: list[dict], folders: dict[str, str], explicit: str) -> str:
    """Which class this export is a review of.

    An export **cannot** be attributed by image id: the slates share images
    (801 of 3,600 rows are an image that appears under a second class), so a
    Good vote in the `bus` dataset would otherwise be recorded as a `dog`
    verdict too. The slate folder is what disambiguates, read from the
    importer's own origin, then from the file name, and never guessed.
    """
    if explicit:
        if explicit not in folders:
            raise SystemExit(f"--class {explicit!r} is not one of {sorted(folders)}")
        return explicit
    blob = " ".join(json.dumps(el.get("origin") or "") + " " + str(el.get("origin_name") or "") for el in elements[:50])
    hits = {c for c, folder in folders.items() if f"/{folder}/" in blob or f"/{folder}" in blob}
    if len(hits) == 1:
        return hits.pop()
    stem = path.stem.lower()
    hits = {c for c, folder in folders.items() if folder.lower() in stem}
    if len(hits) == 1:
        return hits.pop()
    raise SystemExit(
        f"{path.name}: cannot tell which class this export reviews "
        f"(origin paths and file name match {sorted(hits) or 'nothing'}). Pass --class."
    )


def read_api(base: str, detector: str) -> list[dict]:
    """One detector's saved votes, in the same shape as a file export.

    ``GET /api/detectors/<name>/labels-detail`` already returns the two things a
    verdict needs -- the file name (which is the VG image id) and the
    ``region_box`` drawn on a Good vote -- so pulling straight from the running
    app skips the export dialog entirely. The reviewer votes; nothing else is
    asked of them.
    """
    url = f"{base.rstrip('/')}/api/detectors/{urllib.parse.quote(detector)}/labels-detail"
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - caller-supplied http(s) base
        data = json.load(r)
    out = []
    for label in ("good", "bad"):
        for el in data.get(label) or []:
            out.append(
                {
                    "filename": el.get("filename") or el.get("origin_name"),
                    "label": label,
                    "region_box": el.get("region_box"),
                }
            )
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
    ap.add_argument("--export", default="", help="exported JSON (glob allowed)")
    ap.add_argument(
        "--api",
        default="",
        help="pull votes straight from a running VTSearch instead of a file export, "
        "e.g. --api http://rack7n03:11850 (one detector per class, named after it)",
    )
    ap.add_argument("--slates", default=str(pc.PILE / "slates"), help="slate dir from make_audit_slate.py")
    ap.add_argument("--out", default="", help="write the verdict rows here")
    ap.add_argument("--class", dest="klass", default="", help="the class this export reviews (else inferred)")
    args = ap.parse_args()

    manifests, folders = load_manifests(Path(args.slates))
    if not manifests:
        raise SystemExit(f"no manifests under {args.slates}; run make_audit_slate.py first")
    log(f"{len(manifests)} slate entries over {len({c for _, c in manifests})} classes")

    if not args.export and not args.api:
        raise SystemExit("pass --export <file/glob> or --api <base-url>")

    # (label of the source, its elements, the class it reviews)
    sources: list[tuple[str, list[dict], str]] = []
    if args.api:
        for c in sorted(folders):
            if args.klass and c != args.klass:
                continue
            els = read_api(args.api, c)
            sources.append((f"api:{c}", els, c))
    for path in sorted(glob.glob(args.export)) if args.export else []:
        els = read_export(Path(path))
        sources.append((Path(path).name, els, class_of(Path(path), els, folders, args.klass)))

    verdicts: list[dict] = []
    unmatched = 0
    for source, elements, c in sources:
        for el in elements:
            name = Path(el.get("filename") or el.get("origin_name") or "").stem
            if not name.isdigit():
                unmatched += 1
                continue
            iid = int(name)
            row = manifests.get((iid, c))
            if row is None:
                unmatched += 1
                continue
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
                    "export": source,
                }
            )
        log(f"  {source}: {len(elements)} labelled elements, class {c!r}")
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
