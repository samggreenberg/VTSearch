"""Rank VG categories by how well they support a *same-class-across-bands* study.

The old ``vg_box_{small,medium,large}`` sets banded each category by its
**median** voted-box area, so a category landed in exactly one band and the
three sets carried disjoint vocabularies (`nose`/`glasses`/`watch` against
`fence`/`hill`/`lady`). A small-vs-large cost difference therefore confounded
box size with class identity, which is not the question anyone wanted asked.

The question worth asking is "how well can we find buses in the middleground?",
which needs one class list *C* held fixed while only the size varies. For a
class to serve that, VG has to hold enough images of it at **every** size:

* **positives** for band *B* -- images whose voted box for *c* falls in *B*;
* **negatives** -- images holding no instance of *c* at any size;
* and, silently, an **excluded** remainder -- images holding *c* at some *other*
  size, which are neither (there is a bus, so they are not negatives; it is the
  wrong size, so they are not positives for this band).

This script reads ``scan_vg_boxes.py``'s output and reports which categories
clear a per-band floor, so *C* is chosen from measured supply rather than from
intuition about what "ought" to appear at every distance.

Pure stdlib and importless of ``vtscore``: the scan is a 108k-image job that
needs the VG source, but ranking its JSON is a laptop operation and should not
require the cluster.

**What this cannot tell you.** Supply here is counted from VG's own
annotations, and those are exactly what issue #3156 puts under suspicion --
`498326.jpg` is annotated `car, clouds` and has a bus front and centre. So a
per-band count is a lower bound on true positive supply and the negative-pool
size is an upper bound. Use it to choose *C*, not to characterise the data.

Usage::

    python shortlist_scale_classes.py                     # top 40, floor 50
    python shortlist_scale_classes.py --floor 100 --n 25
    python shortlist_scale_classes.py --out shortlist.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pile_config as pc

#: COCO-2017's 80 classes, keyed by the VG free-text name that denotes the same
#: thing. Overlap is worth flagging because COCO val2017 is **exhaustively**
#: annotated over these classes: for a class in both vocabularies, VG's miss
#: rate can be measured against COCO with no human annotation at all, and a
#: human annotator's own accuracy can be scored the same way. That makes a
#: shared class the cheapest possible calibration of the whole correction pass.
COCO_CLASSES: frozenset[str] = frozenset(
    """person bicycle car motorcycle airplane bus train truck boat bench bird cat dog horse sheep
    cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard
    kite skateboard surfboard bottle cup fork knife spoon bowl banana apple sandwich orange broccoli
    carrot pizza donut cake chair couch bed toilet tv laptop mouse remote keyboard microwave oven
    toaster sink refrigerator book clock vase scissors toothbrush""".split()
    + [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "sports ball",
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "wine glass",
        "hot dog",
        "potted plant",
        "dining table",
        "cell phone",
        "hair drier",
        "teddy bear",
    ]
)

#: VG's vocabulary is free text, so the same object arrives under several names.
#: Only aliases that resolve to a COCO class are listed -- this map exists to
#: make the COCO flag correct, not to normalise VG generally.
COCO_ALIASES: dict[str, str] = {
    "plane": "airplane",
    "aeroplane": "airplane",
    "jet": "airplane",
    "motorbike": "motorcycle",
    "sofa": "couch",
    "television": "tv",
    "monitor": "tv",
    "cellphone": "cell phone",
    "phone": "cell phone",
    "fridge": "refrigerator",
    "bike": "bicycle",
    "cycle": "bicycle",
    "hydrant": "fire hydrant",
    "trafficlight": "traffic light",
    "computer": "laptop",
    "purse": "handbag",
    "glass": "wine glass",
}

#: Categories the overview benchmark already ran, so a corrected re-run can be
#: compared against a published number rather than starting from nothing.
#: Sources: `docs/experiments/overview-bench/REPORT.md`, "Classes used".
BENCHMARKED: frozenset[str] = frozenset(
    # wave 1, visual_genome_m
    """ball bed bus cat laptop nose sink sky""".split()
    # wave 2 re-run, vg_box_{small,medium,large}
    + """nose glasses watch camera tip outlet drain mask mustache tusks
    hair shorts clock lamp truck backpack basket frisbee holder chairs
    fence hill lady couch court walkway runway station intersection barn""".split()
    # wave 2 first run
    + """hands lips chest collar dresser sheet""".split()
)


def coco_name(vg_name: str) -> str | None:
    """The COCO class *vg_name* denotes, or ``None`` when COCO has no such class."""
    candidate = COCO_ALIASES.get(vg_name, vg_name)
    return candidate if candidate in COCO_CLASSES else None


def rank(scan: dict, floor: int, max_inflation: float) -> tuple[list[dict], list[tuple[str, str, int]]]:
    """Categories clearing *floor* images in every band, best-supported first.

    Ranked on the **minimum** per-band count, because that is the binding
    constraint: a class with 8,000 large images and 6 small ones cannot carry a
    three-band contrast however abundant it looks overall.

    Returns the survivors *and* the classes that had the supply but failed the
    fitness policy, each with its reason. The rejects are returned rather than
    dropped because they are the list a human has to sanity-check: a wrong
    exclusion silently shrinks the study, and a wrong *inclusion* silently
    changes what it measures.
    """
    meta, stats = scan["meta"], scan["categories"]
    n_total = int(meta["n_images_scanned"])
    rows: list[dict] = []
    excluded: list[tuple[str, str, int]] = []
    for name, s in stats.items():
        per_band = {b: int(s["bands"][b]) for b in pc.BOX_BANDS}
        if min(per_band.values()) < floor:
            continue
        # A category whose union box is much larger than one instance is
        # scattered instances, not a region a user would drag -- and its band
        # assignment would describe the scatter, not the object.
        if s["union_inflation"] > max_inflation:
            excluded.append((name, "scattered", min(per_band.values())))
            continue
        # Pervasiveness is a property of the corpus, so it is measured here
        # rather than listed in the config.
        reason = pc.scale_study_exclusion(name)
        if reason is None and int(s["n_images"]) / n_total >= pc.PERVASIVE_PREVALENCE:
            reason = "pervasive"
        if reason is not None:
            excluded.append((name, reason, min(per_band.values())))
            continue
        rows.append(
            {
                "category": name,
                "min_band": min(per_band.values()),
                "bands": per_band,
                "oversize": int(s["bands"].get("oversize", 0)),
                "n_images": int(s["n_images"]),
                # Images holding no instance of this category at any size. The
                # negative pool a cell draws from -- and the pool whose "no bus
                # here" claim the annotation pass has to actually verify.
                "neg_pool": n_total - int(s["n_images"]),
                "coco": coco_name(name),
                "benchmarked": name in BENCHMARKED,
                "union_inflation": round(float(s["union_inflation"]), 3),
            }
        )
    rows.sort(key=lambda r: (-r["min_band"], r["category"]))
    excluded.sort(key=lambda e: (e[1], -e[2]))
    return rows, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default=str(pc.PILE / "vg_box_scale.json"), help="scan_vg_boxes.py output")
    ap.add_argument("--floor", type=int, default=50, help="minimum images per band (default 50)")
    ap.add_argument(
        "--max-inflation",
        type=float,
        default=pc.BAND_MAX_INFLATION,
        help=f"drop categories whose union box exceeds this multiple of one instance (default {pc.BAND_MAX_INFLATION})",
    )
    ap.add_argument("--n", type=int, default=40, help="how many rows to print (default 40)")
    ap.add_argument("--out", default="", help="also write the full ranking as JSON")
    args = ap.parse_args()

    scan_path = Path(args.scan)
    if not scan_path.exists():
        raise SystemExit(f"missing {scan_path}; run scan_vg_boxes.py first")
    scan = json.loads(scan_path.read_text())
    if "categories" not in scan:
        raise SystemExit(f"{scan_path} predates per-band supply; re-run scan_vg_boxes.py")

    rows, excluded = rank(scan, args.floor, args.max_inflation)
    n_total = int(scan["meta"]["n_images_scanned"])
    print(f"scanned {n_total} images; {len(scan['categories'])} categories in the scan")
    print(f"{len(rows)} clear >= {args.floor} images in ALL THREE bands (inflation <= {args.max_inflation})\n")

    hdr = f"{'category':<18}{'min':>7}{'small':>8}{'medium':>8}{'large':>8}{'negpool':>9}  {'coco':<14}bench"
    print(hdr)
    print("-" * len(hdr))
    for r in rows[: args.n]:
        b = r["bands"]
        print(
            f"{r['category']:<18}{r['min_band']:>7}{b['small']:>8}{b['medium']:>8}{b['large']:>8}"
            f"{r['neg_pool']:>9}  {(r['coco'] or ''):<14}{'yes' if r['benchmarked'] else ''}"
        )

    if excluded:
        print(f"\n=== had the supply, failed the fitness policy ({len(excluded)}) -- CHECK THESE ===")
        by_reason: dict[str, list[str]] = {}
        for name, reason, _ in excluded:
            by_reason.setdefault(reason, []).append(name)
        blurb = {
            "part": "size is the host's, and 'no nose here' is unverifiable wherever a person is",
            "place": "a location, not a thing: the box has no principled extent",
            "polysemous": "one string, several objects, so it cannot be scored as one class",
            "pervasive": f"annotated on >= {pc.PERVASIVE_PREVALENCE:.0%} of images; thin, untrustworthy negatives",
            "scattered": "union box far larger than one instance; the band describes the scatter",
            "non_object": "attribute, frame relation, or mass noun (pile_config.is_object_category)",
        }
        for reason in sorted(by_reason):
            names = sorted(by_reason[reason])
            print(f"\n  {reason} -- {blurb.get(reason, '')}")
            for i in range(0, len(names), 6):
                print(f"    {', '.join(names[i : i + 6])}")

    n_coco = sum(1 for r in rows if r["coco"])
    n_bench = sum(1 for r in rows if r["benchmarked"])
    print(f"\n{n_coco} of {len(rows)} are also COCO classes (free cross-check against exhaustive annotation)")
    print(f"{n_bench} of {len(rows)} were already benchmarked (comparable to the published numbers)")
    print("\nNOTE: these counts come from VG's own annotations, which #3156 disputes.")
    print("      Treat positive supply as a lower bound and the negative pool as an upper bound.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
