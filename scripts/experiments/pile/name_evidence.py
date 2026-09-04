"""If a VG name were the only evidence, how often is the class really there?

``coco_folds.py`` answers a box question: does a VG box named *n* land on a COCO
box of class *c*? That is the right test for an **alias**, because
:func:`pilebuild.loaders.vg_scale.canonicalise` folds the box in and the band is
then read off it -- a name whose box frames something else, or frames the object
plus its cabinet, cannot serve as a positive.

It is the wrong test for the **negative pool**, and the pool is where the defect
of #3605 actually lives. An image is unusable as a negative for *c* the moment
*c* is present on it, however the box is drawn. `grandfather clock` scores 0 of
6 on box agreement -- correctly, since the box is the cabinet and COCO's is the
dial -- and COCO finds a clock on every image where it is the only evidence.

So this asks the image question instead, and asks COCO to answer it:

    over the VG-COCO overlap, take the images where VG has a box named *n*
    and **no** box named *c* -- which is exactly the situation that produces a
    false negative on the other half -- and ask COCO whether *c* is present.

That share is the name's **repair precision**. It is measured on the half with
an exhaustive reference and applied to the half without one, which is the same
trade ``anchor_to_coco`` already makes.

**Two numbers, two tables, and the split is not cosmetic:**

* ``precision`` (image level) is what :data:`pile_config.SCALE_VG_AMBIGUOUS`
  needs. Withholding an image from the pool only claims *the class may be here*.
* ``box_agree`` (box level, the same quantity ``coco_folds.py`` prints) is the
  extra thing :data:`pile_config.SCALE_VG_NAMES` needs. Folding a name in claims
  *this box is the object*, and a band is a claim about that box's size (#3616).

A name can pass the first and fail the second -- `wheel`, `clocks`,
`grandfather clock` all do -- and those are precisely the names that belong in
the ambiguous table rather than the alias one.

``base`` is the class's prevalence over the same overlap images, so a precision
can be read against the rate a name picked at random would score. A co-occurring
name inherits some of it: `wheel` is on bicycle images because bicycles have
wheels, so its precision is well above base and it is still not a bicycle.

Usage::

    python name_evidence.py --candidates cands.json --out evidence.json

``cands.json`` is ``{class: [names]}``; with no file, every class is scored
against its own head-noun family (``vg_name_families.py``).

**The verdict is derived, not drafted.** Three cuts, and each one is a number
this script measures:

* ``precision`` decides whether the name is worth acting on at all, and the
  right way to read it is as a **price**: ``1 / precision`` is how many images
  leave the shared negative pool per contaminated negative removed. At 0.85 that
  is 1.2 images; at `sign`'s 0.04 it is 25, which is why the biggest fold-in
  column in *C* is not actionable. The cut is on the **Wilson lower bound**, so
  a name with four supporting images does not outrank one with four hundred.
* ``box_agree`` then decides which table. Above ``--min-box`` the name's box is
  the object, so it can be folded and banded: **alias**. Below it the class is
  there but this box is not it: **ambiguous**.
* Below ``--context-box`` the box is not the object at all -- `beak`, `stop`,
  `bookshelf` -- and the name is scored **context**. Identical treatment to
  ambiguous, and reported apart because the two differ in what they cost: a
  spelling withholds the images that spell the class oddly, while a context name
  withholds a whole scene type from **every** class's pool, not just its own.
  ``--include-context`` puts them in the proposal; the default leaves them out.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pile_config as pc

pc.setup_env()

import coco_folds as cf  # noqa: E402  (setup_env must run before vtscore resolves)

VG_ROOT = pc.DEMO_CACHE / "visual_genome"


def log(msg: str) -> None:
    print(f"[evidence] {msg}", flush=True)


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson interval for *hits* of *n*.

    Used instead of the raw rate so that one cut serves names measured on five
    images and names measured on two thousand. `dove` is 5 of 5 and `bike` is
    508 of 1088; the raw rates say the first is twice the second, and the bound
    says what each one actually supports.
    """
    if n <= 0:
        return 0.0
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z2 / n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="", help="JSON {class: [names]}; default = head-noun families")
    ap.add_argument("--families", default="", help="vg_name_families.py --out JSON, used when --candidates is absent")
    ap.add_argument("--anchor-dir", default=str(pc.PILE / "coco_anchor"))
    ap.add_argument("--iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
    ap.add_argument("--min-sole", type=int, default=5, help="below this many adjudicable images a rate is not a rate")
    ap.add_argument(
        "--min-precision",
        type=float,
        default=1.0 / 3.0,
        help="Wilson lower bound a name must clear to be worth acting on. The default 1/3 is a "
        "ceiling of three images withheld from the negative pool per contaminated negative removed.",
    )
    ap.add_argument("--min-box", type=float, default=0.5, help="box agreement at or above which a name can be folded")
    ap.add_argument(
        "--min-boxes",
        type=int,
        default=20,
        help="boxes a name needs before it may be folded. Higher than --min-sole on purpose: an "
        "alias claims that EVERY box under this name is the object, and five boxes cannot carry "
        "that claim. A name below the floor falls to the ambiguous table, which is the safe side -- "
        "a wrong ambiguous costs a few pool images, a wrong alias injects a mis-banded positive.",
    )
    ap.add_argument("--context-box", type=float, default=0.1, help="below this the name is not the object at all")
    ap.add_argument("--include-context", action="store_true", help="put `context` names in the proposal too")
    ap.add_argument("--propose-out", default="", help="write the derived tables as a name_coverage.py proposal")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    classes = list(pc.SCALE_CLASSES)
    if args.candidates:
        cands = {c: [n.strip().lower() for n in ns] for c, ns in json.loads(Path(args.candidates).read_text()).items()}
    elif args.families:
        fam = json.loads(Path(args.families).read_text())
        cands = {c: [r["name"] for r in rows] for c, rows in fam["families"].items()}
    else:
        raise SystemExit("pass --candidates or --families")
    unknown = set(cands) - set(classes)
    if unknown:
        raise SystemExit(f"candidates name classes that are not in C: {sorted(unknown)}")

    wanted = set(classes) | {n for ns in cands.values() for n in ns}
    log(f"{len(classes)} classes; {len(wanted)} VG names")

    cboxes, cdims, cpresent = cf.coco_boxes(Path(args.anchor_dir), set(classes))

    log("loading VG image_data.json")
    with (Path(args.anchor_dir) / "image_data.json").open() as fh:
        meta = json.load(fh)
    coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in meta if m.get("coco_id")}
    vdims = {int(m["image_id"]): (int(m["width"]), int(m["height"])) for m in meta}

    log(f"loading VG objects.json ({(VG_ROOT / 'objects.json').stat().st_size / 1e6:.0f} MB)")
    with (VG_ROOT / "objects.json").open() as fh:
        records = json.load(fh)
    log(f"  {len(records)} VG records")

    # per (class, name): the image question, adjudicated by COCO
    sole = defaultdict(int)  # overlap images with a box named n and none named c
    sole_hit = defaultdict(int)  # ... on which COCO says c is present anyway
    # per (class, name): the box question -- n's boxes that land on a COCO c box
    boxes = defaultdict(int)
    boxes_hit = defaultdict(int)
    # supply, and the images the fold would actually act on
    vg_images = defaultdict(int)
    off_sole = defaultdict(int)
    # base rate: overlap images where COCO annotates c at all
    overlap_images = 0
    base_hit = dict.fromkeys(classes, 0)

    skipped_aspect = 0
    for rec in records:
        iid = int(rec["image_id"])
        vd = vdims.get(iid)
        if vd is None:
            continue

        by_name: dict[str, list[list[float]]] = {}
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if name not in wanted:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w <= 0 or h <= 0:
                continue
            by_name.setdefault(name, []).append([x / vd[0], y / vd[1], (x + w) / vd[0], (y + h) / vd[1]])
        for name in by_name:
            vg_images[name] += 1

        cid = coco_of.get(iid)
        if cid is None or cid not in cdims:
            for c, names in cands.items():
                if c not in by_name:
                    for n in names:
                        if n in by_name:
                            off_sole[c, n] += 1
            continue

        cd = cdims[cid]
        if abs((vd[0] / vd[1]) - (cd[0] / cd[1])) > pc.MAX_ASPECT_DRIFT:
            skipped_aspect += 1
            continue
        overlap_images += 1
        here = cpresent.get(cid, set())
        for c in classes:
            base_hit[c] += c in here
        for c, names in cands.items():
            truth = cboxes.get(cid, {}).get(c, [])
            for n in names:
                vb = by_name.get(n)
                if not vb:
                    continue
                if c not in by_name:
                    # exactly the state that becomes a false negative off-COCO
                    sole[c, n] += 1
                    sole_hit[c, n] += c in here
                boxes[c, n] += len(vb)
                boxes_hit[c, n] += sum(1 for b in vb if any(cf.iou(t, b) >= args.iou for t in truth))

    log(f"{overlap_images} adjudicable overlap images; skipped {skipped_aspect} on aspect drift")

    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.0f}%" if b else "--"

    def verdict(c: str, n: str) -> str:
        """Which table *n* belongs in for *c*, from the three cuts above."""
        if sole[c, n] < args.min_sole:
            return "unmeasured"
        if wilson_lower(sole_hit[c, n], sole[c, n]) < args.min_precision:
            return "neither"
        if boxes[c, n] < args.min_boxes:
            # Precision cleared, so the class is there; there are just not enough
            # boxes to claim any of them IS it. Fall to the safe table.
            return "ambiguous"
        agree = boxes_hit[c, n] / boxes[c, n]
        if agree >= args.min_box:
            return "alias"
        return "ambiguous" if agree >= args.context_box else "context"

    verdicts = {(c, n): verdict(c, n) for c, names in cands.items() for n in names if vg_images[n]}

    print("\n" + "=" * 104)
    print("NAME EVIDENCE -- COCO adjudicating the images where a name is the class's ONLY evidence")
    print("`sole` = overlap images with the name and not the class name. `precision` = COCO says the")
    print("class is present anyway. `box` = the name's boxes landing on a COCO box of the class.")
    print(f"A row with fewer than {args.min_sole} sole images is marked `thin`: it is a count, not a rate.")
    print("=" * 104)
    for c in classes:
        if c not in cands:
            continue
        base = 100.0 * base_hit[c] / overlap_images if overlap_images else 0.0
        print(f"\n{c}   (base rate {base:.1f}% of overlap images; {vg_images[c]} VG images under the class name)")
        print(
            f"    {'name':<26}{'VG imgs':>8}{'sole':>7}{'prec':>6}{'lower':>7}"
            f"{'boxes':>7}{'box':>6}{'off-COCO':>10}  verdict"
        )
        rows = sorted(cands[c], key=lambda n: -off_sole[c, n])
        residual = [0, 0, 0]
        for n in rows:
            if not vg_images[n]:
                continue
            v = verdicts[c, n]
            if v == "unmeasured":
                residual[0] += sole[c, n]
                residual[1] += sole_hit[c, n]
                residual[2] += off_sole[c, n]
            if v == "neither" and off_sole[c, n] < 20:
                continue  # a refuted name with no supply is noise in a long table
            lower = wilson_lower(sole_hit[c, n], sole[c, n])
            print(
                f"    {n:<26}{vg_images[n]:>8}{sole[c, n]:>7}{pct(sole_hit[c, n], sole[c, n]):>6}"
                f"{lower:>7.2f}{boxes[c, n]:>7}{pct(boxes_hit[c, n], boxes[c, n]):>6}"
                f"{off_sole[c, n]:>10}  {v}"
            )
        # What the sole-image floor leaves behind, pooled: the same question
        # asked of every name too thin to answer it alone.
        if residual[0]:
            print(
                f"    {'(pooled unmeasured)':<26}{'':>8}{residual[0]:>7}{pct(residual[1], residual[0]):>6}"
                f"{wilson_lower(residual[1], residual[0]):>7.2f}{'':>7}{'':>6}{residual[2]:>10}  residual"
            )

    if args.propose_out:
        keep = {"ambiguous", "context"} if args.include_context else {"ambiguous"}
        proposal = {
            "alias": {
                c: sorted(n for n in names if verdicts.get((c, n)) == "alias")
                for c, names in cands.items()
                if any(verdicts.get((c, n)) == "alias" for n in names)
            },
            "ambiguous": {
                c: sorted(n for n in names if verdicts.get((c, n)) in keep)
                for c, names in cands.items()
                if any(verdicts.get((c, n)) in keep for n in names)
            },
        }
        Path(args.propose_out).write_text(json.dumps(proposal, indent=1) + "\n")
        n_alias = sum(len(v) for v in proposal["alias"].values())
        n_ambig = sum(len(v) for v in proposal["ambiguous"].values())
        print(f"\nwrote {args.propose_out}: {n_alias} alias names, {n_ambig} ambiguous names")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "meta": {
                        "iou": args.iou,
                        "overlap_images": overlap_images,
                        "skipped_aspect_drift": skipped_aspect,
                        "min_sole": args.min_sole,
                        "min_precision": args.min_precision,
                        "min_box": args.min_box,
                        "min_boxes": args.min_boxes,
                        "context_box": args.context_box,
                    },
                    "base_rate": {c: base_hit[c] / overlap_images if overlap_images else 0.0 for c in classes},
                    "class_images": {c: vg_images[c] for c in classes},
                    "names": {
                        c: {
                            n: {
                                "vg_images": vg_images[n],
                                "sole": sole[c, n],
                                "sole_present": sole_hit[c, n],
                                "boxes": boxes[c, n],
                                "boxes_on_class": boxes_hit[c, n],
                                "off_coco_sole": off_sole[c, n],
                                "precision_lower": wilson_lower(sole_hit[c, n], sole[c, n]),
                                "verdict": verdicts[c, n],
                            }
                            for n in names
                            if vg_images[n]
                        }
                        for c, names in cands.items()
                    },
                },
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
