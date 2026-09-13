#!/usr/bin/env python3
"""Why is an image VG DID name not a designated positive? (#3818)

`silence_source.py` splits every confirmed error of the exhaustive pass by what
VG said about the image, and **441 of 815 land in the `folded` bucket**: a human
confirmed the class is present, VG named it under the class's own spelling or one
`SCALE_VG_NAMES` already folds, and the image is still not among that class's
designated positives. That bucket is correct for #3696's purpose -- a folded
image can never be drawn as a negative for its own class, so it is not
contamination -- and #3696 deliberately says nothing more about it. This says the
rest.

**The answer is read off the loader's own passes, not re-derived.** The whole
front half of `pilebuild.loaders.vg_scale.load` is run here in its own order --
`read_vg_labels`, `anchor_to_coco`, `canonicalise`, `apply_corrections`,
`lift_ambiguous`, `band_candidates`, `designate_cells` -- and each folded pair is
asked, after each pass, whether the class is still on the image. The first pass
that drops it is the answer. A second implementation of the band edges or the
scatter guard here would answer a question about the build with its own drift,
which is the mistake `band_for` exists to prevent (`audit_band_drift.py` shares
it for the same reason), and `annotation_queue.py` reads the build's own
`coco_scored` stamp rather than re-joining COCO for the same reason again.

## The outcomes, and which of them is a fault

#3818 anticipated three, and the third is not a fault:

* ``oversize`` -- the union box covers >= ``MAX_VOTED_AREA`` of the frame, so
  ``band_for`` puts it in no band. Not a region, the image;
* ``scattered`` -- the instances are too far apart for their union to describe
  one object (``BAND_MAX_INFLATION``). A property of the *image*, not the class
  (#3156);
* ``cell_full`` -- the image is in the cell's supply and `designate_cells` filled
  the cell from higher `rank`s before reaching it. **This is an ordinary state
  and must not be reported as a fault.** A cell takes ``SCALE_N_POS`` from
  whatever supply it has; a cell 30x over-subscribed leaves 29 of every 30 good
  images undesignated by design, and each is a seat that was never scarce for the
  *reviewed* images, which outrank the rest.

There are more than three, and the extra ones are the reason this runs the passes
instead of assuming. A pair can also leave before it ever reaches a band:

* ``image_dropped`` -- the image is in VG's object table but not in the build's
  read (no JPEG, or no dimensions);
* ``degenerate_box`` -- VG named the class and gave it a zero-width or
  zero-height box, which `vg_boxes_by_name` drops. VG spoke and measured nothing;
* ``anchor_absent`` -- the image is COCO-anchored, so `anchor_to_coco` replaced
  VG's labels with COCO's wholesale, and COCO does not carry the class here. Two
  annotators disagree and the build believes COCO, on purpose;
* ``reviewed_absent`` / ``reviewed_boxless`` -- a row in `corrections.json`
  already answers this pair. The first says a reviewer ruled it absent, the
  second that a reviewer ruled it present without drawing a box, which cannot be
  banded (a band is a claim about size and no size was measured) and leaves the
  pair `unbanded` -- neither a positive nor a negative;
* ``withheld_lift`` -- `lift_ambiguous` dropped the spelling.

``designated`` is the alarm. A folded pair that this replay *does* designate is a
pair the queue says is undesignated and the build says is not, which means the
inputs moved after the queue was written -- almost always `corrections.json`,
which every build of this family shares. It is counted, never quietly folded into
another bucket, and it is the one outcome that invalidates the rest of the table.

Costs a couple of minutes of CPU and ~12 GB: it runs the loader's front half and
reads VG's object table once, shared with `silence_source`'s classification so
the 350 MB parse is not paid twice. No GPU, no rebuild.

Usage::

    python folded_supply.py --rate silence_rate.json --out folded_supply.json
    python folded_supply.py --rate silence_rate.json --per-outcome 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc

import silence_rate as sr
import silence_source as ss
import verdict_store
from pilebuild.corrections import load_corrections
from pilebuild.loaders.vg_scale import (
    OVERSIZE,
    SCATTERED,
    anchor_to_coco,
    apply_corrections,
    band_candidates,
    band_for,
    canonicalise,
    designate_cells,
    lift_ambiguous,
    read_vg_labels,
)
from pilebuild.vgsource import vg_image_paths, vg_source

#: Every outcome, in the order the passes decide them. The order is the table's
#: order too: a reader following it is walking the build forwards.
OUTCOMES = (
    "image_dropped",
    "degenerate_box",
    "anchor_absent",
    "fold_lost",
    "reviewed_absent",
    "reviewed_boxless",
    "withheld_lift",
    "scattered",
    "oversize",
    "cell_full",
    "designated",
)

#: What #3818 asked to be apportioned. The rest of :data:`OUTCOMES` is the part
#: the issue did not anticipate, and is reported under its own heading rather
#: than being folded into the nearest of these.
ASKED = {"oversize": "missed a band", "scattered": "scatter filter", "cell_full": "cell already full"}

#: The only outcome that is not a fault. Named so the report and the script agree
#: on which number is the one that means "nothing to do here".
BENIGN = "cell_full"


def log(msg: str) -> None:
    print(f"[folded] {msg}", flush=True)


def spellings(cls: str) -> set[str]:
    """Every key `labels` could carry the class under, before or after the fold.

    Not `silence_source.known_names`: that one includes `SCALE_VG_AMBIGUOUS` on
    purpose, because it asks whether VG *named* the object. This asks what the
    build is holding, and after `canonicalise` an ambiguous spelling is still its
    own key -- `lift_ambiguous` is what removes it, and seeing that happen is one
    of the outcomes.
    """
    return {cls, *pc.SCALE_VG_NAMES.get(cls, ())}


def held(labels: dict[int, dict[str, list[list[float]]]], iid: int, cls: str) -> bool:
    """Does the build still carry *cls* on this image, under any of its spellings?"""
    by_name = labels.get(iid)
    return by_name is not None and bool(set(by_name) & spellings(cls))


def folded_pairs(
    rate: dict,
    answers: dict[str, dict[int, bool]],
    silent: dict[str, set[int]],
    names: dict[int, Counter],
) -> list[tuple[str, int]]:
    """``[(class, image_id)]`` -- #3696's `folded` bucket, pair by pair.

    `silence_source.py` writes the bucket as a count, which is all its own
    argument needs. Recovered here through its own :func:`~silence_source.classify`
    rather than re-implemented, so the two cannot disagree about what `folded`
    means: the check this script prints first is that the recovered counts equal
    the published ones.
    """
    out = []
    for cls in sorted(set(rate["pooled"]["classes"])):
        pool = silent.get(cls, set())
        for iid in sorted(i for i, yes in answers.get(cls, {}).items() if yes and i in pool):
            if ss.classify(cls, set(names.get(iid, ()))) == "folded":
                out.append((cls, iid))
    return out


def stages(
    pairs: list[tuple[str, int]],
    labels: dict[int, dict[str, list[list[float]]]],
) -> dict[tuple[str, int], bool]:
    """Snapshot "is the class still here?" for every pair, at one point in the build.

    Taken between passes rather than inside them: a pass that reported its own
    casualties would have to be edited to answer a new question, and the whole
    point of #3818 is that the question was not anticipated.
    """
    return {(cls, iid): held(labels, iid, cls) for cls, iid in pairs}


def apportion(
    pairs: list[tuple[str, int]],
    snaps: dict[str, dict[tuple[str, int], bool]],
    unbanded_corr: set[tuple[int, str]],
    labels: dict[int, dict[str, list[list[float]]]],
    designated: dict[tuple[int, str], list[list[float]]],
    box_dims: dict[int, tuple[int, int]],
    supply: dict[str, dict[str, list[int]]],
    chosen: dict[str, list[int]],
    in_read: set[int],
) -> list[dict]:
    """One row per pair: the first pass that dropped it, or the band it reached.

    Pure given the snapshots, so the cascade is testable without VG's object
    table -- the same reason `silence_source.census` is pure.
    """
    rows = []
    for cls, iid in pairs:
        key = (cls, iid)
        row: dict = {"class": cls, "image_id": iid}
        if not snaps["read"][key]:
            # Two different facts, and lumping them would hide the one that is
            # about VG rather than about the build: an image the build never saw
            # at all, versus a box VG drew with no area.
            row["outcome"] = "degenerate_box" if iid in in_read else "image_dropped"
        elif not snaps["anchor"][key]:
            row["outcome"] = "anchor_absent"
        elif not snaps["fold"][key]:
            # `canonicalise` merges alias boxes ONTO the class name and removes
            # nothing, so this is expected to be zero. It is in the cascade
            # anyway: the alternative is that a fold which did start dropping
            # pairs would be charged to `apply_corrections`, the next test down.
            row["outcome"] = "fold_lost"
        elif not snaps["corrections"][key]:
            row["outcome"] = "reviewed_boxless" if (iid, cls) in unbanded_corr else "reviewed_absent"
        elif not snaps["lift"][key]:
            row["outcome"] = "withheld_lift"
        else:
            W, H = box_dims[iid]
            boxes = designated.get((iid, cls)) or labels[iid][cls]
            band = band_for(boxes, W, H)
            # The area and the instance count are what tell a near-miss from an
            # image-sized box, and #3818 cannot be answered "the scatter filter
            # dropped 40" without saying whether they were 1.6x or 12x inflated.
            area = float(W * H)
            ux0 = min(b[0] for b in boxes)
            uy0 = min(b[1] for b in boxes)
            ux1 = max(b[2] for b in boxes)
            uy1 = max(b[3] for b in boxes)
            union = max(0.0, ux1 - ux0) * max(0.0, uy1 - uy0) / area
            largest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / area
            row |= {
                "boxes": len(boxes),
                "union_area": union,
                "largest_area": largest,
                "inflation": union / largest if largest else 0.0,
                "designated_box": (iid, cls) in designated,
            }
            if band == SCATTERED:
                row["outcome"] = "scattered"
            elif band == OVERSIZE:
                row["outcome"] = "oversize"
            else:
                cell = pc.scale_cell(cls, band)
                row |= {"band": band, "cell": cell, "cell_supply": len(supply[cls][band])}
                row["outcome"] = "designated" if iid in chosen.get(cell, []) else "cell_full"
        rows.append(row)
    return rows


def census(rows: list[dict]) -> tuple[list[dict], dict]:
    """``(per-class rows, pooled)`` -- the apportionment, counted.

    Pure, and the only place a share is computed. Every row carries every outcome
    so a zero is a printed zero rather than a gap a reader has to interpret.
    """
    by_class: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_class[r["class"]][r["outcome"]] += 1
    out = []
    for cls in sorted(by_class):
        counts = by_class[cls]
        n = sum(counts.values())
        out.append(
            {
                "class": cls,
                "folded": n,
                **{k: counts[k] for k in OUTCOMES},
                "benign_share": counts[BENIGN] / n if n else 0.0,
            }
        )
    totals: Counter = Counter()
    for counts in by_class.values():
        totals.update(counts)
    n = sum(totals.values())
    pooled = {
        "classes": sorted(by_class),
        "folded": n,
        **{k: totals[k] for k in OUTCOMES},
        "benign_share": totals[BENIGN] / n if n else 0.0,
        "actionable": sum(totals[k] for k in ASKED if k != BENIGN),
    }
    return out, pooled


def control(
    labels: dict[int, dict[str, list[list[float]]]],
    designated: dict[tuple[int, str], list[list[float]]],
    box_dims: dict[int, tuple[int, int]],
    chosen: dict[str, list[int]],
    classes: list[str],
    queue_ids: set[int],
) -> list[dict]:
    """The same apportionment over EVERY undesignated queue image holding the class.

    The control the headline needs, and the reason "the scatter filter dropped 77
    of 441" cannot be read on its own. **The conditioning has to match, or the
    comparison invents a finding.** The 441 are undesignated by construction --
    `silent_pairs` is defined that way -- and a scattered image *can never be
    designated*, so scatter is over-represented among undesignated pairs however
    the build behaves. Comparing 17.5% against the filter's rate over all queue
    images holding the class (4.9%) therefore reports a 3.6x excess that is
    entirely the conditioning, and nothing about these images.

    So the denominator here carries the same condition the numerator does:
    queue images, holding the class after every pass, **not designated for it**.
    What is left between that population and the 441 is the screen and the
    reviewer -- which is exactly the thing #3818 is asking about.

    Judged through :func:`~pilebuild.loaders.vg_scale.band_for` on the same
    designated-box rule `band_candidates` uses, so the control cannot drift from
    the measurement it controls.
    """
    out = []
    for cls in classes:
        seats = {i for band in pc.BOX_BANDS for i in chosen.get(pc.scale_cell(cls, band), [])}
        counts: Counter = Counter()
        for iid, by_name in labels.items():
            if cls not in by_name or iid not in queue_ids:
                continue
            if iid in seats:
                counts["designated"] += 1
                continue
            W, H = box_dims[iid]
            band = band_for(designated.get((iid, cls)) or by_name[cls], W, H)
            counts[band if band in (SCATTERED, OVERSIZE) else "cell_full"] += 1
        n = sum(counts[k] for k in ("scattered", "oversize", "cell_full"))
        out.append(
            {
                "class": cls,
                "queue_held": sum(counts.values()),
                "designated": counts["designated"],
                "undesignated": n,
                **{k: counts[k] for k in ("scattered", "oversize", "cell_full")},
                **{f"{k}_rate": counts[k] / n if n else 0.0 for k in ("scattered", "oversize", "cell_full")},
            }
        )
    return out


def oversubscription(supply: dict[str, dict[str, list[int]]], classes: list[str]) -> list[dict]:
    """Per cell: how much supply there is for ``SCALE_N_POS`` seats.

    The context `cell_full` has to be read against. "441 confirmed positives the
    pile does not designate" is only a finding if seats were available, and a
    cell drawing 100 from several thousand candidates has none to give.
    """
    out = []
    for cls in classes:
        for band in pc.BOX_BANDS:
            n = len(supply[cls][band])
            out.append(
                {
                    "cell": pc.scale_cell(cls, band),
                    "supply": n,
                    "seats": pc.SCALE_N_POS,
                    "oversubscription": n / pc.SCALE_N_POS if pc.SCALE_N_POS else 0.0,
                }
            )
    return out


def main() -> int:
    # Deferred exactly as `vg_scale.load` defers it: `coco_anchor` runs
    # `setup_env` at import, which makes the pile's directories. Importing it at
    # module level would mean this file cannot be imported at all without a
    # writable `/expscratch` -- and the suite imports it to exercise the pure
    # functions on a laptop.
    import coco_anchor as ca  # noqa: PLC0415

    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent
    ap.add_argument("--rate", required=True, help="silence_rate.py's --out")
    ap.add_argument("--source", default="", help="silence_source.py's --out, to check the recovered bucket against")
    ap.add_argument("--queue", default=str(base / "vgscale-3156" / "annotation_queue.jsonl"))
    ap.add_argument("--store", default=str(verdict_store.STORE))
    ap.add_argument("--per-outcome", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rate = json.loads(Path(args.rate).read_text())
    eligible = sorted(set(rate["pooled"]["classes"]))
    queue = [json.loads(x) for x in Path(args.queue).read_text().splitlines() if x.strip()]
    silent = sr.silent_pairs(queue, tuple(pc.SCALE_CLASSES))
    answers = sr.read_answers(Path(args.store), {c: pc.review_name(c) for c in pc.SCALE_CLASSES})[0]

    wanted = set(pc.SCALE_CLASSES)
    corrections = load_corrections()
    paths = vg_image_paths()
    log(f"reading VG source ({len(paths)} images on disk)")
    _, records, dims = vg_source()

    # One parse of the object table, two readers. `silence_source` classifies the
    # bucket and the loader's read builds the boxes; paying 350 MB twice to keep
    # them in separate processes would buy nothing but a second chance to drift.
    names = ss.vg_names_by_image(records, {int(r["image_id"]) for r in queue})
    pairs = folded_pairs(rate, answers, silent, names)
    log(f"{len(pairs)} folded pairs recovered over {len(eligible)} classes")
    if args.source:
        # The published bucket is the contract. If this recovers a different set,
        # everything below is about some other 441 images.
        published = {r["class"]: r["folded"] for r in json.loads(Path(args.source).read_text())["classes"]}
        got = Counter(c for c, _ in pairs)
        bad = {c: (got[c], published.get(c)) for c in published if got[c] != published[c]}
        if bad:
            raise SystemExit(f"folded bucket does not match {args.source}: got/published {bad}")
        log(f"recovered bucket matches {Path(args.source).name} class for class")

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        coco_of = {int(m["image_id"]): int(m["coco_id"]) for m in json.load(fh) if m.get("coco_id")}

    labels = read_vg_labels(records, paths, dims, pc.scale_vg_wanted())
    in_read = set(labels)
    snaps = {"read": stages(pairs, labels)}
    box_dims, exhaustive, *_ = anchor_to_coco(labels, dims, coco_of, truth, ca.COCO_DIMS, wanted)
    snaps["anchor"] = stages(pairs, labels)
    canonicalise(labels, pc.SCALE_VG_NAMES, box_dims, pc.SCALE_FOLD_MODE)
    snaps["fold"] = stages(pairs, labels)
    designated: dict[tuple[int, str], list[list[float]]] = {}
    unbanded_corr = apply_corrections(labels, corrections, box_dims, exhaustive, designated=designated)
    snaps["corrections"] = stages(pairs, labels)
    unbanded = unbanded_corr | lift_ambiguous(labels, pc.SCALE_VG_AMBIGUOUS, exhaustive)
    snaps["lift"] = stages(pairs, labels)

    supply, _boxes, _clean = band_candidates(labels, box_dims, unbanded, designated=designated)
    roster = json.loads(pc.ROSTER.read_text()) if pc.ROSTER.exists() else {}
    log(f"roster pins {len(roster.get('cells', {}))} cells; {len(corrections)} corrections on file")
    chosen = designate_cells(supply, corrections, roster)

    rows = apportion(pairs, snaps, unbanded_corr, labels, designated, box_dims, supply, chosen, in_read)
    per_class, pooled = census(rows)
    cells = oversubscription(supply, eligible)
    rates = control(labels, designated, box_dims, chosen, eligible, {int(r["image_id"]) for r in queue})

    head = f"{'class':<14}{'folded':>8}" + "".join(f"{k.split('_')[0][:9]:>10}" for k in OUTCOMES)
    print("\n" + head)
    print("-" * len(head))
    for r in per_class:
        print(f"{r['class']:<14}{r['folded']:>8}" + "".join(f"{r[k]:>10}" for k in OUTCOMES))
    print("-" * len(head))
    print(f"{'POOLED':<14}{pooled['folded']:>8}" + "".join(f"{pooled[k]:>10}" for k in OUTCOMES))

    # The control, conditioned the same way the 441 are. A share of the folded
    # bucket means nothing until the same share of the population it was drawn
    # from is printed beside it.
    head2 = (
        f"{'class':<14}{'the 441':>9}{'scat':>7}{'over':>7}{'full':>7}  |"
        f"{'undesig.':>10}{'scat':>8}{'over':>8}{'full':>8}"
    )
    print("\n" + head2)
    print("-" * len(head2))
    for r, b in zip(per_class, rates, strict=True):
        print(
            f"{r['class']:<14}{r['folded']:>9}{r['scattered'] / r['folded']:>7.1%}"
            f"{r['oversize'] / r['folded']:>7.1%}{r['cell_full'] / r['folded']:>7.1%}  |"
            f"{b['undesignated']:>10,}{b['scattered_rate']:>8.1%}{b['oversize_rate']:>8.1%}{b['cell_full_rate']:>8.1%}"
        )
    print("-" * len(head2))
    tot = {k: sum(b[k] for b in rates) for k in rates[0] if k != "class" and not k.endswith("_rate")}
    print(
        f"{'POOLED':<14}{pooled['folded']:>9}{pooled['scattered'] / pooled['folded']:>7.1%}"
        f"{pooled['oversize'] / pooled['folded']:>7.1%}{pooled['cell_full'] / pooled['folded']:>7.1%}  |"
        f"{tot['undesignated']:>10,}{tot['scattered'] / tot['undesignated']:>8.1%}"
        f"{tot['oversize'] / tot['undesignated']:>8.1%}{tot['cell_full'] / tot['undesignated']:>8.1%}"
    )

    print(f"\n{'cell':<24}{'supply':>9}{'seats':>7}{'over-subscribed':>18}")
    print("-" * 58)
    for c in sorted(cells, key=lambda x: x["oversubscription"]):
        print(f"{c['cell']:<24}{c['supply']:>9,}{c['seats']:>7}{c['oversubscription']:>17.1f}x")

    examples = pick_examples(rows, names, args.per_outcome)
    print(f"\n{'class':<14}{'image':>9}  {'outcome':<11}{'boxes':>6}{'union':>7}{'infl':>7}  {'cell':<20}what VG said")
    print("-" * 126)
    for r in examples:
        print(
            f"{r['class']:<14}{r['image_id']:>9}  {r['outcome']:<11}{r.get('boxes', ''):>6}"
            f"{r.get('union_area', 0):>7.3f}{r.get('inflation', 0):>7.2f}  {r.get('cell', '-'):<20}"
            + ", ".join(r["vg_named"][:6])
        )
    print("-" * 126)

    log(
        f"{pooled[BENIGN]} of {pooled['folded']} ({pooled['benign_share']:.0%}) are `{BENIGN}` -- "
        "the cell took its SCALE_N_POS from a larger supply and never reached this image. "
        "That is the designed behaviour of an over-subscribed cell, not a fault."
    )
    log(
        f"{pooled['actionable']} are the two #3818 would act on: "
        + ", ".join(f"{k}={pooled[k]}" for k in ASKED if k != BENIGN)
    )
    # Against the population the 441 were drawn from, not against 441 alone. A
    # share of a selected sample is not a finding until the same share of the
    # unselected population is beside it.
    for k in (SCATTERED, OVERSIZE, BENIGN):
        log(
            f"  {k}: {pooled[k] / pooled['folded']:.1%} of the folded bucket, against "
            f"{tot[k] / tot['undesignated']:.1%} of all {tot['undesignated']:,} undesignated queue "
            f"images holding one of these classes"
        )
    other = pooled["folded"] - sum(pooled[k] for k in ASKED)
    if other:
        log(
            f"{other} are outcomes #3818 did not anticipate: "
            + ", ".join(f"{k}={pooled[k]}" for k in OUTCOMES if k not in ASKED and pooled[k])
        )
    if pooled["designated"]:
        log(
            f"ALARM: {pooled['designated']} folded pairs ARE designated by this replay. The queue says "
            "they are not, so an input moved after the queue was written (corrections.json is shared "
            "by every build of this family) and the table above describes a different build."
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "classes": per_class,
                    "pooled": pooled,
                    "control": rates,
                    "cells": cells,
                    "rows": rows,
                    "examples": examples,
                },
                indent=1,
            )
            + "\n"
        )
        log(f"wrote {args.out}")
    return 0


def pick_examples(rows: list[dict], names: dict[int, Counter], per_class: int) -> list[dict]:
    """The first *per_class* rows of each ``(outcome, class)``, by image id.

    By id, which is arbitrary and therefore not a choice -- any ranking would be
    picking the examples that make the point, and the point is the census.

    **Per class, not per outcome.** Taking the first rows of each outcome alone
    returns whichever class sorts first and nothing else: the first draw of this
    was four `backpack` rows for `scattered` and four more for `cell_full`, which
    illustrates one class's habits and calls it the corpus. `boat` is 41%
    scattered and `bicycle` 8%, and an examples table that cannot show both is
    not showing the errors behind the rate.

    Each row carries what VG said, so a reader can check the classification
    rather than taking it: an image listed as `folded` whose names do not include
    a spelling of the class would be a bug in the bucket, not an example of it.
    """
    out = []
    for outcome in OUTCOMES:
        got = [r for r in rows if r["outcome"] == outcome]
        for cls in sorted({r["class"] for r in got}):
            for r in sorted((x for x in got if x["class"] == cls), key=lambda x: x["image_id"])[:per_class]:
                got_names = names.get(r["image_id"], Counter())
                out.append(
                    r
                    | {
                        "vg_names_total": sum(got_names.values()),
                        "vg_named": [n for n, _ in got_names.most_common() if n not in ss._NOISE][:10],
                        "matched": sorted(set(got_names) & spellings(cls)),
                    }
                )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
