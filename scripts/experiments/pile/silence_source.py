#!/usr/bin/env python3
"""Was VG really silent, or did the build not listen? (#3696)

`silence_rate.py` calls a `(image, class)` pair silent when the queue does not
**designate** that image for that class, because a designation is all a cell
pickle carries (#3678). That is not the same fact as "VG never named it", and
the gap is not small: an image VG called a `bike` is not designated `bicycle`,
because `lift_ambiguous` withholds the name on purpose (#3605); an image VG
called a `backpack` is not designated either if its box missed the band edges or
the scatter filter. Both look identical from the pickle, and a human confirming
the object in either one is recorded as VG having been silent.

So this reads VG's own `objects.json` and splits every confirmed error three
ways:

* **coverage** -- no name on the image belongs to the class under any spelling
  the tables know. This is the reading that matters, and it is itself an upper
  bound: a name nobody has audited yet (`seat` for a bench) lands here, because
  the tables are the only authority on what counts as the same object and a
  resemblance rule would manufacture findings out of `bike rack`. #3618 is the
  loop that shrinks it;
* **withheld** -- VG used a name `SCALE_VG_AMBIGUOUS` deliberately refuses
  (`bike`, `stop`, `sailboat`). VG spoke; the build declined to listen, which is
  a ruling and not an error;
* **folded** -- VG used the class's own name or a spelling `SCALE_VG_NAMES`
  already folds. Nothing to do with VG at all: the image is simply not among the
  class's *designated* positives. Which of the reasons applies is not read here
  and must not be guessed -- its box may have missed a band, the scatter filter
  may have dropped it, or `designate_cells` may have filled the cell from higher
  ranks and never reached it. Over-subscription is an ordinary state, not a
  fault, which is why this bucket is reported as one count and not as a diagnosis
  (#3818).

**Only the first can contaminate a negative pool**, which is the whole reason
the rate has a consumer. `lift_ambiguous` withholds an ambiguous name from the
shared pool as well as from the bands, and a folded name puts the image in the
class outright, so an image in either of the other two buckets is never drawn as
a negative for that class. The designation-based rate is therefore a valid upper
bound that is loose by however much of it is not `coverage`, and that share is
what this measures.

**Kept out of `silence_rate.py` on purpose.** The rate reads three small JSON
files in ~20s; this reads VG's whole object table. Folding them together would
make the number nobody disputes pay for the correction, every run.

Usage::

    python silence_source.py --rate silence_rate.json --out silence_source.json
    python silence_source.py --rate silence_rate.json --per-class 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pile_config as pc
import silence_rate as sr
import verdict_store

#: Names carried by nearly every scene. Dropped from the *printed* example list
#: only -- never from a count, because "VG said 26 things and none of them was
#: the backpack" is the observation, and trimming the 26 hides how hard it was
#: trying.
_NOISE = frozenset({"sky", "ground", "wall", "floor", "background", "picture", "photo", "image"})

#: The three readings of a confirmed error, worst-for-the-pool first.
KINDS = ("coverage", "withheld", "folded")


def log(msg: str) -> None:
    print(f"[source] {msg}", flush=True)


def known_names(cls: str) -> set[str]:
    """Every spelling the build would recognise as *cls*, refusals included.

    `SCALE_VG_AMBIGUOUS` is in here because the question is whether VG *named*
    the object, not whether the loader accepted the name -- and because an
    ambiguous name keeps the image out of the negative pool just as a folded one
    does (`lift_ambiguous` withholds from both).
    """
    return {cls, *pc.SCALE_VG_NAMES.get(cls, ()), *pc.SCALE_VG_AMBIGUOUS.get(cls, ())}


def classify(cls: str, names: set[str]) -> str:
    """``"folded"``, ``"withheld"`` or ``"coverage"`` for one confirmed error.

    A near-miss is never guessed at. The tables are the only authority on what
    counts as the same object, and a string-similarity rule here would
    manufacture a finding out of `bike rack` and `book shelf`.
    """
    if names & {cls, *pc.SCALE_VG_NAMES.get(cls, ())}:
        return "folded"
    if names & set(pc.SCALE_VG_AMBIGUOUS.get(cls, ())):
        return "withheld"
    return "coverage"


def vg_names_by_image(records: list, wanted: set[int]) -> dict[int, Counter]:
    """``{image_id: Counter(name)}`` over *wanted*, every name VG put on the image.

    Deliberately unfiltered by :func:`pile_config.scale_vg_wanted`: the question
    is what VG *did* say, and reading only the names the loader folds would
    answer it with the answer already assumed.
    """
    out: dict[int, Counter] = {}
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in wanted:
            continue
        names: Counter = Counter()
        for obj in rec.get("objects") or []:
            for name in obj.get("names") or []:
                names[str(name).strip().lower()] += 1
        out[iid] = names
    return out


def census(
    rows: list[dict],
    answers: dict[str, dict[int, bool]],
    silent: dict[str, set[int]],
    names: dict[int, Counter],
    eligible: set[str],
) -> tuple[list[dict], dict]:
    """``(per-class rows, pooled)`` -- the three-way split, and the rate it corrects.

    **Both halves of the fraction move, and they move the same way.** Splitting
    the numerator alone would divide `coverage` errors by a denominator that
    still counts every pair VG *did* name, which understates the rate by exactly
    the share it just removed from the top. So ``vg_silent`` recounts the
    denominator on the same test the numerator uses -- pairs where no name on the
    image belongs to the class under any spelling the tables know -- and
    ``coverage_rate`` is the two together. That is the fraction a negative pool
    actually draws from.

    Pure, so both are testable without VG's object table.
    """
    out = []
    totals: Counter = Counter()
    for row in sorted(rows, key=lambda r: r["class"]):
        cls = row["class"]
        if cls not in eligible:
            continue
        pool = silent.get(cls, set())
        found = [i for i, yes in answers.get(cls, {}).items() if yes and i in pool]
        kinds: Counter = Counter(classify(cls, set(names.get(i, ()))) for i in found)
        totals.update(kinds)
        known = known_names(cls)
        vg_silent = sum(1 for i in pool if not (known & set(names.get(i, ()))))
        lo, hi = sr.wilson(kinds["coverage"], vg_silent)
        out.append(
            {
                "class": cls,
                "silent_pairs": row["silent_pairs"],
                "vg_silent_pairs": vg_silent,
                "found_present": len(found),
                **{k: kinds[k] for k in KINDS},
                "designation_rate": row["rate"],
                "coverage_rate": kinds["coverage"] / vg_silent if vg_silent else 0.0,
                "coverage_wilson95": [lo, hi],
            }
        )
    n = sum(r["silent_pairs"] for r in out)
    vg_n = sum(r["vg_silent_pairs"] for r in out)
    k = sum(totals[x] for x in KINDS)
    lo, hi = sr.wilson(totals["coverage"], vg_n)
    pooled = {
        "classes": [r["class"] for r in out],
        "silent_pairs": n,
        "vg_silent_pairs": vg_n,
        "found_present": k,
        **{k_: totals[k_] for k_ in KINDS},
        "designation_rate": k / n if n else 0.0,
        "coverage_rate": totals["coverage"] / vg_n if vg_n else 0.0,
        "coverage_wilson95": [lo, hi],
    }
    return out, pooled


def pick(answers: dict, silent: dict, eligible: set[str], per_class: int) -> list[tuple[str, int]]:
    """``[(class, image_id)]`` -- the first *per_class* confirmed errors of each pooled class.

    "First" is by image id, which is arbitrary and therefore not a choice: any
    ranking would be picking the examples that make the point, and the point is
    supposed to come from the census.
    """
    out = []
    for cls in sorted(eligible):
        found = sorted(i for i, yes in answers.get(cls, {}).items() if yes and i in silent.get(cls, set()))
        out.extend((cls, iid) for iid in found[:per_class])
    return out


def main() -> int:
    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent
    ap.add_argument("--rate", required=True, help="silence_rate.py's --out")
    ap.add_argument("--queue", default=str(base / "vgscale-3156" / "annotation_queue.jsonl"))
    ap.add_argument("--store", default=str(verdict_store.STORE))
    ap.add_argument("--objects", default=str(pc.DEMO_CACHE / "visual_genome" / "objects.json"))
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--deep-unprovable", type=int, default=0, help="translate the bound onto vg_scale_deep (#3723)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rate = json.loads(Path(args.rate).read_text())
    eligible = set(rate["pooled"]["classes"])
    queue = [json.loads(x) for x in Path(args.queue).read_text().splitlines() if x.strip()]
    silent = sr.silent_pairs(queue, tuple(pc.SCALE_CLASSES))
    answers = sr.read_answers(Path(args.store), {c: pc.review_name(c) for c in pc.SCALE_CLASSES})[0]

    # The whole queue, not only the images carrying a confirmed error: the
    # denominator is corrected on the same test as the numerator, and reading
    # names for the errors alone would leave it uncorrectable.
    log(f"loading {args.objects} ({Path(args.objects).stat().st_size / 1e9:.1f} GB)")
    names = vg_names_by_image(json.loads(Path(args.objects).read_text()), {int(r["image_id"]) for r in queue})
    log(f"{len(names)} of {len(queue)} queue images found in the object table")

    rows, pooled = census(rate["classes"], answers, silent, names, eligible)
    head = (
        f"{'class':<14}{'found':>7}{'coverage':>10}{'withheld':>10}{'folded':>8}"
        f"{'silent':>9}{'VG-silent':>11}{'desig.':>9}{'coverage':>10}"
    )
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['class']:<14}{r['found_present']:>7}{r['coverage']:>10}{r['withheld']:>10}"
            f"{r['folded']:>8}{r['silent_pairs']:>9,}{r['vg_silent_pairs']:>11,}"
            f"{r['designation_rate']:>9.2%}{r['coverage_rate']:>10.2%}"
        )
    print("-" * len(head))
    print(
        f"{'POOLED':<14}{pooled['found_present']:>7}{pooled['coverage']:>10}{pooled['withheld']:>10}"
        f"{pooled['folded']:>8}{pooled['silent_pairs']:>9,}{pooled['vg_silent_pairs']:>11,}"
        f"{pooled['designation_rate']:>9.2%}{pooled['coverage_rate']:>10.2%}"
    )

    samples = pick(answers, silent, eligible, args.per_class)
    examples = []
    print(f"\n{'class':<14}{'image':>9}  {'kind':<10}{'names':>6}  what VG said instead")
    print("-" * 108)
    for cls, iid in samples:
        got = names.get(iid, Counter())
        shown = [n for n, _ in got.most_common() if n not in _NOISE][:12]
        kind = classify(cls, set(got))
        examples.append(
            {
                "class": cls,
                "image_id": iid,
                "kind": kind,
                "vg_names_total": sum(got.values()),
                "vg_names_distinct": len(got),
                "vg_named": shown,
                "matched": sorted(set(got) & known_names(cls)),
            }
        )
        print(f"{cls:<14}{iid:>9}  {kind:<10}{sum(got.values()):>6}  {', '.join(shown[:8])}")
    print("-" * 108)

    spoke = pooled["withheld"] + pooled["folded"]
    log(
        f"{spoke} of {pooled['found_present']} confirmed errors "
        f"({spoke / max(pooled['found_present'], 1):.0%}) are images VG DID name -- "
        f"{pooled['folded']} under a spelling the build folds and {pooled['withheld']} under one it refuses. "
        "Neither can reach a negative pool, so neither is contamination."
    )
    log(
        f"designation-based {pooled['designation_rate']:.2%} of {pooled['silent_pairs']:,} pairs "
        f"-> VG-silence {pooled['coverage_rate']:.2%} "
        f"[{pooled['coverage_wilson95'][0]:.2%}, {pooled['coverage_wilson95'][1]:.2%}] "
        f"of {pooled['vg_silent_pairs']:,}"
    )

    # The screening allowance is carried over in POINTS, not recomputed: #3768's
    # below-cut slates asked whether the object is there, never which name VG
    # gave it, so there is no split of them to apply and pretending otherwise
    # would shrink the one term that is already an assumption. Conservative in
    # the direction a bound should be.
    allowance = rate["pooled"]["bound"] - rate["pooled"]["wilson95"][1]
    pooled["screening_allowance"] = allowance
    pooled["bound"] = min(1.0, pooled["coverage_wilson95"][1] + allowance)
    log(
        f"bound {pooled['bound']:.2%} = {pooled['coverage_wilson95'][1]:.2%} Wilson upper "
        f"+ {allowance:.2%} screening allowance carried from silence_rate.py unsplit"
    )
    if args.deep_unprovable:
        log(
            f"vg_scale_deep: at most {args.deep_unprovable * pooled['bound']:.0f} of its "
            f"{args.deep_unprovable:,} unprovable negatives are contaminated ({pooled['bound']:.2%}) -- "
            "the number to quote, since a pool draws only from VG-silent pairs"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps({"classes": rows, "pooled": pooled, "examples": examples}, indent=1) + "\n"
        )
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
