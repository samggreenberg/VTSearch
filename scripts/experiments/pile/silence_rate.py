#!/usr/bin/env python3
"""How often VG is silent about a class that is really there (#3696).

`vg_scale`'s negative pool used to rest on VG's silence: an image VG never named
a `car` in was taken to hold no car. #3670 ended that for the shipped set by
drawing all 9,900 shared negatives from the COCO-scored half, so contamination
there is 0 *by construction* -- and with it went the only population in which the
silence rate could still be measured. The rate is what justified the composition
in the first place (#3666 measured 1.40% pool error; #3667 priced it at 1.18x on
the FPR scale), and `provenance_shortcut.py` can only re-derive it by being
pointed at an archived pre-change cell, which is not a thing anyone re-measures.

**The exhaustive pass produces the number as a by-product, at no extra labour.**
Its queue is the off-COCO images VG named at least one class of *C* in
(`annotation_queue.py`), and the pass asks a human which of the 25 are present.
Every class VG did **not** name in one of those images is therefore a
measurement of VG's silence against a human reference -- tens of thousands of
pairs, rather than the few hundred a hand-annotated stratum could have afforded.
This counts them.

## What the number is, and is not

**An upper bound on the uniform off-COCO rate, not an estimate of it.** The
queue's images are selected for holding a class of *C*, so they are cluttered
scenes, and clutter correlates with holding more classes: #3667 found 3,247
images excluded for holding a *different* class of *C*, and #3679 measured the
scene-clutter shortcut at 2.5x at `@small`. Silence measured on cluttered scenes
is at least as high as silence measured on a uniform draw, so the rate here sits
above the one a consumer like `vg_scale_deep` actually wants. That is enough for
the consumer that is left: deep is pinned to the pre-#3670 draw and #3723's
`--verify` reports 6,264 of its negatives resting on VG's silence with nothing
able to check them, and "deep's contamination is at most X" is exactly what a
bound supplies. Pass ``--deep-unprovable`` to have that translation printed.

Three things push the other way, and each is counted rather than argued:

* **screening.** The pass is screened, not swept (#3760): OWLv2 over the queue,
  cut per class at ~95% recall, and only what clears the cut reaches a reviewer.
  A positive below the cut is a silence error nobody was shown, so the found
  count is a floor. #3768 drew 100 below-cut images for each of six classes and
  found **zero** positives; that sample is what sizes the allowance here, per
  class where it exists and pooled where it does not.
* **the question the reviewer was asked.** A slate asks *is this box one?* of
  the screen's best box, so a Bad is evidence about that box and not about the
  image -- a second instance the detector never boxed reads as absent. #3768's
  below-cut slates ask the wider *is there one anywhere in this image?*, which
  is the question this measurement actually wants, and they are the only rows
  here that answer it. Both count a Good as a silence error because both prove
  one; only the wider question can prove the negative, so the box slates push
  the found count down and never up.
* **coverage.** A class whose slate was banked part-way has above-cut candidates
  with no verdict. They are neither positives nor negatives, so they enter the
  bound at their worst case (all positive) and the class is excluded from the
  pooled figure. A class reads as thin here rather than as low.
* **the rule in force.** A verdict answers the rule its reviewer was shown, and
  three rulings have moved a rule's name since the slates it covers were voted.
  Folding those verdicts into a pooled rate would quietly answer today's
  question with yesterday's boundary -- the whole of #3814 -- so a class whose
  labelset names a superseded rule is reported on its own line and kept out of
  the pool. It is reported, not refused, because this script measures and does
  not write: dropping the evidence would lose more than naming its vintage does.

## What was given up

A designated off-COCO stratum would have measured the *uniform* rate directly
rather than bounding it, and it was dropped (#3696) because it would have to be
hand-annotated to be a reference and the pass supplies a bound for free. If a
decision ever turns on the uniform rate specifically, this cannot supply it, and
drawing the stratum afterwards means a different annotator cohort under a
different protocol.

Usage::

    python silence_rate.py --queue .../annotation_queue.jsonl \\
                           --dets .../owl_queue.jsonl --slates .../slates/slates.json
    python silence_rate.py ... --deep-unprovable 6264 --out silence_rate.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pile_config as pc
import verdict_store

#: Labelset kinds this reads, and what each one answers. `slate` and `pass25`
#: are the pass proper -- "is this box one?" over the screened candidates, and
#: the earlier one-dataset form of the same question. `belowcut` is #3768's
#: complement, drawn at random from under each class's cut and asked the wider
#: "is there one anywhere in this image?". The two are kept apart because they
#: are different questions (#3612) and they do different jobs here: only the
#: below-cut rows size the screening allowance, and only they can settle a
#: negative about an *image* rather than about a box.
PASS_KINDS = ("slate", "pass25")
BELOW_KIND = "belowcut"

#: `bank_verdicts.py` wrote no ``kind`` until the slates existed, so the two
#: `pass25` labelsets carry none. Their filename is the only thing that says
#: what they are, which is the shape that clobbered two banks (#3760) -- so the
#: fallback is keyed on the exact prefix and never on a substring.
_KIND_BY_PREFIX = {"LABELSETS__pass25__": "pass25", "LABELSETS__slate__": "slate", "LABELSETS__belowcut__": BELOW_KIND}


def log(msg: str) -> None:
    print(f"[silence] {msg}", flush=True)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """``(lower, upper)`` Wilson bounds -- the instrument `pool_contamination` uses.

    Both ends: the headline here is a rate that must be shown to be *small*, and
    for the classes that found nothing at all the upper end is the entire
    result.
    """
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denom = 1 + z2 / n
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def image_id(verdict: dict) -> int:
    """The VG image id a verdict is about.

    Every slate is built from ``<image_id>.jpg``, so the filename *is* the id.
    Read from ``filename`` rather than ``name``, which the app rewrites on a
    collision.
    """
    return int(Path(verdict["filename"]).stem)


def labelset_kind(path: Path, doc: dict) -> str | None:
    """The question a labelset answers, from its own field or its filename prefix."""
    kind = doc.get("kind")
    if kind:
        return str(kind)
    for prefix, fallback in _KIND_BY_PREFIX.items():
        if path.name.startswith(prefix):
            return fallback
    return None


def silent_pairs(queue: list[dict], classes: tuple[str, ...]) -> dict[str, set[int]]:
    """``{class: queue images VG did NOT name that class in}`` -- the denominator.

    A queue row's ``classes`` is what the image was *designated* for, which is
    exactly what VG named: the pass owes a judgement on everything else. Every
    class of *C* gets an entry even when the queue never designates it, because a
    class absent from the queue still has 3,397 silent pairs in it and dropping
    the key would read as "no denominator" rather than "no designations".
    """
    all_ids = {int(row["image_id"]) for row in queue}
    named: dict[str, set[int]] = {c: set() for c in classes}
    for row in queue:
        for cls in row["classes"]:
            named.setdefault(cls, set()).add(int(row["image_id"]))
    return {cls: all_ids - ids for cls, ids in named.items()}


def above_cut(dets: list[dict], cuts: dict[str, float]) -> dict[str, set[int]]:
    """``{class: images whose best box for it cleared that class's cut}``.

    The screen's own decision, recomputed from the detections and the cuts
    `make_class_slates.py` recorded, rather than read off the slates: a slate
    excludes the class's known As and anything already answered, so its size is
    the *reviewable* share of the candidates and not the candidates. Using it as
    the above-cut set would move those exclusions into the below-cut mass, where
    they would draw a screening allowance they have not earned.
    """
    out: dict[str, set[int]] = {c: set() for c in cuts}
    for row in dets:
        iid = int(row["image_id"])
        for det in row.get("dets") or []:
            cls = det["cls"]
            if cls in cuts and det["score"] >= cuts[cls]:
                out[cls].add(iid)
    return out


def read_answers(
    store: Path,
    rules: dict[str, str],
    digests: dict[str, str] | None = None,
) -> tuple[dict[str, dict[int, bool]], dict[str, dict[int, bool]], dict[str, dict], list[str]]:
    """``(pass answers, below-cut answers, provenance, problems)`` from the committed labelsets.

    Read from :data:`verdict_store.STORE` rather than from the live app or from
    scratch: a labelset on scratch is purgeable and the app's copy is what #3729
    is about, so the committed copy is the record and its diff is a diff of
    answers.

    *rules* maps a class to the rule name in force and *digests* to its
    :func:`pile_config.rule_digest`. A labelset naming a different rule is
    recorded with ``rule_in_force: False`` and a problem line -- not dropped, and
    not folded in. `measure` is what keeps it out of the pool.
    """
    digests = digests or {}
    answers: dict[str, dict[int, bool]] = {}
    below: dict[str, dict[int, bool]] = {}
    prov: dict[str, dict] = {}
    problems: list[str] = []

    for path in sorted(store.glob("LABELSETS__*.json")):
        doc = json.loads(path.read_text())
        # `LABELSETS__` is the *root* a copy came from, not a shape: the store
        # holds whole-dashboard exports and bare verdict lists under the same
        # prefix. Only a labelset is a mapping with a class on it.
        if not isinstance(doc, dict):
            continue
        kind = labelset_kind(path, doc)
        if kind not in (*PASS_KINDS, BELOW_KIND):
            continue
        cls = doc.get("class")
        if not cls or cls not in rules:
            problems.append(f"{path.name}: class {cls!r} is not in C")
            continue

        # `bank_verdicts.py` wrote no `rule` field until the slates existed, and
        # the two `pass25` labelsets predate it. Their `detector_name` IS the
        # wording the reviewer was shown, and `rule_of_review_name` is its
        # documented inverse -- reading it back is the difference between
        # "answered under a superseded rule" and "we did not look".
        voted = doc.get("rule") or pc.rule_of_review_name(doc.get("detector_name") or "") or None
        in_force = rules[cls]
        current = voted == in_force
        voted_digest = doc.get("rule_digest")
        if current and voted_digest and voted_digest != digests.get(cls):
            # A rule can be rewritten in the body while its name stands still --
            # #3756 did that to `bench` mid-pass -- and a reviewer who voted
            # before the edit answered a different boundary under identical
            # words. Weaker than a rename and counted the same way.
            current = False
            problems.append(
                f"{cls}: {path.name} voted under {in_force!r} as written at {voted_digest}, but its wording has "
                f"been EDITED since (now {digests.get(cls)}) -- the name is unchanged, the test is not"
            )
        elif not current:
            problems.append(
                f"{cls}: {path.name} voted under rule {voted!r}, but {in_force!r} is in force -- "
                "those verdicts answer a superseded question"
            )

        target = below if kind == BELOW_KIND else answers
        seen = target.setdefault(cls, {})
        good = {image_id(v) for v in doc.get("good", [])}
        bad = {image_id(v) for v in doc.get("bad", [])}
        for iid in sorted(good & bad):
            problems.append(f"{cls}: image {iid} is in both good and bad of {path.name}")
        for iid, present in [*((i, True) for i in good), *((i, False) for i in bad)]:
            if iid in seen and seen[iid] != present:
                problems.append(f"{cls}: image {iid} answered both ways across labelsets ({path.name})")
            seen[iid] = present

        entry = prov.setdefault(cls, {"rule_in_force": True, "labelsets": []})
        entry["rule_in_force"] = entry["rule_in_force"] and current
        entry["labelsets"].append({"file": path.name, "kind": kind, "rule": voted, "good": len(good), "bad": len(bad)})
    return answers, below, prov, problems


def screening_allowance(below: dict[str, dict[int, bool]], z: float = 1.96) -> tuple[dict[str, float], float, Counter]:
    """``(per-class upper bound on the below-cut rate, the pooled one, counts)``.

    #3768's samples are drawn at random from below each class's cut and asked the
    *image* question, so they measure what the screen let through and nothing
    else. A class with its own sample gets its own bound; a class without one
    gets the pooled bound, which is the assumption this makes and the reason the
    per-class counts come back with it.
    """
    counts: Counter = Counter()
    per_class: dict[str, float] = {}
    for cls, verdicts in below.items():
        hits = sum(1 for v in verdicts.values() if v)
        per_class[cls] = wilson(hits, len(verdicts), z)[1]
        counts[cls] = len(verdicts)
    pooled_hits = sum(1 for verdicts in below.values() for v in verdicts.values() if v)
    pooled_n = sum(len(v) for v in below.values())
    return per_class, wilson(pooled_hits, pooled_n, z)[1], counts


def classify_excluded(
    excluded: set[int], designated: set[int], controls: set[int]
) -> tuple[set[int], set[int], set[int]]:
    """``(known As, controls, unexplained)`` for verdicts outside a class's silent pairs.

    Three things put a verdict outside the population, and only the third is a
    fault. The earlier `pass25` form of the pass reviewed **one dataset** rather
    than a per-class slate, so its reviewer was shown, and voted on, images the
    silence question does not apply to:

    * a **known A** -- the queue designates this image for this class, so VG did
      name it. It is the answer to a different question, not a silence error;
    * a **control** -- one of `make_pass25.py`'s COCO-anchored images, mixed in
      to score the pass against a key the reviewer cannot see. Anchored by
      construction, so it is not in the off-COCO population at all;
    * **anything else** -- a verdict with no pair to land on, which is the review
      and the queue having moved apart (a rebuild orphaning reviews is the way
      that happens here: three rebuilds once retired 577 of 743).

    Measured on the shipped record, the first two account for **every** excluded
    verdict -- 87 `bench` (60 + 27) and 187 `bicycle` (115 + 72), 0 unexplained.
    Reporting all three as drift is what this exists to stop: an alarm that fires
    on the designed behaviour is one nobody reads on the day it means something.
    """
    known_a = excluded & designated
    control = (excluded - known_a) & controls
    return known_a, control, excluded - known_a - control


def measure(
    silent: dict[str, set[int]],
    answers: dict[str, dict[int, bool]],
    below_answers: dict[str, dict[int, bool]],
    above: dict[str, set[int]],
    prov: dict[str, dict],
    designated: dict[str, set[int]] | None = None,
    controls: set[int] | None = None,
    z: float = 1.96,
) -> tuple[list[dict], dict, list[str]]:
    """``(per-class rows, the pooled row, problems)``.

    Pure, so every guard below is testable without a pile or a GPU.

    *designated* is what the queue says VG named (so a verdict there is a known
    A) and *controls* the anchored images `make_pass25.py` mixes in. Both are
    supplied so that a verdict falling outside the silent pairs can be
    *classified* rather than merely counted -- see :func:`classify_excluded`.

    Each class's silent pairs split four ways -- answered, above the cut and
    unanswered, below the cut and unanswered, and (rarely) answered from under
    the cut -- and the bound is built from all of them. ``rate`` counts only what
    a human confirmed present, so it is a floor; ``bound`` adds the unreviewed
    above-cut mass at its worst case (every one a positive) and the unreviewed
    below-cut mass at #3768's measured ceiling, on top of the Wilson upper end.
    A class is ``eligible`` for the pooled figure only when its slate is finished
    and its verdicts answer the rule in force.
    """
    below_ub, pooled_below_ub, below_counts = screening_allowance(below_answers, z)
    rows: list[dict] = []
    problems: list[str] = []

    designated = designated or {}
    controls = controls or set()

    for cls in sorted(silent):
        pool = silent[cls]
        cand = above.get(cls, set()) & pool

        given = set(answers.get(cls, {})) | set(below_answers.get(cls, {}))
        known_a, control, unexplained = classify_excluded(given - pool, designated.get(cls, set()), controls)
        if unexplained:
            # The only one of the three that is drift. Named rather than dropped,
            # because a silently smaller denominator is the one error this script
            # cannot show -- the shape `apply_recheck` refuses.
            problems.append(
                f"{cls}: {len(unexplained)} verdict(s) are neither a known A nor a control and are not silent "
                f"pairs of the queue (e.g. {', '.join(str(i) for i in sorted(unexplained)[:3])}) -- "
                "the review and the queue have drifted"
            )

        seen = {iid: v for iid, v in answers.get(cls, {}).items() if iid in pool}
        seen_below = {iid: v for iid, v in below_answers.get(cls, {}).items() if iid in pool}
        # The wider question wins where both were asked: "there is none anywhere
        # in this image" settles the pair, and "this box is not one" does not.
        answered = {**seen, **seen_below}
        found = sum(1 for v in answered.values() if v)
        n = len(pool)
        unreviewed_above = len(cand - answered.keys())
        unreviewed_below = len((pool - cand) - answered.keys())
        ub_below = below_ub.get(cls, pooled_below_ub)

        lo, hi = wilson(found, n, z)
        slack = (unreviewed_above + ub_below * unreviewed_below) / n if n else 0.0
        entry = prov.get(cls, {})
        rows.append(
            {
                "class": cls,
                "silent_pairs": n,
                "candidates": len(cand),
                "answered": len(answered),
                "answered_below_cut": len(seen_below),
                "unreviewed_candidates": unreviewed_above,
                "unreviewed_below_cut": unreviewed_below,
                "excluded_known_a": len(known_a),
                "excluded_control": len(control),
                "excluded_unexplained": len(unexplained),
                "found_present": found,
                "rate": found / n if n else 0.0,
                "wilson95": [lo, hi],
                "below_cut_rate_ub": ub_below,
                "below_cut_sample": below_counts.get(cls, 0),
                "bound": min(1.0, hi + slack),
                "slate_complete": unreviewed_above == 0,
                "rule_in_force": bool(entry.get("rule_in_force", False)) and bool(answered),
                "labelsets": entry.get("labelsets", []),
            }
        )

    eligible = [r for r in rows if r["slate_complete"] and r["rule_in_force"] and r["answered"]]
    k = sum(r["found_present"] for r in eligible)
    n = sum(r["silent_pairs"] for r in eligible)
    lo, hi = wilson(k, n, z)
    slack = sum(r["below_cut_rate_ub"] * r["unreviewed_below_cut"] for r in eligible)
    pooled = {
        "classes": [r["class"] for r in eligible],
        "silent_pairs": n,
        "found_present": k,
        "rate": k / n if n else 0.0,
        "wilson95": [lo, hi],
        "bound": min(1.0, hi + (slack / n if n else 0.0)),
        "pooled_below_cut_rate_ub": pooled_below_ub,
        "below_cut_sample": sum(below_counts.values()),
    }
    return rows, pooled, problems


def print_table(rows: list[dict], pooled: dict, total_classes: int) -> None:
    """Per class, in the order the bound ranks them -- worst silence first."""
    head = f"{'class':<14}{'silent':>9}{'cand':>7}{'seen':>7}{'found':>7}{'rate':>8}{'95% CI':>16}{'bound':>8}  why"
    print("\n" + head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: (-r["bound"], r["class"])):
        lo, hi = r["wilson95"]
        ci = "[{:.2%}, {:.2%}]".format(lo, hi)  # noqa: UP032 - py310 forbids the nested quotes an f-string needs
        why = []
        if not r["answered"]:
            why.append("not started")
        elif not r["slate_complete"]:
            why.append(f"{r['unreviewed_candidates']} unreviewed")
        if r["answered"] and not r["rule_in_force"]:
            why.append("stale rule")
        print(
            f"{r['class']:<14}{r['silent_pairs']:>9,}{r['candidates']:>7,}{r['answered']:>7,}"
            f"{r['found_present']:>7,}{r['rate']:>8.2%}{ci:>16}{r['bound']:>8.2%}"
            f"  {', '.join(why)}"
        )
    print("-" * len(head))
    lo, hi = pooled["wilson95"]
    ci = "[{:.2%}, {:.2%}]".format(lo, hi)  # noqa: UP032 - as above
    print(
        f"{'POOLED':<14}{pooled['silent_pairs']:>9,}{'':>7}{'':>7}{pooled['found_present']:>7,}"
        f"{pooled['rate']:>8.2%}{ci:>16}{pooled['bound']:>8.2%}"
        f"  {len(pooled['classes'])} of {total_classes} classes"
    )


def main() -> int:
    # Deferred like `annotation_queue`'s: `setup_env` rewrites the import
    # machinery process-wide, and the arithmetic above is worth a unit test that
    # does not pay for that.
    pc.setup_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = pc.PILE.parent
    ap.add_argument("--queue", default=str(base / "vgscale-3156" / "annotation_queue.jsonl"))
    ap.add_argument("--dets", default=str(base / "vlm-3720" / "owl_queue.jsonl"), help="OWLv2 over the queue")
    ap.add_argument("--slates", default=str(base / "vlm-3720" / "slates" / "slates.json"), help="for the cuts")
    ap.add_argument("--store", default=str(verdict_store.STORE), help="the committed labelsets")
    ap.add_argument(
        "--controls",
        default=str(base / "vgscale-3156" / "pass25" / "controls.json"),
        help="make_pass25.py's key, so its anchored controls read as controls rather than as drift",
    )
    ap.add_argument("--deep-unprovable", type=int, default=0, help="translate the bound onto vg_scale_deep (#3723)")
    ap.add_argument("--out", default="", help="write the measurement as JSON")
    args = ap.parse_args()

    queue = [json.loads(x) for x in Path(args.queue).read_text().splitlines() if x.strip()]
    manifest = json.loads(Path(args.slates).read_text())
    cuts = {cls: float(body["cut"]) for cls, body in manifest.items()}
    dets = [json.loads(x) for x in Path(args.dets).read_text().splitlines() if x.strip()]
    log(f"{len(queue)} queue images, {len(dets)} detection rows, cuts for {len(cuts)} classes")

    classes = tuple(pc.SCALE_CLASSES)
    silent = silent_pairs(queue, classes)
    above = above_cut(dets, cuts)
    designated = {cls: {int(r["image_id"]) for r in queue if cls in r["classes"]} for cls in classes}
    controls: set[int] = set()
    if Path(args.controls).exists():
        controls = {int(i) for i in json.loads(Path(args.controls).read_text()).get("controls", {})}
        log(f"{len(controls)} anchored controls from {args.controls}")
    else:
        # Without the key its controls cannot be told from orphaned reviews, so
        # they would report as drift. Say so rather than letting the alarm lie.
        log(f"NOTE: no control key at {args.controls}; pass25's mixed-in controls will read as unexplained")

    rules = {c: pc.review_name(c) for c in classes}
    digests = {c: pc.rule_digest(c) for c in classes}
    answers, below, prov, problems = read_answers(Path(args.store), rules, digests)
    rows, pooled, more = measure(silent, answers, below, above, prov, designated, controls)
    problems += more

    # The slates were built from these detections at these cuts, minus each
    # class's known As and whatever the earlier pass had already answered. If
    # that identity does not hold, the cuts and the detections are not the ones
    # the reviewer's slate was cut at, and every candidate count below is about
    # a different screen.
    for cls, body in sorted(manifest.items()):
        reviewable = len((above.get(cls, set()) & silent.get(cls, set())))
        if body["n"] > reviewable:
            problems.append(
                f"{cls}: slates.json holds {body['n']} candidates but the detections clear the cut on only "
                f"{reviewable} silent pairs -- these are not the screen that built that slate"
            )

    print_table(rows, pooled, len(classes))
    print()
    log(f"{sum(r['silent_pairs'] for r in rows):,} silent (image, class) pairs over {len(queue):,} queue images")
    log(
        f"POOLED over {len(pooled['classes'])} finished, rule-current classes: {pooled['rate']:.2%} "
        f"[{pooled['wilson95'][0]:.2%}, {pooled['wilson95'][1]:.2%}], bound {pooled['bound']:.2%}"
    )
    log(
        f"below-cut allowance: {pooled['pooled_below_cut_rate_ub']:.2%} pooled, "
        f"from {pooled['below_cut_sample']} randomly sampled below-cut images (#3768)"
    )
    log("CONDITIONING: an upper bound on the UNIFORM off-COCO rate, not an estimate of it -- the queue's")
    log("  images are selected for holding a class of C, and clutter correlates with holding more (#3667, #3679).")
    log("  The slates ask about the screen's best BOX, so a second instance it never boxed reads as absent.")
    excluded = sum(r["excluded_known_a"] + r["excluded_control"] for r in rows)
    if excluded:
        log(
            f"{excluded} verdict(s) sit outside the silent pairs and are not counted: "
            f"{sum(r['excluded_known_a'] for r in rows)} known As (VG named the class) and "
            f"{sum(r['excluded_control'] for r in rows)} anchored controls (COCO answers for them)"
        )

    if args.deep_unprovable:
        log(
            f"vg_scale_deep: at most {args.deep_unprovable * pooled['bound']:.0f} of its "
            f"{args.deep_unprovable:,} unprovable negatives are contaminated ({pooled['bound']:.2%})"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "queue_images": len(queue),
                    "classes": rows,
                    "pooled": pooled,
                    "problems": problems,
                    "conditioning": {
                        "target": "uniform off-COCO silence rate",
                        "direction": "upper bound",
                        "why": "the queue's images are selected for holding a class of C (#3667, #3679)",
                        "screening": "OWLv2 cut per class at ~95% recall (#3760); below-cut allowance from #3768",
                    },
                },
                indent=1,
            )
            + "\n"
        )
        log(f"wrote {args.out}")

    for p in problems:
        log(f"PROBLEM: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
