#!/usr/bin/env python3
"""Apply a recheck pass's verdicts to `corrections.json` (#3778).

A ruling changes what a class *is*, and every row cast under the old wording
then means something the table no longer says. The planter ruling (#3784) is the
first one to have done that to rows already on disk: `vase` claimed "flower pots,
planters" and said in as many words that a potted plant's pot is a vase, so 31
`vase` rows and 49 `bowl` rows were cast under a definition that is now
withdrawn. `make_class_recheck.py` put all 80 back in front of the reviewer as an
image-level question -- *does this photo contain one under the new rule?* -- and
this is the other end of that: 21 of them came back Bad, and a Bad here is a
positive that has to leave the class.

**A recheck re-answers an existing row and never writes a new one.** The slate
was built from the present rows of `corrections.json` itself, so every verdict in
it has a row to land on. An image with no row is therefore not a new correction
to add, it is evidence that the slate and the file have drifted apart -- so it
refuses rather than inventing membership from a question nobody asked about this
image.

**The rule in force is checked, not assumed.** The labelset records the rule
whose name the reviewer was voting under (#3612: that name is the only wording a
reviewer ever sees). If `SCALE_CLASS_RULES` has moved on since, these verdicts
answer a superseded question and applying them would silently apply the old
ruling under the new one's name. That is the failure this whole issue was: a
correction row records no rule version, so a ruling quietly changes what old rows
mean. Here it is at least checkable, so it is checked.

**And now it is recorded, on every row this pass re-answers** (#3814). The check
above proves which rule these verdicts answer; writing it onto the row is what
keeps the next person from having to prove it again. Both directions get the
stamp, because both are answers: a Bad retires the row and carries the rule that
retired it, and a **Good** changes nothing about the label while establishing the
one thing nothing else can -- that a human looked at this photo under the rule now
in force. 59 of the 80 planter images came back that way and the file had no way
to say so, which is precisely why all 80 had to be asked rather than 31.

That is also the only back-fill this design permits, and it is not really one:
the evidence is the labelset sitting beside the file, not a guess about what
somebody was shown. Rows nobody re-answered stay `unknown` and must.

**Why `present: false` and no boxes.** The question was asked of the image, not
of a box, deliberately (an image may hold a planter *and* a real bowl, in which
case the photo is still a positive and only its box is wrong -- a smaller and
separate question). `apply_corrections` answers a boxless `present: false` by
popping the class from the image's labels, so the image leaves every cell of that
class, keeps its pinned seat only if it is still eligible (it is not), and is
backfilled by rank. It also becomes a *provable negative* for that class, which
is the half a rejection usually cannot buy: a human looked at this photo and said
there is none.

The verdicts are read from the committed copies under
:data:`verdict_store.STORE` rather than from the live app or from scratch. A
labelset on scratch is purgeable and the app's copy is the thing #3729 is about;
the committed one is the record, and its diff is a diff of answers.

Usage::

    python apply_recheck.py --dry-run      # what would change
    python apply_recheck.py                # write it
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pile_config as pc
import verdict_store
from pilebuild.corrections import write_json_locked

#: The ``source`` of a row retired because a ruling moved the class boundary
#: under it. Distinct from `human_reject+adjudicated` (the reviewer rejected the
#: box and an adjudicator named the object) because nothing was wrong with these
#: rows when they were cast: the reviewer was right, the definition moved. An
#: audit that cannot tell those apart reads a ruling as a pile of annotator
#: errors.
SOURCE_RECHECK = "human_recheck"


def log(msg: str) -> None:
    print(f"[recheck] {msg}", flush=True)


def load_labelsets(classes: tuple[str, ...], store: Path | None = None) -> dict[str, dict]:
    """The committed recheck labelset for each class in *classes*."""
    store = store or verdict_store.STORE
    out = {}
    for cls in classes:
        path = store / f"LABELSETS__recheck__{cls}.json"
        if not path.exists():
            raise SystemExit(f"no committed recheck labelset for {cls!r} at {path}")
        out[cls] = json.loads(path.read_text())
    return out


def plan(
    labelsets: dict[str, dict],
    rows: list[dict],
    rules: dict[str, str],
    rule_digests: dict[str, str] | None = None,
) -> tuple[list[dict], Counter, list[str]]:
    """``(rows, stats, problems)`` -- the corrections file this pass implies.

    Pure, so the guards below are testable without a pile: *rows* is the
    corrections file as read, *rules* maps a class to the name of the rule
    currently in force, *rule_digests* to that rule's
    :func:`pile_config.rule_digest` (omit it to check the name alone, which is
    all a labelset written before #3814 can be held to), and the returned rows
    are the same rows with the Bad
    verdicts' entries retired and **every** re-answered row stamped with the rule
    it was re-answered under.

    Every problem is a refusal rather than a skip. Each one means the pass and
    the file disagree about what was reviewed, and there is no reading of that
    which a silent partial apply improves.
    """
    rule_digests = rule_digests or {}
    by_key = {(int(r["image_id"]), r["class"]): r for r in rows}
    stats: Counter = Counter()
    problems: list[str] = []
    retire: dict[tuple[int, str], str] = {}
    # Every pair this pass re-answered, Good or Bad, and the stamp it earns.
    stamp: dict[tuple[int, str], dict[str, str]] = {}

    for cls, ls in sorted(labelsets.items()):
        if ls.get("kind") != "recheck":
            problems.append(f"{cls}: labelset kind is {ls.get('kind')!r}, not 'recheck'")
            continue
        if ls.get("class") != cls:
            problems.append(f"{cls}: labelset says class {ls.get('class')!r}")
            continue
        in_force = rules.get(cls)
        if ls.get("rule") != in_force:
            problems.append(
                f"{cls}: voted under rule {ls.get('rule')!r}, but {in_force!r} is in force -- "
                f"these verdicts answer a superseded question"
            )
            continue
        # The same refusal one notch finer, for a labelset that records the
        # rule's DIGEST as well as its name (`bank_verdicts.py` writes both since
        # #3814). A rule can be rewritten in the body while the name stands still
        # -- #3756 changed `bench`'s Bad list and kept its name -- and a reviewer
        # who voted before that edit answered about a different boundary under
        # identical words. Weaker than a rename and refused the same way, because
        # the alternative is deciding on the reviewer's behalf which edits were
        # cosmetic.
        voted_digest = ls.get("rule_digest")
        if voted_digest and voted_digest != rule_digests.get(cls):
            problems.append(
                f"{cls}: voted under rule {in_force!r} as written at {voted_digest}, but its wording has been "
                f"EDITED since (now {rule_digests.get(cls)}) -- the name is unchanged, the test is not"
            )
            continue

        good = {_image_id(v) for v in ls.get("good", [])}
        bad = {_image_id(v) for v in ls.get("bad", [])}
        for iid in sorted(good & bad):
            problems.append(f"{cls}: image {iid} is in both good and bad")
        for iid in sorted(good | bad):
            row = by_key.get((iid, cls))
            if row is None:
                problems.append(f"{cls}: image {iid} has no row to re-answer (the slate and the file have drifted)")
            elif iid in good and not row.get("present"):
                problems.append(f"{cls}: image {iid} was re-confirmed present, but its row already says absent")
            # The rule the LABELSET carries, which the checks above have just
            # established is the one in force. The digest rides along only when
            # the labelset recorded one: a labelset that names a rule without
            # pinning its wording cannot vouch for the body, and taking the
            # body's digest from the table anyway would assert exactly what was
            # not checked.
            stamp[(iid, cls)] = {"rule": in_force or cls, **({"rule_digest": voted_digest} if voted_digest else {})}
        for iid in sorted(bad):
            retire[(iid, cls)] = in_force or cls
        stats[f"{cls}: re-confirmed"] += len(good)
        stats[f"{cls}: retired"] += len(bad)

    if problems:
        return rows, stats, problems

    out = []
    for r in rows:
        key = (int(r["image_id"]), r["class"])
        fields = stamp.get(key, {})
        if key not in retire:
            # A re-confirmation changes no answer and is not nothing: a human
            # looked at this photo under the rule now in force and said it still
            # holds one, which is the only thing that can move a row out of
            # `unknown` without inventing history (#3814). 59 of the 80 planter
            # images came back this way and the file could not say so.
            stamped, moved = _stamped(r, fields)
            stats["rows stamped"] += int(moved)
            out.append(stamped)
            continue
        if not r.get("present"):
            # Retired by an earlier run of this same pass, under this same rule.
            # The stamp is recoverable and true, so a re-run backfills it -- the
            # one back-fill #3814 permits, because the evidence is the labelset
            # sitting right here rather than a guess about what someone saw.
            stamped, moved = _stamped(r, fields)
            stats["already retired"] += 1
            stats["rows stamped"] += int(moved)
            out.append(stamped)
            continue
        stats["rows changed"] += 1
        out.append(
            {
                "image_id": key[0],
                "class": key[1],
                **fields,
                "present": False,
                "boxes": [],
                "source": SOURCE_RECHECK,
                "note": (
                    f"re-asked at image level under {retire[key]!r} after the planter ruling (#3778, #3784) "
                    f"and the reviewer said this photo holds none"
                ),
            }
        )
    return out, stats, []


def _stamped(row: dict, fields: dict[str, str]) -> tuple[dict, bool]:
    """``(row, moved)`` -- *row* carrying *fields*, and whether that changed it.

    Returning the original object when nothing moves is what keeps a second run
    a no-op all the way down to object identity, which is what the idempotence
    test actually asserts.
    """
    if not fields or all(row.get(k) == v for k, v in fields.items()):
        return row, False
    return {**row, **fields}, True


def _image_id(verdict: dict) -> int:
    """The VG image id a recheck verdict is about.

    `make_class_recheck.py` builds each dataset out of ``<image_id>.jpg``
    symlinks, so the filename *is* the id. Read from ``filename`` rather than
    ``name`` because the app rewrites the latter on a collision.
    """
    return int(Path(verdict["filename"]).stem)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="bowl,vase", help="classes whose recheck labelset to apply")
    ap.add_argument("--corrections", default=os.environ.get("VTS_CORRECTIONS", str(pc.PILE / "corrections.json")))
    ap.add_argument("--dry-run", action="store_true", help="report the change and write nothing")
    args = ap.parse_args()

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    path = Path(args.corrections)
    rows = json.loads(path.read_text())
    # `review_name`, not `SCALE_CLASS_RULES[c].name`: an unruled class's rule IS
    # its bare name, and that is what a slate's detector carries and what
    # `bank_verdicts.py` reads back out of it. Reading the table directly gave
    # such a class `""`, which no labelset can ever match.
    rules = {c: pc.review_name(c) for c in classes}
    digests = {c: pc.rule_digest(c) for c in classes}

    log(f"{len(rows)} rows in {path}")
    new_rows, stats, problems = plan(load_labelsets(classes), rows, rules, digests)
    for k, v in sorted(stats.items()):
        log(f"  {k:<28}{v:>5}")

    if problems:
        print("\nREFUSING to write: the pass and the file disagree about what was reviewed\n")
        for p in problems:
            print(f"    {p}")
        return 2

    changed = [r for r in new_rows if r.get("source") == SOURCE_RECHECK]
    for r in sorted(changed, key=lambda r: (r["class"], r["image_id"])):
        log(f"  retired {r['class']:<6} {r['image_id']}")
    if not (stats["rows changed"] or stats["rows stamped"]):
        log("nothing to do: every retirement is already applied and every re-answered row already carries its rule")
        return 0
    moved = f"{stats['rows changed']} retired, {stats['rows stamped']} stamped with their rule"
    if args.dry_run:
        log(f"--dry-run: {moved}, nothing written")
        return 0

    # Locked and atomic: `corrections.json` is shared by every session on this
    # pile and is last-writer-wins with no history (#3729).
    write_json_locked(path, new_rows)
    log(f"wrote {len(new_rows)} rows ({moved}) to {path}")
    log("run `python verdict_store.py export` to update the committed copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
