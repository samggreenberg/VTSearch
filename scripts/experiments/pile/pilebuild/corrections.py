"""Human verdicts on ``(image, class)`` pairs, and the one box-space crossing.

Two things about a verdict have to survive the trip to disk, and this module owns
both. They are the two halves of the same sentence: *what* a reviewer answered,
and *which question* they were shown.

The box space is the first, and it is made to happen exactly once.
A correction's boxes arrive normalised (they come from the app's ``region_box``);
every other box in the scale build is in pixels. Converting here, on the way in,
is what keeps the rest of the loader in a single space -- and #3281 is what the
other arrangement costs: a normalised box merged unconverted is normalised a
second time by the region write, which divides it by ~500 and parks it on the
frame origin, with the band derived from the same corrupted box so that nothing
downstream can see the disagreement.

The **rule** is the second (#3814). A class rule is not a constant: three rulings
landed in September on classes with rows already on disk, and a row that records
only its answer starts meaning something else the moment one does. Since the
``rule`` / ``rule_digest`` fields exist, a row carries the wording it was cast
under, and :func:`rule_state` is where that is compared against the wording in
force. Its one law is that **an absent stamp means unknown**: the 872 rows
written before the field cannot be back-filled from anything -- nothing on disk
ever said which wording they answered, which is why re-asking a human all 80
planter images was the only way to find out (#3778) -- so reading them as current
would be the failure itself rather than a rounding of it.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterable
from pathlib import Path

import pile_config as pc

#: The row's rule is the one in force: nothing has moved under it.
RULE_CURRENT = "current"
#: The name matches but the *wording behind it* has been edited since (#3756
#: rewrote `bench`'s Bad list and kept its name). The reviewer read the same
#: words off the detector; what a near-miss resolves to may have changed.
RULE_EDITED = "edited"
#: The row names a rule that is no longer in force. These answer a superseded
#: question, and they are the only state anybody can act on: they are exactly
#: the rows a recheck slate should be built from.
RULE_SUPERSEDED = "superseded"
#: The row records no rule at all. **Not** a synonym for `current`: it is the
#: state #3814 exists to make visible, and the 872 rows that predate the field
#: are all in it. Nothing can back-fill them -- the wording they were cast under
#: was never written down, which is why the planter recheck had to ask a human
#: all 80 images again (#3778).
RULE_UNKNOWN = "unknown"


def dropped_rows(old: list[dict], new: list[dict]) -> dict[str, int]:
    """``{source: n}`` for pairs the old file has and the new one does not.

    A regeneration is supposed to be a function of the inputs, so a *smaller*
    result means the inputs are not the ones that produced what is on disk.
    Measured on the live file: the defaults in this script reproduce 488 of its
    640 rows, and no invocation anyone could reconstruct reproduces all of them
    -- 379 rows come from verdict files, triage flags and adjudications nobody
    recorded. Overwriting on those terms is a silent deletion of human work,
    which is why :func:`main` refuses rather than warns.
    """
    have = {(int(r["image_id"]), r["class"]) for r in new}
    lost: dict[str, int] = {}
    for r in old:
        if (int(r["image_id"]), r["class"]) not in have:
            lost[r.get("source", "unknown")] = lost.get(r.get("source", "unknown"), 0) + 1
    return lost


def write_json_locked(path: Path, payload: object, indent: int = 1) -> None:
    """Write *payload* to *path* atomically, under an exclusive lock.

    Both halves matter for a file several sessions share on one pile (#3729).

    **Atomic**, because the old spelling was ``path.write_text(...)``: it
    truncates first, so a reader that arrives mid-write gets a half file and a
    writer that dies mid-write leaves one. The verdicts are the least
    reproducible thing here and were being rewritten in place with no landing
    strip.

    **Locked**, because two sessions regenerating this file at once is not
    hypothetical -- the pile is shared, and `corrections.json` is rebuilt from
    the verdict files by whoever ingests a slate. Without the lock the later
    writer silently wins with whatever inputs *it* could see; with it, the two
    runs serialise and the second reads the first's output. The lock is held on
    a sidecar rather than on the file itself so that the ``os.replace`` below
    cannot pull the locked inode out from under a waiter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=indent) + "\n")
        os.replace(tmp, path)


def load_corrections() -> dict[tuple[int, str], dict]:
    """``{(image_id, class): verdict}`` from the corrections file, if any.

    Verdicts, not corrections: a row exists for every reviewed ``(image, class)``
    pair whether or not the human disagreed, so review *coverage* is knowable.
    Without that, "no bus here" and "nobody looked" are the same absence, and
    every rate computed afterwards is biased by an unknown amount.

    Written by ``ingest_slate.py``; absent until the first review lands, which
    is why this returns empty rather than failing.

    **Boxes here are NORMALISED, unlike VG's and COCO's** -- they come from the
    app's ``region_box``. The space is validated on the way in and converted to
    pixels once, by :func:`correction_boxes_px`, so that everything downstream
    of this function is in one space. See ``pile_config.CORRECTION_BOX_SPACE``.

    **It names, but does not refuse, rows whose rule has been ruled away.** A
    superseded row is not malformed -- the reviewer answered honestly, and the
    definition moved afterwards -- so refusing would stop every build over a debt
    that only a human re-review can settle. Reporting it here rather than only
    from a tool that is asked is deliberate, and it is the counting that makes it
    bearable: `unknown` is **excluded**, because 872 unstamped rows is a constant
    nobody can act on and a line printed every run is a line everybody learns to
    skip. What is left starts at zero and becomes non-zero exactly when a ruling
    lands on stamped rows, which is the moment worth interrupting somebody for.
    `rule_drift.py` is where the full four-state breakdown lives.
    """
    path = Path(os.environ.get("VTS_CORRECTIONS", pc.PILE / "corrections.json"))
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out: dict[tuple[int, str], dict] = {}
    for r in rows:
        assert_correction_box_space(r, path)
        out[(int(r["image_id"]), r["class"])] = r
    stale = superseded_rows(rows)
    if stale:
        classes = sorted({r["class"] for r in stale})
        print(
            f"[corrections] {len(stale)} of {len(rows)} rows answer a rule that has since been ruled away "
            f"({', '.join(classes)}) -- `python rule_drift.py --state superseded --ids` names them (#3814)",
            flush=True,
        )
    return out


def assert_correction_box_space(row: dict, path: Path) -> None:
    """Refuse a correction row whose boxes are not in the declared space.

    The space is a *declaration*, not a guess: a pixel-space box handed to the
    normalised path is undetectable once it has been divided by (W, H) -- the
    result is a small, plausible, entirely wrong box, which is #3281 in the
    other direction. Checking it here costs one comparison per row and is the
    only point at which the two spaces are still distinguishable.
    """
    space = row.get("box_space", pc.CORRECTION_BOX_SPACE)
    where = f"{path.name}: image {row.get('image_id')} / {row.get('class')!r}"
    if space != pc.CORRECTION_BOX_SPACE:
        raise SystemExit(f"{where}: box_space {space!r}, expected {pc.CORRECTION_BOX_SPACE!r}")
    for b in row.get("boxes") or []:
        if len(b) != 4:
            raise SystemExit(f"{where}: box {b} is not [x0, y0, x1, y1]")
        if not all(-1e-6 <= float(v) <= 1.0 + 1e-6 for v in b):
            raise SystemExit(
                f"{where}: box {b} has a coordinate outside [0, 1], so it is in PIXEL space "
                f"while the file declares {pc.CORRECTION_BOX_SPACE!r}"
            )
        if float(b[2]) <= float(b[0]) or float(b[3]) <= float(b[1]):
            raise SystemExit(f"{where}: box {b} is degenerate (x1 <= x0 or y1 <= y0)")


def rule_state(row: dict, in_force: dict[str, str] | None = None) -> str:
    """Where *row*'s recorded rule stands against the one now in force (#3814).

    One of :data:`RULE_CURRENT`, :data:`RULE_EDITED`, :data:`RULE_SUPERSEDED` or
    :data:`RULE_UNKNOWN`. *in_force* is the class's :func:`pile_config.rule_stamp`,
    passed in so the states are testable against a rule table this repository
    does not have to actually hold.

    **A row with no `rule` reads as unknown, never as current.** That asymmetry
    is the entire point: reading an unstamped row as current is precisely the
    silent re-interpretation #3814 was filed about, and it would make the 872
    legacy rows look answered.

    A row carrying a matching *name* and no digest reads as `current`. The name
    is what the reviewer was shown, so the row's claim is satisfied; the digest
    is the stricter, optional check, and its absence means the `edited` question
    cannot be asked of that row rather than that the answer is no.
    """
    recorded = row.get("rule")
    if not recorded:
        return RULE_UNKNOWN
    stamp = in_force if in_force is not None else pc.rule_stamp(row["class"])
    if recorded != stamp.get("rule"):
        return RULE_SUPERSEDED
    digest = row.get("rule_digest")
    if digest and digest != stamp.get("rule_digest"):
        return RULE_EDITED
    return RULE_CURRENT


def rule_states(rows: Iterable[dict], stamps: dict[str, dict[str, str]] | None = None) -> list[tuple[dict, str]]:
    """``[(row, state)]`` for every row, resolving each class's rule once.

    The lookup is hoisted because :func:`pile_config.rule_digest` hashes the
    rule's whole ``test`` -- a few hundred words for `truck` -- and a correction
    file holds hundreds of rows per class.
    """
    stamps = {} if stamps is None else dict(stamps)
    out = []
    for row in rows:
        cls = row["class"]
        if cls not in stamps:
            stamps[cls] = pc.rule_stamp(cls)
        out.append((row, rule_state(row, stamps[cls])))
    return out


def superseded_rows(rows: Iterable[dict], stamps: dict[str, dict[str, str]] | None = None) -> list[dict]:
    """The rows whose recorded rule has since been ruled away.

    The one state a build can usefully name, and the reason the build's report is
    not simply "rows that are not current": `unknown` is a large constant that
    decays only as new rows are cast, and a number nobody can act on is a number
    everybody learns to skip. This one is **zero until a ruling actually lands on
    stamped rows**, which is exactly the event worth interrupting a run for. The
    full four-state breakdown belongs to `rule_drift.py`, which is asked.
    """
    return [row for row, state in rule_states(rows, stamps) if state == RULE_SUPERSEDED]


def correction_boxes_px(row: dict, W: int, H: int) -> list[list[float]]:
    """A verdict's boxes in the pixel space of ``(W, H)``.

    ``(W, H)`` is the space the image's *other* boxes were measured in -- the
    COCO original for an anchored image, the VG copy otherwise -- because that
    is what the region write later divides by. Scaling up here and dividing down
    there is an exact round trip, so the stored box is the reviewer's box to the
    last bit rather than merely close to it.
    """
    return [[float(b[0]) * W, float(b[1]) * H, float(b[2]) * W, float(b[3]) * H] for b in (row.get("boxes") or [])]
