#!/usr/bin/env python3
"""The negative pass's 2,400 judgements, read from the committed CSV (#4012).

The pass's raw material -- ``negbank/*.json``, ``polarity.json``, ``seeded.json``
-- was deleted with its study dir on 2026-09-18 (#4001), is not in #3729's
record, and is not in the app's detector store either, which today holds only
the three confirm-the-box detectors. #4011 gave the readers a guard, which was
honest and terminal. This restores them instead, because the judgements
themselves survive in
``docs/experiments/2026-09-06-shipped-pool-3666/verdicts.csv``.

**They survive in the one form still readable: ``present`` is already
POLARITY-RESOLVED.** The raw labels meant ``good == clean`` for some detectors
and ``good == present`` for others -- the pass flipped mid-review -- and
``polarity.json``, the file that said which, went with the directory. A raw
label string would be unresolvable today. Storing what an answer *meant* rather
than what it looked like on arrival is the whole reason 2,400 judgements are
still legible; see #4012.

What the CSV cannot give back: the export wrapper (detector name,
``origin_name`` paths, raw label strings) and the ability to re-derive a verdict
from a click. No judgement and no box -- the negative pass drew none.

Two properties of the file that the readers here depend on, pinned by
``tests_lib/meta/test_negative_pass_verdicts.py`` so a rewrite cannot quietly
break them: all twelve passes judged the **same 200 images**, and a stratum
belongs to an image rather than to a pass.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

#: The committed record of the pass. Repo-relative so it reads the same from any
#: working directory -- the readers used to take a `/expscratch` path, which is
#: what made them deletable.
VERDICTS = (
    Path(__file__).resolve().parents[3] / "docs" / "experiments" / "2026-09-06-shipped-pool-3666" / "verdicts.csv"
)

#: The ten passes that lived in ``negbank/``. `Backpack` and `Umbrella` finished
#: after the last banking run and existed only in the app's detector store, so
#: the two readers that globbed ``negbank/*.json`` never saw them. Kept because
#: a published number depends on it: the union rate in #3666 is over these ten.
NEGBANK_PASSES = (
    "Bench",
    "Book",
    "Cell Phone",
    "Chair",
    "Clock",
    "Fire Hydrant",
    "Outdoor Objects",
    "Stop Sign",
    "Table Objects",
    "Vehicles",
)

#: The three groups `score_negative_pass.py` asks about: several classes judged
#: together, as against the nine single-class passes.
GROUP_PASSES = ("Vehicles", "Outdoor Objects", "Bench")


@dataclass(frozen=True)
class Verdict:
    """One image, one pass, one resolved answer."""

    pass_name: str
    members: tuple[str, ...]
    image_id: int
    #: The reviewer's answer, polarity already applied: an object of this pass's
    #: classes is in this image.
    present: bool
    #: Written from COCO by `seed_pos.py` to stop the trainer starving, not
    #: judged by anyone. Scoring one against COCO would be circular, so every
    #: estimator here drops them unless asked not to.
    seeded: bool
    stratum: str
    driver: str
    exhaustive: bool


def load(path: str | Path = VERDICTS) -> list[Verdict]:
    """Read the committed verdicts. Raises if the file is missing -- it is committed."""
    p = Path(path)
    with p.open(newline="") as fh:
        return [
            Verdict(
                pass_name=r["pass"],
                members=tuple(r["members"].split("|")),
                image_id=int(r["image_id"]),
                present=bool(int(r["present"])),
                seeded=bool(int(r["seeded"])),
                stratum=r["stratum"],
                driver=r["driver"],
                exhaustive=r["exhaustive"] == "yes",
            )
            for r in csv.DictReader(fh)
        ]


def strata(rows: list[Verdict]) -> dict[int, str]:
    """``image_id -> stratum``. A stratum is a property of the image, not the pass."""
    return {v.image_id: v.stratum for v in rows}


def for_pass(rows: list[Verdict], name: str) -> list[Verdict]:
    return [v for v in rows if v.pass_name == name]


def found(
    rows: list[Verdict],
    *,
    include_seeded: bool = False,
    only: tuple[str, ...] | None = None,
) -> dict[int, list[str]]:
    """``image_id -> the passes whose reviewer found one``, for the union rate.

    ``include_seeded`` exists for one reason: to reproduce a published number
    that was computed before the seeded rows were separated out. New work wants
    the default.
    """
    out: dict[int, list[str]] = {}
    for v in rows:
        if only is not None and v.pass_name not in only:
            continue
        if v.present and (include_seeded or not v.seeded):
            out.setdefault(v.image_id, []).append(v.pass_name)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval as percentages -- what #3666 published."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def rate(rows: list[Verdict], *, stratum: str = "random", passes: tuple[str, ...] | None = None) -> tuple[int, int]:
    """``(finds, n)`` for one stratum, pooled over *passes* -- per-row, not union."""
    sel = [v for v in rows if v.stratum == stratum and not v.seeded and (passes is None or v.pass_name in passes)]
    return sum(1 for v in sel if v.present), len(sel)
