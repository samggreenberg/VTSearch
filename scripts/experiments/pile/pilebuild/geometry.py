"""Checks a region box must pass whoever wrote it, and the derived-label digest."""

from __future__ import annotations

import hashlib
import json

import pile_config as pc


def scale_label_digest(medias: dict[int, dict]) -> str:
    """A hash of exactly what ``vg_scale_any`` copies out of ``vg_scale``.

    ``vg_scale_any`` is a *relabel* of the built ``vg_scale`` pickle, so a fix to
    ``vg_scale``'s labels, boxes or bands leaves the derived cell holding the old
    ones -- with the right media count, the right vectors and a healthy-looking
    ``--verify``. #3281 is the case: the box repair moves 97 images between
    bands, and ``build_pile.py --force vg_scale`` alone would ship a
    ``vg_scale_any`` still carrying the pre-repair regions.

    Vectors are deliberately not in it: they are identical by construction (the
    derived build never re-embeds) and ``cell_fingerprint`` already covers them.
    What this pins is the half a rebuild of the parent can actually change.
    """
    h = hashlib.sha256()
    for mid in sorted(medias):
        m = medias[mid]
        h.update(
            json.dumps(
                [
                    mid,
                    m.get("category"),
                    m.get("categories"),
                    m.get("evaluable_categories"),
                    [
                        [r.get("label"), [round(float(v), 9) for v in r.get("box") or []]]
                        for r in m.get("regions") or []
                    ],
                ],
                sort_keys=True,
            ).encode()
        )
    return h.hexdigest()


def region_geometry_problems(medias: dict[int, dict]) -> list[str]:
    """Geometry no honest normalised region box can have (#3281).

    The band check in :func:`pilebuild.audit.verify` cannot see a coordinate-space mistake made
    *before* banding, because the band is computed from the very box it would be
    checking: crush a box to the origin and it is filed under ``@small``, where
    a sub-pixel area is exactly what the band's name claims. Both sides move
    together and the cell stays self-consistent. So the box has to be checked
    against the frame rather than against its own label.

    Two rules, and they are different in kind:

    * **Sub-pixel** is absolute. A side below ``MIN_BOX_SIDE`` is under one pixel
      on any image the pile holds, so no such box was ever drawn or annotated.
      One is a failure.
    * **Crushed to the origin** is a rate. A real small object can sit in the
      top-left corner and 1.2% of healthy boxes do, so a single hit proves
      nothing; a *population* of them is a double-normalise, which put 100% of
      the affected images there.
    """
    problems: list[str] = []
    n_boxes = 0
    subpixel: list[str] = []
    cornered = 0
    edge = pc.CORNER_AREA_FRAC**0.5
    for mid, m in medias.items():
        for r in m.get("regions") or []:
            b = r.get("box") or []
            if len(b) != 4:
                problems.append(f"media {mid} / {r.get('label')!r}: box {b} is not [x0, y0, x1, y1]")
                continue
            n_boxes += 1
            if (b[2] - b[0]) < pc.MIN_BOX_SIDE or (b[3] - b[1]) < pc.MIN_BOX_SIDE:
                subpixel.append(f"media {mid} / {r.get('label')!r} {[round(v, 6) for v in b]}")
            if b[2] <= edge and b[3] <= edge:
                cornered += 1
    if subpixel:
        problems.append(
            f"{len(subpixel)} region boxes are sub-pixel (side < {pc.MIN_BOX_SIDE:g} of the frame), "
            f"which no drawn or annotated box is -- e.g. {'; '.join(subpixel[:3])}"
        )
    if n_boxes and cornered / n_boxes > pc.MAX_CORNER_RATE:
        problems.append(
            f"{cornered}/{n_boxes} ({cornered / n_boxes:.1%}) of region boxes lie wholly inside the "
            f"top-left {pc.CORNER_AREA_FRAC:.0%} of the frame, against a healthy rate near 1% -- "
            f"the signature of a box normalised twice"
        )
    return problems
