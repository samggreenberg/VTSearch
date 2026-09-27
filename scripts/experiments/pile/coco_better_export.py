#!/usr/bin/env python3
"""Cut a study-ready set out of `coco_better_full`: any prevalence, per-class negatives.

`SCALE_PREVALENCE`, `SCALE_N_POS` and `SCALE_N_NEG` were fixed when a pile was
built, so moving any of them meant a rebuild and a re-embed (#3987). Over an
exhaustively annotated corpus every ``(image, class)`` pair is answered, and once
the whole corpus is embedded (#4017, #4022) a cell stops being a designation and
becomes a **filter**. This is that filter.

**Nothing here embeds anything.** The output is a manifest of media ids; a study
loads the full-corpus cell it wants and keeps those ids. Prevalence becomes a
sampling parameter rather than a property of the pile.

## Two pools, and the default is the one #3986 argues for

A shared-pool negative holds **no class in C at all**, which is stricter than
scoring needs -- a negative for `book` need only lack `book` -- and it draws an
unrepresentative sample: such an image is barren by construction, so a detector
scored against it answers an easier question than the app poses, and can learn
*an image containing a B cannot contain an A* (#3667 measured that shortcut at
1.88 on its FPR-inflation scale).

Per-class, the negatives for `book@small` are every image the cell can be scored
against that is not one of its positives -- which is what ``evaluable_categories``
already records. Measured on the built cells: **~100,000 negatives per cell
against the designated 9,900, about half of them holding another class in C**,
where the shared pool is 0% by construction.

``--pool shared`` reproduces the old shape for an arm that needs to be compared
against a published number.

## Prevalence is a ratio, so the envelope is two-dimensional

N = P(1-pi)/pi. The corpus caps N at ~100k per cell, so a target pi and a target
positive count cannot both be chosen freely: `car@large` at pi=0.001 would need
1.29M negatives against 96,458 available, while the same cell at 100 positives
needs 99,900 and fits. ``--envelope`` prints what each cell can actually reach,
and an infeasible request is refused with the two numbers that would make it
feasible rather than a silently thinner set.

## The draw reproduces the designated build

Positives are ordered by :func:`~pilebuild.scale_core.rank`, the same
hash-of-(cell, image_id) the builder uses, so an export at ``SCALE_N_POS`` with
no other constraint selects the same images `coco_better` designated. That is
what makes a query-time cell comparable with a build-time one rather than merely
similar, and it is pinned by a test.

## Size is an axis on BOTH sides of the split (#4044)

A cell is ``class@band``, and until now that band was the train set *and* the
test set: train on small cars, test on small cars. Two things follow from COCO
being exhaustively annotated, and this cuts both.

``--test-bands`` lists, beside the cell, the positives of the class's **other**
bands, so an arm trained at one size can be scored at all three. That is legal
without any re-draw because the three bands of a class **share one negative
pool** by construction -- the ``3 *`` in ``SCALE_PREVALENCE`` -- which
``--check-bands`` asserts rather than assumes. The same fact is why there is no
per-band FPR to export: a negative holds no instance of the class, so it has no
size for the class, and an arm has exactly one false-positive rate. What
decomposes is the miss rate.

``--train-mix`` draws the positives across bands at chosen proportions instead
of from one band -- ``natural`` reads the shares off the corpus, which is what
"how often is a car small in the wild" actually means here. **Only the train
side needs this.** An FNR over disjoint per-band cohorts is linear in the mix,
so any *test*-side mix is the mix-weighted average of a 3x3 matrix already
measured; training is not linear in its input, so each train-side mix is a real
new arm and has to be run.

    python coco_better_export.py --cell bus@small --positives 100 --prevalence 0.01
    python coco_better_export.py --cell car@small --test-bands all --out car_small.json
    python coco_better_export.py --cell car --train-mix natural --positives 300
    python coco_better_export.py --mix-census
    python coco_better_export.py --envelope
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import pile_config as pc  # noqa: E402
from pilebuild.scale_core import rank  # noqa: E402

#: The cheapest full-corpus column to read membership from. Cell membership is
#: identical across columns -- it comes from the annotations, not the vectors --
#: so there is no reason to page in a patch shard to answer "which images".
MEMBERSHIP_CELL = "coco_better_full__clip.pkl"

POOLS = ("per_class", "shared")


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def load_membership(path: Path | None = None) -> dict[int, dict]:
    """``{image_id: {"cells": [...], "evaluable": [...]}}`` for the whole corpus."""
    from _cells_io import load_medias  # noqa: PLC0415

    src = path or (pc.EMBEDDINGS / MEMBERSHIP_CELL)
    if not src.exists():
        raise SystemExit(f"missing {src}; build coco_better_full first (#4017)")
    return {
        iid: {"cells": list(d.get("categories") or []), "evaluable": list(d.get("evaluable_categories") or [])}
        for iid, d in load_medias(src).items()
    }


def candidates(members: dict[int, dict], cell: str, pool: str) -> tuple[list[int], list[int]]:
    """``(positives, negatives)`` available for *cell*, before any sampling."""
    if pool not in POOLS:
        raise SystemExit(f"unknown pool {pool!r}; known: {', '.join(POOLS)}")
    pos = [i for i, m in members.items() if cell in m["cells"]]
    if pool == "shared":
        # The old shape: an image holding nothing in C at all.
        neg = [i for i, m in members.items() if not m["cells"]]
    else:
        pos_set = set(pos)
        neg = [i for i, m in members.items() if cell in m["evaluable"] and i not in pos_set]
    return pos, neg


def cells_of_class(members: dict[int, dict], cls: str) -> list[str]:
    """Every ``cls@band`` cell the corpus carries, sorted.

    Read off the membership rather than spelled from ``pile_config.BOX_BANDS``,
    so a band that is configured but empty does not appear as a cell with no
    images in it.
    """
    prefix = cls + "@"
    out: set[str] = set()
    for m in members.values():
        for key in ("cells", "evaluable"):
            out |= {c for c in m[key] if c.startswith(prefix)}
    return sorted(out)


def band_of(cell: str) -> str:
    """``"small"`` for ``"car@small"``. A class name never holds an ``@``."""
    return cell.partition("@")[2]


def class_candidates(members: dict[int, dict], cls: str, pool: str) -> tuple[dict[str, list[int]], list[int]]:
    """``({band: positives}, negatives)`` for a whole class.

    The negatives are the class's, not a band's, and that is not a widening:
    :func:`check_shared_negatives` asserts the three bands already draw from one
    set. A positive of *any* band is excluded from the negatives of *every*
    band, exactly as ``scale_core._evaluable`` already has it -- an image with a
    large car is not a small-car negative.
    """
    cells = cells_of_class(members, cls)
    if not cells:
        raise SystemExit(f"no cells for class {cls!r}; is it in SCALE_CLASSES?")
    by_band: dict[str, list[int]] = {}
    for cell in cells:
        pos, _ = candidates(members, cell, pool)
        if pos:
            by_band[band_of(cell)] = pos
    held = {i for ids in by_band.values() for i in ids}
    if pool == "shared":
        neg = [i for i, m in members.items() if not m["cells"]]
    else:
        neg = [i for i, m in members.items() if any(c in m["evaluable"] for c in cells) and i not in held]
    return by_band, sorted(neg)


def check_shared_negatives(members: dict[int, dict], cls: str, pool: str) -> tuple[bool, str]:
    """Do the class's bands really draw their negatives from one set?

    The cross-band test rests on it: three FNRs are only readable against **one**
    FPR if the images behind that FPR are the same for each band. It is true by
    construction today, and a construction is exactly the kind of thing that
    stops being true without anyone editing the file that states it -- #3667
    changed what a negative is and #3986 changed which pool it comes from, both
    after the guarantee was written down.
    """
    cells = cells_of_class(members, cls)
    sets = {cell: set(candidates(members, cell, pool)[1]) for cell in cells}
    first = next(iter(sets.values()))
    odd = {cell: len(ids ^ first) for cell, ids in sets.items() if ids != first}
    if odd:
        return False, f"{cls}: bands do NOT share a negative pool ({pool}); symmetric differences {odd}"
    return True, f"{cls}: {len(cells)} bands share {len(first):,} negatives ({pool} pool)"


def natural_mix(members: dict[int, dict], cls: str) -> dict[str, float]:
    """The share of *cls*'s positives sitting in each band, as the corpus has it.

    On an exhaustively annotated source this is a **measurement** of how often
    the object really is that size, not a design choice -- which is what makes
    ``--train-mix natural`` the arm that asks the question a user of the app
    actually faces.
    """
    counts = {band: len(ids) for band, ids in class_candidates(members, cls, "per_class")[0].items()}
    total = sum(counts.values())
    return {band: n / total for band, n in sorted(counts.items())} if total else {}


def parse_mix(spec: str) -> "str | dict[str, float]":
    """``"natural"``, or ``"small=1,medium=2,large=1"`` as weights."""
    if spec == "natural":
        return spec
    out: dict[str, float] = {}
    for part in spec.split(","):
        band, sep, weight = part.strip().partition("=")
        if not sep:
            raise SystemExit(f"bad --train-mix {spec!r}; want 'natural' or 'small=1,medium=2,large=1'")
        out[band.strip()] = float(weight)
    return out


def quota(shares: dict[str, float], n_pos: int, available: dict[str, int]) -> dict[str, int]:
    """Per-band positive counts, largest-remainder, capped by supply.

    Largest-remainder so the quotas sum to *n_pos* exactly: a mixed arm whose
    realised size drifts with the *rounding* cannot be compared against the pure
    arms it exists to be compared against. Supply is the other story: see the
    body. Single-sourced in spirit with
    :func:`vtscore.eval.scale_bands._quota`, and pinned against it by
    ``test_the_two_quota_rules_agree``.
    """
    exact = {b: shares[b] * n_pos for b in shares}
    out = {b: int(v) for b, v in exact.items()}
    # Largest-remainder first, so rounding alone never costs the set a seat.
    order = sorted(shares, key=lambda b: (-(exact[b] - int(exact[b])), b))
    for b in order[: n_pos - sum(out.values())]:
        out[b] += 1
    # Then cap by supply, and DELIBERATELY do not backfill the deficit into the
    # bands that do have headroom. Handing a short band's seats to its
    # neighbours keeps the count round at the price of quietly running a
    # different mix from the one that was asked for -- which is the
    # wrong-but-plausible arm, not a smaller one. A mix the corpus cannot serve
    # comes out short and says so.
    return {b: min(n, available.get(b, 0)) for b, n in out.items()}


def negatives_for(n_pos: int, prevalence: float) -> int:
    """N = P(1-pi)/pi, the count that puts *n_pos* positives at *prevalence*."""
    if not 0 < prevalence < 1:
        raise SystemExit(f"prevalence must be in (0, 1), got {prevalence}")
    return round(n_pos * (1 - prevalence) / prevalence)


def select(members: dict[int, dict], cell: str, n_pos: int, prevalence: float, pool: str = "per_class") -> dict:
    """One study-ready selection, or a refusal naming what would make it fit."""
    pos, neg = candidates(members, cell, pool)
    if n_pos > len(pos):
        raise SystemExit(f"{cell}: asked for {n_pos:,} positives, {len(pos):,} exist")
    want_neg = negatives_for(n_pos, prevalence)
    if want_neg > len(neg):
        # Both ways out, because which one is acceptable is the caller's to judge.
        max_pos = max(1, int(len(neg) * prevalence / (1 - prevalence)))
        max_pi = n_pos / (n_pos + len(neg))
        raise SystemExit(
            f"{cell}: {n_pos:,} positives at pi={prevalence} needs {want_neg:,} negatives and "
            f"{len(neg):,} are available ({pool} pool) -- SHORT {want_neg - len(neg):,}. "
            f"Negatives are the scarce side and every extra positive costs "
            f"{round((1 - prevalence) / prevalence):,} more of them, so adding positives makes "
            f"this worse, not better. Either drop to {max_pos:,} positives at this pi, or raise "
            f"pi to {max_pi:.4g} at this positive count."
        )
    # Hash-ordered, so the draw is stable when the corpus changes and an export
    # at SCALE_N_POS reproduces what `designate_cells` chose (#3726).
    chosen_pos = sorted(sorted(pos, key=lambda i: rank(cell, i))[:n_pos])
    neg_key = "__negatives__" if pool == "shared" else f"{cell}:negatives"
    chosen_neg = sorted(sorted(neg, key=lambda i: rank(neg_key, i))[:want_neg])
    return {
        "cell": cell,
        "pool": pool,
        "prevalence": prevalence,
        "n_positives": len(chosen_pos),
        "n_negatives": len(chosen_neg),
        "available_positives": len(pos),
        "available_negatives": len(neg),
        "positives": chosen_pos,
        "negatives": chosen_neg,
    }


def add_test_bands(members: dict[int, dict], sel: dict, bands: "list[str] | None") -> dict:
    """Attach the class's other bands' positives to a single-cell selection.

    The selection keeps its own positives and negatives untouched; the sibling
    bands ride along under ``test_positives_by_band``, which is what lets one
    trained arm be scored at all three sizes. They are **not** merged into
    ``positives``: an image with a large car is not a positive of ``car@small``,
    and folding it in would move the very FNR the breakdown exists to keep
    separate.

    Every band's list is the whole band. Which slice of it a run tests on is the
    harness's business -- ``vtscore.eval.scale_bands`` takes the held-out
    fraction, so an off-diagonal reading uses the images the diagonal arm holds
    out and the 3x3 table pairs.
    """
    cls = sel["cell"].partition("@")[0]
    by_band, _ = class_candidates(members, cls, sel["pool"])
    own = band_of(sel["cell"])
    wanted = set(bands) if bands else None
    out = {b: ids for b, ids in sorted(by_band.items()) if b != own and (wanted is None or b in wanted)}
    ok, note = check_shared_negatives(members, cls, sel["pool"])
    if not ok:
        raise SystemExit(
            note + " -- cross-band testing reads three FNRs against ONE FPR, which needs one "
            "negative pool behind them; refusing to export a table whose columns are not paired"
        )
    return {
        **sel,
        "test_positives_by_band": out,
        "n_test_positives_by_band": {b: len(ids) for b, ids in out.items()},
        "natural_mix": {b: round(v, 6) for b, v in natural_mix(members, cls).items()},
        "shared_negatives_check": note,
    }


def select_mix(
    members: dict[int, dict],
    cls: str,
    mix: "str | dict[str, float]",
    n_pos: int,
    prevalence: float,
    pool: str = "per_class",
) -> dict:
    """One train-side size-mix selection: positives across bands at chosen shares.

    The cell is named ``<cls>@mix`` so it can never be mistaken for a pure band
    in a frame holding both. Each band's share is drawn by the same
    :func:`~pilebuild.scale_core.rank` order the pure cell uses, so the small
    cars in a mix are the *first* small cars -- two mixes differ in their
    proportions and not in their sample, and a mix's small slice is a subset of
    the pure ``@small`` cell rather than a second draw from the same supply.

    A band that cannot fill its share is reported by ``realised_mix`` differing
    from ``requested_mix``, never papered over by handing its seats to another
    band: a mix the corpus cannot serve must not look like one it can.
    """
    by_band, neg = class_candidates(members, cls, pool)
    if not by_band:
        raise SystemExit(f"{cls}: no banded positives")
    if mix == "natural":
        shares = natural_mix(members, cls)
    else:
        weights = {b: float(w) for b, w in mix.items() if float(w) > 0}
        unknown = sorted(set(weights) - set(by_band))
        if unknown:
            raise SystemExit(f"{cls}: no such band(s) {unknown}; have {sorted(by_band)}")
        total = sum(weights.values())
        if total <= 0:
            raise SystemExit(f"{cls}: mix {mix!r} has no positive weight")
        shares = {b: w / total for b, w in sorted(weights.items())}

    want = quota(shares, n_pos, {b: len(ids) for b, ids in by_band.items()})
    chosen_by_band = {
        b: sorted(sorted(by_band[b], key=lambda i: rank(f"{cls}@{b}", i))[: want.get(b, 0)])
        for b in sorted(want)
        if want.get(b, 0)
    }
    chosen_pos = sorted(i for ids in chosen_by_band.values() for i in ids)
    got = len(chosen_pos)
    want_neg = negatives_for(got, prevalence)
    if want_neg > len(neg):
        max_pos = max(1, int(len(neg) * prevalence / (1 - prevalence)))
        raise SystemExit(
            f"{cls}@mix: {got:,} positives at pi={prevalence} needs {want_neg:,} negatives and "
            f"{len(neg):,} are available ({pool} pool) -- SHORT {want_neg - len(neg):,}. "
            f"Either drop to {max_pos:,} positives at this pi, or raise pi to "
            f"{got / (got + len(neg)):.4g} at this positive count."
        )
    neg_key = "__negatives__" if pool == "shared" else f"{cls}:negatives"
    chosen_neg = sorted(sorted(neg, key=lambda i: rank(neg_key, i))[:want_neg])
    return {
        "cell": f"{cls}@mix",
        "class": cls,
        "pool": pool,
        "prevalence": prevalence,
        "requested_mix": {b: round(v, 6) for b, v in shares.items()},
        "realised_mix": {b: round(len(ids) / got, 6) for b, ids in sorted(chosen_by_band.items())} if got else {},
        "positives_by_band": {b: len(ids) for b, ids in sorted(chosen_by_band.items())},
        "available_by_band": {b: len(ids) for b, ids in sorted(by_band.items())},
        "n_positives": got,
        "n_positives_requested": n_pos,
        "n_negatives": len(chosen_neg),
        "available_positives": sum(len(v) for v in by_band.values()),
        "available_negatives": len(neg),
        "positives": chosen_pos,
        "positives_by_band_ids": chosen_by_band,
        "negatives": chosen_neg,
    }


def mix_census(members: dict[int, dict]) -> list[dict]:
    """Per class: the positives in each band and the shares they imply."""
    classes = sorted({c.partition("@")[0] for m in members.values() for c in m["cells"]})
    rows = []
    for cls in classes:
        by_band, _ = class_candidates(members, cls, "per_class")
        counts = {b: len(ids) for b, ids in sorted(by_band.items())}
        total = sum(counts.values())
        rows.append(
            {
                "class": cls,
                "counts": counts,
                "shares": {b: n / total for b, n in counts.items()} if total else {},
                "total": total,
            }
        )
    return rows


def envelope(members: dict[int, dict], pool: str) -> list[dict]:
    """Per cell: what is available, and the rarest pi reachable at 100 positives."""
    cells = sorted({c for m in members.values() for c in m["cells"]})
    rows = []
    for cell in cells:
        pos, neg = candidates(members, cell, pool)
        rows.append(
            {
                "cell": cell,
                "positives": len(pos),
                "negatives": len(neg),
                "min_pi_at_100": 100 / (100 + len(neg)) if neg else None,
                "min_pi_at_all_positives": len(pos) / (len(pos) + len(neg)) if neg else None,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help="e.g. bus@small")
    ap.add_argument("--positives", type=int, default=pc.SCALE_N_POS)
    ap.add_argument("--prevalence", type=float, default=pc.SCALE_PREVALENCE)
    ap.add_argument("--pool", default="per_class", choices=POOLS)
    ap.add_argument("--out", type=Path, help="write the manifest here (default: stdout summary only)")
    ap.add_argument("--envelope", action="store_true", help="what every cell can reach, then exit")
    ap.add_argument("--membership", type=Path, help="read membership from this cell instead")
    ap.add_argument(
        "--test-bands",
        help="also list the class's other bands' positives for cross-band testing: 'all', or e.g. 'medium,large'",
    )
    ap.add_argument(
        "--train-mix",
        help="draw positives across bands instead of from one: 'natural', or 'small=1,medium=2,large=1'. "
        "Takes a bare class in --cell",
    )
    ap.add_argument("--mix-census", action="store_true", help="per-class band shares, then exit")
    ap.add_argument("--check-bands", help="assert this class's bands share one negative pool, then exit")
    args = ap.parse_args()

    members = load_membership(args.membership)
    log(f"membership over {len(members):,} images")

    if args.envelope:
        rows = envelope(members, args.pool)
        print(f"\n{'cell':<20}{'pos':>8}{'neg':>10}{'min pi @100':>13}{'min pi @all pos':>17}")
        for r in rows:
            lo = f"{r['min_pi_at_100']:.5f}" if r["min_pi_at_100"] else "--"
            la = f"{r['min_pi_at_all_positives']:.5f}" if r["min_pi_at_all_positives"] else "--"
            print(f"{r['cell']:<20}{r['positives']:>8,}{r['negatives']:>10,}{lo:>13}{la:>17}")
        return 0

    if args.mix_census:
        print(f"\n{'class':<16}{'small':>9}{'medium':>9}{'large':>9}{'total':>9}   shares S/M/L")
        for r in mix_census(members):
            c = r["counts"]
            sh = r["shares"]
            got = "/".join(f"{sh.get(b, 0):.2f}" for b in ("small", "medium", "large"))
            print(
                f"{r['class']:<16}{c.get('small', 0):>9,}{c.get('medium', 0):>9,}"
                f"{c.get('large', 0):>9,}{r['total']:>9,}   {got}"
            )
        return 0

    if args.check_bands:
        ok, note = check_shared_negatives(members, args.check_bands, args.pool)
        log(note)
        return 0 if ok else 1

    if not args.cell:
        raise SystemExit("--cell is required unless --envelope, --mix-census or --check-bands")

    if args.train_mix:
        if "@" in args.cell:
            raise SystemExit(f"--train-mix takes a bare class in --cell, got {args.cell!r}")
        sel = select_mix(members, args.cell, parse_mix(args.train_mix), args.positives, args.prevalence, args.pool)
        log(
            f"{sel['cell']}: {sel['n_positives']:,} positives + {sel['n_negatives']:,} negatives "
            f"at pi={sel['prevalence']} ({sel['pool']} pool)"
        )
        log(f"  requested {sel['requested_mix']}  realised {sel['realised_mix']}  counts {sel['positives_by_band']}")
        if sel["n_positives"] < sel["n_positives_requested"]:
            log(
                f"  SHORT {sel['n_positives_requested'] - sel['n_positives']:,} positives: a band could not "
                f"fill its share (available {sel['available_by_band']})"
            )
        if args.out:
            args.out.write_text(json.dumps(sel, indent=1) + "\n")
            log(f"wrote {args.out}")
        return 0

    sel = select(members, args.cell, args.positives, args.prevalence, args.pool)
    if args.test_bands:
        bands = None if args.test_bands == "all" else [b.strip() for b in args.test_bands.split(",")]
        sel = add_test_bands(members, sel, bands)
    log(
        f"{sel['cell']}: {sel['n_positives']:,} positives + {sel['n_negatives']:,} negatives "
        f"at pi={sel['prevalence']} ({sel['pool']} pool; "
        f"{sel['available_positives']:,}/{sel['available_negatives']:,} available)"
    )
    if args.test_bands:
        log(f"  cross-band test positives: {sel['n_test_positives_by_band']}")
        log(f"  natural mix: {sel['natural_mix']}; {sel['shared_negatives_check']}")
    if args.out:
        args.out.write_text(json.dumps(sel, indent=1) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
