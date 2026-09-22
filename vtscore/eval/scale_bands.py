"""Testing a scale-banded cell across sizes, and training it on a size mix (#4044).

A scale-banded dataset keys its cells ``class@band`` -- ``car@small``,
``car@medium``, ``car@large`` -- and
:func:`~vtscore.eval.labels.media_is_evaluable` excludes an image that holds the
class at *another* size from the cell entirely: it is neither a positive (the
object is the wrong size) nor a negative (there really is a car in it). That
exclusion is right, and it is also why a cell could only ever be tested at the
size it was trained at. ``car@medium`` was not merely unmeasured under the
``car@small`` arm; it was **unaddressable**, because those images are dropped
before the split.

This module adds the two things #4044 asks for, without moving any number the
harness already reports.

## Cross-band testing: one FPR, three FNRs

The three bands of one class **share a negative pool by construction** -- that is
the ``3 *`` in ``pile_config.SCALE_PREVALENCE``, and it is what makes the bands
a paired contrast. A negative holds no instance of the class, so it has no size
*for the class*, and there is exactly one FPR per arm. Breaking FPR down by size
is not a measurement anybody withheld; it has no referent. What decomposes is the
**FNR**, because a positive does have a size.

So the shape is one threshold, one FPR, and an FNR per band:
:func:`band_cohorts` builds the per-band held-out positive sets and the harness
scores each at the shipped cut.

**The cohorts are the ones the diagonal arm would have held out.** A sibling
band's cohort is taken by replaying the same split rule against that band's own
evaluable pool at the same seed, so the images in ``fnr_medium`` are the same
images whether the arm trained on ``small``, ``medium`` or ``large``. That is
what makes a 3x3 matrix a matrix rather than nine unrelated readings, and it is
why the cohort is the held-out fraction rather than the whole band -- testing the
off-diagonal on 100% of a band while the diagonal sees 20% of it would compare
two different populations. Precision is bought at the *export* layer instead, by
cutting more positives per cell (``quarry_export.py``), which costs nothing now
that the corpus is embedded whole.

## Train-side mixes: a retag, not a new code path

An object has some prevalence at each size in the wild, and the bands let that
proportion be chosen. Test-side that needs no new arm at all -- once the 3x3
matrix exists, the FNR of **any** test-side mix is the mix-weighted average of a
row of it, because an FNR over disjoint cohorts is linear in the mix. Train-side
it is not: training is not linear in its input, so each mix is a genuinely new
arm and has to be run.

:func:`project_mix` runs one by **retagging**, not by threading a second notion
of "the target" through the harness. It draws positives from each band at the
requested shares and rewrites their cell membership to a single synthetic cell
(``car@mix``), so every downstream consumer -- the pool filter, the prevalence
accounting, the vote order, the region boxes -- sees one ordinary cell and needs
no change. The per-band columns still resolve, because the band is read off the
media's original ``categories`` rather than off the target string.
"""

from __future__ import annotations

from typing import Any, Optional

#: What ``pile_config.scale_cell`` joins a class and a band with. A class name
#: never contains it -- `stop sign` has a space, not an ``@`` -- so the head of
#: one split is the class and the tail is the band.
BAND_SEPARATOR = "@"

#: The bands that get their own reported columns, in size order. Fixed rather
#: than discovered because the result frame's schema is fixed
#: (:data:`vtscore.eval.voting_columns.VOTING_COLUMNS`): a column per discovered
#: band would make two runs of the same study unconcatenable. A dataset banding
#: on some other vocabulary is not reported rather than silently mis-reported --
#: :func:`unreportable_bands` is what says so out loud.
REPORTED_BANDS: tuple[str, ...] = ("small", "medium", "large")

#: The band suffix :func:`project_mix` writes. Not one of
#: :data:`REPORTED_BANDS`, so a mixed arm's own cell can never be confused with
#: a pure one in a frame that holds both.
MIX_BAND = "mix"


def parse_cell(category: str) -> tuple[str, Optional[str]]:
    """``("car", "small")`` for ``"car@small"``; ``("dog", None)`` for ``"dog"``.

    The band is ``None`` for any category that is not band-suffixed, which is
    every non-scale dataset and the band-free sibling sets. Callers treat that
    as "there is no size axis here" rather than as an error: cross-band testing
    is simply off.
    """
    cls, sep, band = category.partition(BAND_SEPARATOR)
    return (cls, band) if sep else (category, None)


def cells_of_class(medias: dict[Any, dict[str, Any]], cls: str) -> list[str]:
    """Every ``cls@band`` cell that *medias* actually carries, sorted.

    Read off ``evaluable_categories`` rather than spelled from a band list,
    because that field is where a dataset declares its own cells -- the same
    argument that made ``scale_core._evaluable`` derive its map from the
    caller's cell list instead of writing ``scale_cell(c, band)`` inline.
    """
    prefix = cls + BAND_SEPARATOR
    out: set[str] = set()
    for media in medias.values():
        for key in ("evaluable_categories", "categories"):
            for cell in media.get(key) or ():
                if cell.startswith(prefix):
                    out.add(cell)
    return sorted(out)


def unreportable_bands(medias: dict[Any, dict[str, Any]], cls: str) -> list[str]:
    """Bands of *cls* present in the data that have no reported column.

    Returned so the harness can warn. A band measured into nothing is the
    failure this exists to prevent: it would look exactly like a band that was
    empty.
    """
    bands = [parse_cell(c)[1] for c in cells_of_class(medias, cls)]
    return sorted({b for b in bands if b and b not in REPORTED_BANDS and b != MIX_BAND})


def band_of(media: dict[str, Any], cls: str) -> Optional[str]:
    """The band at which *media* holds *cls*, or ``None`` if it holds none.

    Reads the media's **own** ``categories``, so it keeps answering after
    :func:`project_mix` has retagged the image into a synthetic cell -- which is
    the whole reason a mixed arm can still report per-band columns.
    """
    prefix = cls + BAND_SEPARATOR
    for cell in media.get("categories") or ():
        if cell.startswith(prefix):
            band = cell[len(prefix) :]
            if band != MIX_BAND:
                return band
    return None


def holdout_ids(pool_ids: list[int], sim_fraction: float, seed: int) -> list[int]:
    """The held-out half of *pool_ids*, by the harness's own split rule.

    A re-implementation would be free to drift from
    ``voting_iterations._split_media_ids``, and a drifted copy here would put
    training images into an off-diagonal test cohort without anything failing.
    So it is spelled to match, and
    ``test_the_own_band_cohort_is_the_harness_holdout`` pins the two together by
    running both.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(sorted(pool_ids)).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return [int(i) for i in shuffled[n_sim:]]


def band_cohorts(
    medias: dict[Any, dict[str, Any]],
    target_category: str,
    *,
    sim_fraction: float,
    seed: int,
    bands: Optional[list[str]] = None,
) -> dict[str, list[int]]:
    """``{band: held-out positive ids}`` for each band of the target's class.

    *medias* is the **unfiltered** pool -- the cross-band positives are exactly
    the images ``evaluable_pool`` removes, so a cohort cannot be built from what
    it returns.

    Each band's cohort is the held-out fraction of *that band's own* evaluable
    pool at *seed*, which is the set the arm trained on that band would have
    tested against. Negatives are deliberately absent: they are shared across
    the class's bands, so they belong to the run's single FPR rather than to any
    one cohort.

    *bands* restricts the answer; ``None`` takes every band present.
    """
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415

    cls, band = parse_cell(target_category)
    if band is None:
        return {}
    wanted = set(bands) if bands is not None else None
    out: dict[str, list[int]] = {}
    for cell in cells_of_class(medias, cls):
        this_band = parse_cell(cell)[1]
        if this_band is None or this_band == MIX_BAND:
            continue
        if wanted is not None and this_band not in wanted:
            continue
        pool = evaluable_pool(medias, cell)
        held = holdout_ids(list(pool), sim_fraction, seed)
        positives = [cid for cid in held if media_is_positive(pool[cid], cell)]
        if positives:
            out[this_band] = positives
    return out


def natural_mix(medias: dict[Any, dict[str, Any]], cls: str) -> dict[str, float]:
    """The share of *cls*'s positives sitting in each band, as the data has it.

    This is the "proportion in the wild" a mix can be set *against*: on an
    exhaustively annotated corpus it is a measurement, not a design choice, so a
    ``natural`` mix arm is the one that asks the question the app's user
    actually faces. Returns ``{}`` when the class has no banded positives.
    """
    counts: dict[str, int] = {}
    for media in medias.values():
        band = band_of(media, cls)
        if band is not None:
            counts[band] = counts.get(band, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {band: n / total for band, n in sorted(counts.items())}


def resolve_mix(medias: dict[Any, dict[str, Any]], cls: str, mix: "str | dict[str, float]") -> dict[str, float]:
    """Normalise a mix spec to shares summing to 1.

    ``"natural"`` reads :func:`natural_mix`; a dict of per-band weights is
    normalised as given, so ``{"small": 1, "medium": 1, "large": 2}`` and
    ``{"small": 0.25, ...}`` are the same request.
    """
    if isinstance(mix, str):
        if mix != "natural":
            raise ValueError(f"unknown mix spec {mix!r}; pass 'natural' or a per-band mapping")
        shares = natural_mix(medias, cls)
        if not shares:
            raise ValueError(f"{cls!r} has no banded positives in this pool, so it has no natural mix")
        return shares
    weights = {band: float(w) for band, w in mix.items() if float(w) > 0}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"mix {mix!r} has no positive weight")
    return {band: w / total for band, w in sorted(weights.items())}


def _quota(shares: dict[str, float], n_positives: int, available: dict[str, int]) -> dict[str, int]:
    """How many positives each band contributes, largest-remainder, capped by supply.

    Largest-remainder rather than rounding each share independently, so the
    quotas sum to *n_positives* exactly instead of to whatever the rounding
    happened to give -- a mixed arm whose realised size drifts with the
    *rounding* is not comparable with the pure arms it is measured against.

    Supply is the other story: see the body.
    """
    exact = {b: shares[b] * n_positives for b in shares}
    out = {b: int(v) for b, v in exact.items()}
    # Largest-remainder first, so rounding alone never costs the set a seat.
    order = sorted(shares, key=lambda b: (-(exact[b] - int(exact[b])), b))
    for b in order[: n_positives - sum(out.values())]:
        out[b] += 1
    # Then cap by supply, and DELIBERATELY do not backfill the deficit into the
    # bands that do have headroom. Handing a short band's seats to its
    # neighbours keeps the count round at the price of quietly running a
    # different mix from the one that was asked for -- which is the
    # wrong-but-plausible arm, not a smaller one. A mix the corpus cannot serve
    # comes out short and says so.
    return {b: min(n, available.get(b, 0)) for b, n in out.items()}


def project_mix(
    medias: dict[Any, dict[str, Any]],
    cls: str,
    mix: "str | dict[str, float]",
    *,
    n_positives: int,
    label: str = MIX_BAND,
) -> tuple[dict[Any, dict[str, Any]], dict[str, Any]]:
    """A pool whose positives are *cls* at the requested size mix, plus a report.

    The draw takes no seed, for the same reason a pure cell's membership takes
    none: which images a cell *contains* is a property of the cell, and the seed
    varies the split and the vote order on top of it. Two mixes therefore differ
    only in their proportions, never in which small cars they happened to draw.

    Returns ``(medias, report)``. The returned pool is a new dict of shallow
    copies -- the pixel bytes and the vectors are shared by reference, so this
    costs a dict per media and no data.

    Three rewrites, and each is needed for the retag to be invisible downstream:

    * The chosen positives get ``cls@<label>`` as their cell, in ``categories``
      and ``evaluable_categories`` alike, so they are positives of it.
    * Every image that could be scored against *any* of the class's bands but
      holds none of them -- the class's shared negative pool -- gains
      ``cls@<label>`` in ``evaluable_categories`` only, so it stays the negative
      it already was.
    * Their ``regions`` are relabelled too. ``region_box_for_category`` matches
      a region's ``label`` against the target string, so a region-voting arm on
      an unrelabelled mix would find no box on any positive and silently fall
      back to whole-image votes -- a run that looks like region voting and is
      not, which is exactly #2877's failure.

    **A positive the mix did not draw is excluded, not demoted.** It holds the
    class at a size this population does not contain, so it is neither a
    positive nor a negative here -- the same rule the bands already apply to
    each other, applied to the quota.
    """
    from vtscore.eval.labels import media_is_evaluable  # noqa: PLC0415

    shares = resolve_mix(medias, cls, mix)
    cells = cells_of_class(medias, cls)
    cell_of_band = {parse_cell(c)[1]: c for c in cells}
    by_band: dict[str, list[Any]] = {band: [] for band in shares}
    negatives: list[Any] = []
    for cid, media in medias.items():
        band = band_of(media, cls)
        if band is None:
            # A negative for the class is one that can be scored against its
            # cells and holds none of them; anything else is simply not in this
            # population (another class's positive on an unannotated source).
            if any(media_is_evaluable(media, cell) for cell in cells):
                negatives.append(cid)
        elif band in by_band:
            by_band[band].append(cid)

    quota = _quota(shares, n_positives, {band: len(ids) for band, ids in by_band.items()})
    mixed_cell = f"{cls}{BAND_SEPARATOR}{label}"
    chosen: dict[str, list[Any]] = {}
    for band, ids in by_band.items():
        # Ranked by the band's own cell rather than by the mixed one, so the
        # first N of `car@small` are the same images whichever mix drew them --
        # two mixes then differ only in their proportions, not in their sample.
        ranked = sorted(ids, key=lambda i: _rank(cell_of_band[band], i))
        chosen[band] = ranked[: quota.get(band, 0)]

    out: dict[Any, dict[str, Any]] = {}
    for band, ids in chosen.items():
        for cid in ids:
            media = dict(medias[cid])
            own = cell_of_band[band]
            media["categories"] = sorted({*(media.get("categories") or ()), mixed_cell})
            media["evaluable_categories"] = sorted({*(media.get("evaluable_categories") or ()), mixed_cell})
            regions = media.get("regions")
            if regions:
                media["regions"] = [({**r, "label": mixed_cell} if r.get("label") == own else r) for r in regions]
            out[cid] = media
    for cid in negatives:
        media = dict(medias[cid])
        media["evaluable_categories"] = sorted({*(media.get("evaluable_categories") or ()), mixed_cell})
        out[cid] = media

    realised = {band: len(ids) for band, ids in sorted(chosen.items()) if ids}
    total = sum(realised.values())
    report = {
        "cell": mixed_cell,
        "class": cls,
        "requested_mix": {band: round(share, 6) for band, share in shares.items()},
        "realised_mix": {band: round(n / total, 6) for band, n in realised.items()} if total else {},
        "positives_by_band": realised,
        "n_positives": total,
        "n_positives_requested": n_positives,
        "n_negatives": len(negatives),
        "available_by_band": {band: len(ids) for band, ids in sorted(by_band.items())},
    }
    return out, report


def _rank(cell: str, iid: Any) -> str:
    """The pile's own cell-local ordering key, so a draw here matches a draw there.

    ``pilebuild.scale_core.rank`` is experiment-tier and cannot be imported from
    the library, so the rule is restated -- and it is restated rather than
    replaced by ``rng.sample`` for the reason that function documents: a hash of
    ``(cell, id)`` does not reshuffle the whole draw when one image enters or
    leaves the pool.
    """
    import hashlib  # noqa: PLC0415

    return hashlib.sha1(f"{cell}:{iid}".encode()).hexdigest()  # noqa: S324 - not security
