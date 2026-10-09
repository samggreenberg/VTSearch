#!/usr/bin/env python
"""Sample the FHIBE identities a study runs, as a ``CALIB_CATEGORY_FILE`` (#4699).

A FHIBE cell is one identity, and there are ~1,735 of them with two or more one-person photos. All
of them is ~7k cells per arm. This draws a seeded sample of the identities that every cell in the
study can run, and writes one subject id per line.

"Can run" is fixed, not drawn. Under the stratified split (``CALIB_STRATIFY_TARGET=1``) an identity
with *n* positives in a cell keeps ``ceil(n/2)`` of them in the voting half
(:func:`vtscore.eval.example_opening.positive_split_sizes`), and the example opening needs
``--examples`` of those. An identity has to clear that in EVERY cell named, so the photo and crop
arms run the same people and pair. The face cells hold fewer positives than the photo cells, because
MTCNN misses some faces, so the crop arms are what exclude people. The header says how many each
cell excluded, because that exclusion is conditioned on the face being found.

Pass the LARGEST example count the study will run. A smaller count then runs the same identities, so
arms that differ only in the count pair.

``--strata`` draws ``--n`` from each band of photo counts instead (#4731), counted in
``--strata-cell`` (the whole-photo cell: one positive per one-person photo), and writes each drawn
identity's count and band beside the file as ``<out>.strata.tsv``.

Usage (on a compute node; it loads four ~10k-media cells)::

    python identities.py --examples 4 --n 300 --out "$CALIB_EXP/identities.txt"
    python identities.py --examples 1 --n 100 --strata 2-4,5-6,7- --out "$CALIB_EXP/identities.txt"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "calibration"), str(_HERE.parent / "pile"), str(_HERE.parents[2])]

import pile_config as pc  # noqa: E402
from _cells_io import load_medias  # noqa: E402

from vtscore.eval.example_opening import positive_split_sizes  # noqa: E402

#: The study's cells, ``dataset:embedder`` (the embedder whose pickle the cell loads).
DEFAULT_CELLS = ("fhibe_1024:siglip", "fhibe_640:siglip", "fhibe_faces_1024:face", "fhibe_faces_640:face")


def positives_per_identity(medias: dict) -> Counter:
    """``{subject id: positives}`` in one cell. A face no subject owns carries no category."""
    return Counter(m["category"] for m in medias.values() if m.get("category"))


def eligible(counts: Counter, examples: int, sim_fraction: float) -> set[str]:
    """Identities whose voting half holds *examples* positives and whose withheld half holds one."""
    out = set()
    for cat, n in counts.items():
        n_sim, n_test = positive_split_sizes(n, sim_fraction)
        if n_test >= 1 and n_sim >= examples:
            out.add(cat)
    return out


def sample(pools: dict[str, set[str]], n: int | None, seed: int) -> tuple[list[str], set[str]]:
    """``(chosen, every identity all cells admit)``: a seeded sample of the intersection, sorted."""
    common = set.intersection(*pools.values()) if pools else set()
    ordered = sorted(common)
    if n is None or n >= len(ordered):
        return ordered, common
    picked = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return sorted(ordered[int(i)] for i in picked), common


def parse_strata(spec: str) -> list[tuple[str, int, float]]:
    """``"2-4,5-6,7-"`` as ``[(label, lo, hi), ...]``, inclusive, ``hi`` infinite on an open band."""
    out = []
    for band in spec.split(","):
        lo_s, sep, hi_s = band.strip().partition("-")
        if not sep or not lo_s.isdigit() or (hi_s and not hi_s.isdigit()):
            raise ValueError(f"stratum {band!r} is not <lo>-<hi> or <lo>-")
        lo, hi = int(lo_s), (int(hi_s) if hi_s else float("inf"))
        if hi < lo:
            raise ValueError(f"stratum {band!r} ends before it starts")
        out.append((band.strip(), lo, hi))
    for (a, _, a_hi), (b, b_lo, _) in zip(out, out[1:]):
        if b_lo <= a_hi:
            raise ValueError(f"strata {a!r} and {b!r} overlap or are out of order")
    return out


def sample_strata(
    pools: dict[str, set[str]], counts: Counter, strata: list[tuple[str, int, float]], n: int | None, seed: int
) -> tuple[list[str], dict[str, str], set[str]]:
    """``(chosen, {id: band}, admitted)``: up to *n* from each band of *counts*, all seeded off *seed*.

    Each band draws from its own stream, so adding a band never changes another band's draw.
    """
    common = set.intersection(*pools.values()) if pools else set()
    chosen: list[str] = []
    band_of: dict[str, str] = {}
    for k, (label, lo, hi) in enumerate(strata):
        members = {c for c in common if lo <= counts.get(c, 0) <= hi}
        picked, _ = sample({label: members}, n, seed * 1000 + k)
        chosen += picked
        band_of.update({c: label for c in picked})
    return sorted(chosen), band_of, common


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS), help="dataset:embedder cells to admit in")
    ap.add_argument("--examples", type=int, required=True, help="the LARGEST example count the study runs")
    ap.add_argument("--n", type=int, default=None, help="identities to draw (default: every one admitted)")
    ap.add_argument("--seed", type=int, default=4699)
    ap.add_argument("--sim-fraction", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strata", default=None, help="bands of photo counts, e.g. 2-4,5-6,7-; --n is then per band")
    ap.add_argument(
        "--strata-cell", default=None, help="dataset:embedder the photo counts come from (default: the first cell)"
    )
    args = ap.parse_args(argv)
    if args.examples < 1:
        ap.error("--examples must be >= 1")
    try:
        strata = parse_strata(args.strata) if args.strata else None
    except ValueError as exc:
        ap.error(str(exc))
    strata_cell = args.strata_cell or args.cells[0]
    if strata and strata_cell not in args.cells:
        ap.error(f"--strata-cell {strata_cell} is not one of --cells")

    pools: dict[str, set[str]] = {}
    seen: dict[str, int] = {}
    photo_counts: Counter = Counter()
    for spec in args.cells:
        ds, _, emb = spec.partition(":")
        if pc.DATASETS.get(ds, {}).get("kind") != "fhibe":
            ap.error(f"{ds} is not a FHIBE dataset")
        counts = positives_per_identity(load_medias(pc.EMBEDDINGS / f"{ds}__{emb}.pkl"))
        seen[spec] = len(counts)
        pools[spec] = eligible(counts, args.examples, args.sim_fraction)
        if spec == strata_cell:
            photo_counts = counts
        print(f"{spec}: {len(counts)} identities, {len(pools[spec])} admitted at {args.examples} example(s)")

    band_of: dict[str, str] = {}
    if strata:
        chosen, band_of, common = sample_strata(pools, photo_counts, strata, args.n, args.seed)
    else:
        chosen, common = sample(pools, args.n, args.seed)
    union = set.union(*pools.values())
    header = [
        f"# FHIBE identities for #4699: {len(chosen)} drawn (seed {args.seed}) of {len(common)} that every cell admits",
        f"# admitted = >= {args.examples} positive(s) in the voting half and >= 1 withheld, stratified at "
        f"sim_fraction {args.sim_fraction}",
    ]
    if strata:
        drawn = Counter(band_of.values())
        header.append(
            f"# strata by photo count in {strata_cell}, up to {args.n} each: "
            + ", ".join(
                f"{label}: {drawn[label]} of {sum(1 for c in common if lo <= photo_counts.get(c, 0) <= hi)}"
                for label, lo, hi in strata
            )
        )
    for spec in args.cells:
        header.append(
            f"# {spec}: {seen[spec]} identities, {len(pools[spec])} admitted, "
            f"{len(union - pools[spec])} of the others' admitted excluded here"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + chosen) + "\n")
    if strata:
        side = args.out.with_name(args.out.name + ".strata.tsv")
        side.write_text(
            "identity\tphotos\tstratum\n" + "".join(f"{c}\t{photo_counts[c]}\t{band_of[c]}\n" for c in chosen)
        )
        print(f"wrote {side}")
    print(f"wrote {args.out}: {len(chosen)} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
