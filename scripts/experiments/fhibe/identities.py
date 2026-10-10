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

Usage (on a compute node; it loads four ~10k-media cells)::

    python identities.py --examples 4 --n 300 --out "$CALIB_EXP/identities.txt"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS), help="dataset:embedder cells to admit in")
    ap.add_argument("--examples", type=int, required=True, help="the LARGEST example count the study runs")
    ap.add_argument("--n", type=int, default=None, help="identities to draw (default: every one admitted)")
    ap.add_argument("--seed", type=int, default=4699)
    ap.add_argument("--sim-fraction", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.examples < 1:
        ap.error("--examples must be >= 1")

    pools: dict[str, set[str]] = {}
    seen: dict[str, int] = {}
    for spec in args.cells:
        ds, _, emb = spec.partition(":")
        if pc.DATASETS.get(ds, {}).get("kind") != "fhibe":
            ap.error(f"{ds} is not a FHIBE dataset")
        counts = positives_per_identity(load_medias(pc.EMBEDDINGS / f"{ds}__{emb}.pkl"))
        seen[spec] = len(counts)
        pools[spec] = eligible(counts, args.examples, args.sim_fraction)
        print(f"{spec}: {len(counts)} identities, {len(pools[spec])} admitted at {args.examples} example(s)")

    chosen, common = sample(pools, args.n, args.seed)
    union = set.union(*pools.values())
    header = [
        f"# FHIBE identities for #4699: {len(chosen)} drawn (seed {args.seed}) of {len(common)} that every cell admits",
        f"# admitted = >= {args.examples} positive(s) in the voting half and >= 1 withheld, stratified at "
        f"sim_fraction {args.sim_fraction}",
    ]
    for spec in args.cells:
        header.append(
            f"# {spec}: {seen[spec]} identities, {len(pools[spec])} admitted, "
            f"{len(union - pools[spec])} of the others' admitted excluded here"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + chosen) + "\n")
    print(f"wrote {args.out}: {len(chosen)} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
