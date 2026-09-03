"""Does a drawn box earn its keep? The binary arm against its boxed twin.

Every other table in the overview benchmark confounds two axes: which encoder,
and whether the user drew a box. The binary run fixes that by re-running the
patch encoder's own cells with region voting off — same datasets, categories,
seeds and splits, differing only in the vector a Good vote contributes. That
makes the interaction axis measurable on its own, and it is the only comparison
in the study that answers "should I ask my users to draw?".

Because the two arms are separate runs, the contrast has to be paired *across*
result dirs, which `analyze_bench.py` (one dir at a time) cannot do.

    python analyze_bench_interaction.py <binary_results> <boxed_results> [out.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from _cells_io import describe_load
from bench_cells import load_cells, paired_contrasts

BINARY_SUFFIX = " (binary)"
METRICS = ("cost", "fpr", "fnr", "average_precision", "auroc")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    binary_dir, boxed_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    pd.set_option("display.width", 220)
    binary, prov_b = load_cells(binary_dir, embedder_suffix=BINARY_SUFFIX)
    boxed, prov_x = load_cells(boxed_dir)
    if binary.empty or boxed.empty:
        print("one side has no cells; nothing to pair")
        return 1

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 110)
    emit("BINARY vs BOXES - the interaction axis, isolated")
    emit("=" * 110)
    for prov in (prov_b, prov_x):
        emit(f"{prov['results']}: {describe_load(prov)}")
    starved_b = {n for n in prov_b["header_only"]}
    emit(f"starved cells in the binary arm: {sorted(starved_b)}")

    both = pd.concat([binary, boxed[boxed["dataset"].isin(set(binary["dataset"]))]], ignore_index=True)
    deep = both[both["t"] >= 100]

    emit()
    emit("DEEP-REGIME MEANS (t >= 100)")
    emit(deep.groupby(["dataset", "embedder"])[list(METRICS)].mean().round(3).to_string())

    binary_arm = f"dinov3_patch{BINARY_SUFFIX}"
    others = [e for e in sorted(set(deep["embedder"])) if e != binary_arm]
    emit()
    emit("PAIRED CONTRASTS - same (category, seed); mean difference +- standard error")
    emit("positive cost / negative AP = the binary arm is WORSE")
    emit(
        paired_contrasts(deep, contrasts=[(binary_arm, other) for other in others], metrics=METRICS).to_string(
            index=False
        )
    )
    emit()
    emit("A difference smaller than twice its standard error is not resolvable by this sample.")

    if out:
        out.write_text("\n".join(lines) + "\n")
        print(f"\n(written to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
