"""#4930's figures: which numpy pin works on which numpy, and the cells on numpy 2.5 end to end.

    python figures_4930.py --probes /expscratch/$USER/det-4930/probes \
        --analysis /expscratch/$USER/det-4930/cells/analysis \
        --out docs/experiments/2026-10-15-numpy-pin-4930/figures

* ``pins_by_numpy.png`` - per numpy version and ``NPY_DISABLE_CPU_FEATURES`` value
  (``probe_numpy_pins.py``, 7 nodes, 5 CPU models): how many of the probed ops still
  differ between nodes, and whether numpy warned that it skipped a name.  The float32
  GEMM's numpy 2.5 drift on the EPYC 9534 is OpenBLAS's (#4938), not numpy's, so it is
  left out of the count.
* ``verdicts.png`` - per pair of arms, the share of cells in each verdict of
  ``analyze_4110.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_4110 import VERDICT_COLORS, VERDICT_LABELS  # noqa: E402

PINS = [
    ("base", "unset"),
    ("old", "AVX512F\nAVX512_SKX"),
    ("old_icl", "AVX512F AVX512_SKX\nAVX512_ICL"),
    ("old_all", "every AVX512*\nname"),
    ("v4", "X86_V4"),
    ("v4_icl_spr", "X86_V4\nAVX512_ICL AVX512_SPR"),
]
#: Left out of the count: numpy's OpenBLAS, which OPENBLAS_CORETYPE pins (#4481), not this variable.
BLAS_OPS = {"matmul_f32"}
PAIRS = [
    ("i_ship", "a_ship", "version-chosen pin\nIntel vs AMD"),
    ("i_full", "a_full", "pre-#4930 pin\nIntel vs AMD"),
    ("a_ship", "a_full", "old vs new pin\nAMD (no AVX-512)"),
    ("i_ship", "i_full", "old vs new pin\nIntel (AVX-512)"),
]


def pins_by_numpy(probes: Path, out: Path) -> None:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in probes.glob("*__*__*.json"):
        r = json.loads(p.read_text())
        by_key[(r["np"], r["tag"].split("__", 1)[1])].append(r)
    versions = sorted({k[0] for k in by_key}, key=lambda v: tuple(int(x) for x in v.split(".")))
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(versions) + 1.6))
    for i, ver in enumerate(versions):
        for j, (tag, _) in enumerate(PINS):
            rs = by_key.get((ver, tag), [])
            if not rs:
                continue
            ops = [op for op in rs[0]["hash"] if op not in BLAS_OPS]
            differ = sum(len({r["hash"][op] for r in rs}) > 1 for op in ops)
            text = f"{differ} ops differ" if differ else "all agree"
            if any(r["warnings"] for r in rs):
                text += "\n(warns)"
            if tag == "base":
                color = "#d9d9d9"
            else:
                color = VERDICT_COLORS["trajectory"] if differ else VERDICT_COLORS["identical"]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=color, linewidth=0))
            ink = "white" if color in (VERDICT_COLORS["trajectory"], VERDICT_COLORS["identical"]) else "black"
            ax.text(j + 0.48, i + 0.46, text, ha="center", va="center", fontsize=8.5, color=ink)
    ax.set_xlim(0, len(PINS))
    ax.set_ylim(len(versions), 0)
    ax.set_xticks([j + 0.48 for j in range(len(PINS))], [label for _, label in PINS], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.46 for i in range(len(versions))], [f"numpy {v}" for v in versions])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    n_nodes = len({r["node"] for rs in by_key.values() for r in rs})
    ax.set_title(
        f"NPY_DISABLE_CPU_FEATURES by numpy version: {n_nodes} nodes, 5 CPU models, "
        f"{len(ops)} numpy ops\n(warns) = numpy skipped a name its build does not dispatch",
        fontsize=10,
        pad=34,
    )
    fig.tight_layout()
    fig.savefig(out / "pins_by_numpy.png", dpi=130)
    plt.close(fig)


def verdicts(analysis: Path, out: Path) -> None:
    summary = {p["pair"]: p for p in json.loads((analysis / "summary.json").read_text())["pairs"]}
    keys = ["identical", "diagnostics", "threshold", "objective", "trajectory"]
    rows = [(f"{a} ~ {b}", label) for a, b, label in PAIRS if f"{a} ~ {b}" in summary]
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(rows) + 1.4))
    for i, (pair, _) in enumerate(rows):
        s = summary[pair]
        left = 0.0
        for k, name in zip(keys, VERDICT_LABELS, strict=True):
            share = s[k] / s["cells"] if s["cells"] else 0.0
            if share:
                ax.barh(i, share, left=left, color=VERDICT_COLORS[k], edgecolor="white", linewidth=2, height=0.6)
                if share > 0.08:
                    ink = "black" if k in ("diagnostics", "threshold") else "white"
                    ax.text(left + share / 2, i, f"{s[k]}", ha="center", va="center", fontsize=9, color=ink)
                left += share
        ax.text(1.01, i, f"{s['identical']}/{s['cells']} identical", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of cells")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[k]) for k in keys]
    ax.legend(handles, VERDICT_LABELS, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("72 calibration cells a pair on numpy 2.5.4: Xeon Silver 4114 (Intel), EPYC 7713 (AMD)", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "verdicts.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pins_by_numpy(args.probes, args.out)
    if args.analysis:
        verdicts(args.analysis, args.out)


if __name__ == "__main__":
    main()
