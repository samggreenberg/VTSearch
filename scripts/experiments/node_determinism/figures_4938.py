"""#4938's figure: which BLAS ops crash or disagree across the GRID's CPUs, per numpy and L2 setting.

    python figures_4938.py --probes /expscratch/$USER/det-4938/{l2,l2-svd} --np-dir /expscratch/$USER/det-4930/np \
        --out docs/experiments/2026-10-15-openblas-zen4-4938/figures

* ``ops_by_numpy.png`` - rows: numpy version (its bundled OpenBLAS) under ``OPENBLAS_CORETYPE=Haswell``,
  without and with ``OPENBLAS_L2_SIZE=2048``; columns: the probed ops (``probes/openblas_l2.sh``).  A cell
  says whether every node gave the same result, some node's differed, or some node crashed.  The SVD and
  long-K run (``l2-svd``) covered two numpy versions on five nodes; its other cells read "not run".
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_4110 import VERDICT_COLORS  # noqa: E402

OPENBLAS = {"2.2.6": "0.3.29", "2.4.6": "0.3.31", "2.5.1": "0.3.33", "2.5.4": "0.3.34"}


def load(probes: list[Path], np_dir: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """(numpy, L2 setting, op) -> {node: hash or CRASH(..)}, as ``collate_openblas_l2.py`` reads it."""
    res: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for p in sorted(f for d in probes for f in d.glob("*.tsv")):
        for line in p.read_text().splitlines():
            node, np_v, env, op, m, k, digest = line.split("\t")
            np_v = np_v.split("=", 1)[1]
            info = list((np_dir / f"np-{np_v}").glob("numpy-*.dist-info"))
            if info:
                np_v = info[0].name.removeprefix("numpy-").removesuffix(".dist-info")
            res[(np_v, env.split("OPENBLAS_L2_SIZE=", 1)[1], f"{op}\nK={k}")][node] = digest
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, nargs="+", required=True)
    ap.add_argument("--np-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res = load(args.probes, args.np_dir)
    versions = sorted({k[0] for k in res}, key=lambda v: tuple(int(x) for x in v.split(".")))
    rows = [(v, l2) for v in versions for l2 in ("-", "2048")]
    ops = sorted({k[2] for k in res}, key=lambda o: (o[0] != "d", o.split("\n")[0], int(o.rsplit("=", 1)[1])))
    nodes = sorted({n for per in res.values() for n in per})

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(ops), 0.55 * len(rows) + 1.8))
    for i, (v, l2) in enumerate(rows):
        for j, op in enumerate(ops):
            per = res.get((v, l2, op), {})
            if not per:
                ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color="#e6e6e6", linewidth=0))
                ax.text(j + 0.47, i + 0.45, "not run", ha="center", va="center", fontsize=6, color="#555555")
                continue
            ok = {n: d for n, d in per.items() if not d.startswith("CRASH")}
            if len(ok) < len(per):
                color, text = VERDICT_COLORS["trajectory"], "crash"
            elif len(set(ok.values())) > 1:
                color, text = VERDICT_COLORS["objective"], "differ"
            else:
                color, text = VERDICT_COLORS["identical"], "same"
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color=color, linewidth=0))
            ax.text(j + 0.47, i + 0.45, text, ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(0, len(ops))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.47 for j in range(len(ops))], ops, fontsize=7)
    ax.xaxis.tick_top()
    labels = [
        f"numpy {v} (OpenBLAS {OPENBLAS.get(v, '?')}), " + ("no L2 pin" if l2 == "-" else "L2_SIZE=2048")
        for v, l2 in rows
    ]
    ax.set_yticks([i + 0.45 for i in range(len(rows))], labels, fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(
        f"OPENBLAS_CORETYPE=Haswell on up to {len(nodes)} nodes, 5 CPU models: does every node give the same result?",
        fontsize=10,
        pad=40,
    )
    fig.tight_layout()
    fig.savefig(args.out / "ops_by_numpy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
