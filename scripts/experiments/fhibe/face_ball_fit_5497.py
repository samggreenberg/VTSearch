"""#5497: the face ball fit's go/no-go check, broken down by crop size and K, on FHIBE's three practice spaces.

``scripts/data/fit_face_ball.py`` writes one entry per space; its check is one number per balance. This reads
the same cells (:func:`vtscore.media.face.face_ball_fit.build_cells`) and the same cross-fitted floors, and
writes what the report shows: the gain against the pre-ball line per space, size and K, and the figure.

    python face_ball_fit_5497.py table --space bs|bl|fn --out <dir>     # <dir>/bycell_<space>.csv (one row per cell)
    python face_ball_fit_5497.py figure --tables <dir> --out check.webp

The spaces are #5486's: buffalo_s (the shipped pack), buffalo_l and FaceNet (the plain arm before #5320). The
cells name FHIBE people: the CSVs are owner-only, the figure is aggregate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))

EMB = Path("/expscratch/sgreenberg/vts-cache/datadir/embeddings")
PEOPLE = Path("/expscratch/sgreenberg/fhibe/fhibe-full-resolution-674a7dcf-derived/runs/2026-10-09-k4/identities.txt")
SPACES = {
    "bs": ("buffalo_s", ["fhibe_faces_ik_1024__face", "fhibe_faces_ik_640__face"]),
    "bl": ("buffalo_l", ["fhibe_faces_ik_1024__imagekit_face", "fhibe_faces_ik_640__imagekit_face"]),
    "fn": ("FaceNet", ["fhibe_faces_1024__face", "fhibe_faces_640__face"]),
}


def table(space: str, out: Path) -> None:
    import pandas as pd  # noqa: PLC0415
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.detectors.centroid_head import FACE_BALL_BELOW, centroid_cut  # noqa: PLC0415
    from vtscore.media.face import face_ball_fit as F  # noqa: PLC0415

    people = [s for line in PEOPLE.read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]
    corpora = {m: load_medias(EMB / f"{m}.pkl", repair=True) for m in SPACES[space][1]}
    cells = F.build_cells(corpora, people)
    rows = []
    for beta in F.FIT_BETAS:
        fits = {f: F.fit_floor([c for c in cells if F.fold_of(c.person) != f], beta) for f in (0, 1)}
        for c in cells:
            floor, slope, centre = fits[F.fold_of(c.person)]
            ball = F._f_at(c, floor + slope * (c.spread - centre), beta)
            line = F._f_at(c, centroid_cut(c.scores, rule=FACE_BALL_BELOW, beta=beta), beta)
            size = "1024" if "1024" in c.corpus else "640"
            rows.append(
                {
                    "space": space,
                    "size": size,
                    "k": c.k,
                    "person": c.person,
                    "beta": beta,
                    "ball_f": ball,
                    "line_f": line,
                    "gain": ball - line,
                    "separable": c.separable,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"bycell_{space}.csv", index=False)
    print(pd.DataFrame(rows).groupby(["beta", "size", "k"])["gain"].mean().round(4).to_string())


def figure(tables: Path, out: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from vtscore.media.face.face_ball_fit import LEVEL  # noqa: PLC0415

    colors = {"bs": "#2a78d6", "bl": "#eb6834", "fn": "#1baf7a"}
    markers = {"bs": "o", "bl": "s", "fn": "D"}
    df = pd.concat([pd.read_csv(p) for p in sorted(tables.glob("bycell_*.csv"))])
    df["kt"] = df["k"].map(lambda k: "all" if k == 0 else str(k))
    order = ["2", "3", "4", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, beta in zip(axes, (1.0, 4.0), strict=True):
        sub = df[df["beta"] == beta]
        for j, (space, (name, _)) in enumerate(SPACES.items()):
            for s, size in enumerate(("1024", "640")):
                g = sub[(sub["space"] == space) & (sub["size"].astype(str) == size)].groupby("kt")["gain"]
                m, se = g.mean(), g.sem()
                x = np.arange(len(order)) + (j - 1) * 0.22 + (s - 0.5) * 0.08
                ax.errorbar(
                    x,
                    [m[k] for k in order],
                    yerr=[1.96 * se[k] for k in order],
                    fmt=markers[space],
                    ms=6 if size == "1024" else 4.5,
                    color=colors[space],
                    mfc=colors[space] if size == "1024" else "white",
                    lw=1.2,
                    capsize=0,
                    label=f"{name}" if size == "1024" else None,
                )
            y = sub[sub["space"] == space].groupby("kt")["gain"].mean()["all"]
            ax.annotate(
                name, (len(order) - 1 + (j - 1) * 0.22 + 0.15, y), fontsize=8, color="#333333", va="center", ha="left"
            )
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axhline(-LEVEL, color="#999999", lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(range(len(order)), [f"K={k}" for k in order])
        ax.set_title(f"β {beta:g}", fontsize=10)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(-0.5, len(order) - 0.1)
    axes[0].set_ylabel("ball F − pre-ball line F\n(cross-fitted, mean ± 95% CI)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.01,
        0.01,
        "filled: 1024 px crops, hollow: 640 px. Dashed: the go rule's level line (−0.01).",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--space", choices=sorted(SPACES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("figure")
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "table":
        table(args.space, args.out)
    else:
        figure(args.tables, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
