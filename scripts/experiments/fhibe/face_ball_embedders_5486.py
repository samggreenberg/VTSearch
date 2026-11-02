"""#5486: #5478's face ball on two hypothetical face embedders, beside buffalo_s.

The ball's floors were fitted on FHIBE's buffalo_s crops (#5474, #5478). This asks whether a cosine floor
travels with the idea or with the embedder: the same shadow replay (``face_ball_line_5478.py replay``) on
sessions run with FaceNet's vectors (MTCNN crops, the face arm before #5320) and with buffalo_l's (w600k_r50),
each priced twice, at buffalo_s's shipped floors (``transfer``) and at the embedder's own, refitted on its
#5474 dumps and cross-fitted by person fold (``refit``). Neither embedder is a proposal for the app.

Subcommands::

    table    the shipped form (the ball from beta 1 up with two Goods, else Find's line) minus Find's line,
             paired by person, per embedder x floors x size x K x beta: votes 25 and 150 and the mean over 1-150
    figure   mean F-beta per vote at 4 photos, 1024 px, beta 1 and 4: Find's line, the ball at buffalo_s's
             floors and at the embedder's own, one panel per embedder

Usage (any environment with pandas and matplotlib; each ``--arm`` is name=analysis_dir:replay_dir[:column])::

    python face_ball_embedders_5486.py table --arm "buffalo_s=<a>:<r>" --arm "FaceNet transfer=<a>:<r>" ... --out t.csv
    python face_ball_embedders_5486.py figure --arm ... --out embedders.webp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
READS = ("vote 25", "vote 150", "mean 1-150")


def parse_arm(spec: str) -> tuple[str, Path, Path, str]:
    """``name=analysis:replay[:column]``; the column is the replay's shipped form, ``gated`` (pooled floors) by default."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    return name, Path(parts[0]), Path(parts[1]), parts[2] if len(parts) > 2 else "gated"


def table(arms: list[str], out: Path) -> None:
    import pandas as pd  # noqa: PLC0415

    rows = []
    for spec in arms:
        name, analysis, _replay, col = parse_arm(spec)
        p = pd.read_csv(analysis / "paired.csv")
        p = p[(p["arm"] == col) & p["read"].isin(READS) & (p["beta"] >= 1)]
        rows.append(p.assign(embedder=name))
    t = pd.concat(rows, ignore_index=True)
    t = t[["embedder", "size", "K", "beta", "read", "people", "app_f", "ball_f", "diff", "lo", "hi", "wins", "loses"]]
    t.round(3).to_csv(out, index=False)
    for rd in READS:
        x = t[t["read"] == rd].assign(
            cell=lambda d: d.apply(lambda r: f"{r['diff']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1)
        )
        print(f"\n{rd}: ball - Find's line (paired 95% CI)")
        print(
            x.pivot_table(index=["size", "K", "beta"], columns="embedder", values="cell", aggfunc="first").to_string()
        )


def figure(arms: list[str], out: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    from face_ball_line_5478 import load_replay  # noqa: PLC0415

    # One panel per embedder; within it, one replay per set of floors.
    panels: dict[str, list[tuple[str, Path, str]]] = {}
    for spec in arms:
        name, _analysis, replay, col = parse_arm(spec)
        emb, _, floors = name.partition(" ")
        panels.setdefault(emb, []).append((floors or "own", replay, col))
    fig, ax = plt.subplots(2, len(panels), figsize=(5 * len(panels), 8), squeeze=False)
    style = {"own": ("#1b2f52", "-"), "refit": ("#1b2f52", "-"), "transfer": ("#c0392b", "--")}
    for j, (emb, reps) in enumerate(panels.items()):
        for i, beta in enumerate((1.0, 4.0)):
            a = ax[i, j]
            drew_line = False
            for floors, replay, col in reps:
                df = load_replay(replay)
                h = df[(df["t"] <= 150) & (df["size"] == "1024") & (df["K"] == 4) & (df["beta"] == beta)]
                m = h.groupby("t")[["app_f", f"{col}_f"]].mean()
                if not drew_line:
                    a.plot(m.index, m["app_f"], color="#7f8c8d", lw=1.4, ls=":", label="Find's line (before #5478)")
                    drew_line = True
                c, ls = style.get(floors, ("#1b2f52", "-"))
                label = "ball, buffalo_s's floors" if floors == "transfer" else "ball, its own floors"
                a.plot(m.index, m[f"{col}_f"], color=c, ls=ls, lw=2, label=label)
            a.set(xscale="log", ylim=(0.0, 1.0), xlabel="vote", ylabel=f"mean F-beta (beta {beta:g})",
                  title=f"{emb}, beta {beta:g} (4 photos, 1024 px)")  # fmt: skip
            if i == 0:
                a.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("table", "figure"):
        p = sub.add_parser(cmd)
        p.add_argument("--arm", action="append", required=True)
        p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"table": table, "figure": figure}[a.cmd](a.arm, a.out)


if __name__ == "__main__":
    main()
