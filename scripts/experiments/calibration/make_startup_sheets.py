"""The opening as pictures: what each arm actually clicked on.  (#3267)

`analyze_startup.py` answers *whether* an opening mined better - how many
positives it found, how deep it had to reach, what the detector was worth
afterwards.  The issue asks a second question that no aggregate can answer:

    "I'll want to see examples of how it was best.  And even WHY it was best."

That is a question about the items, so it has to be answered with the items.
This renders one contact sheet per arm for a chosen cell: the opening's clicks
**in the order the arm made them**, each captioned with its round, its rank in
the seed sort, and whether it turned out to be a positive, with the dataset's
ground-truth box drawn where it has one.  Read two arms' sheets side by side and
the mechanism is visible rather than inferred - `deep_first` picking near-misses
where `easy_med_hard` picks the object, or two arms finding the same count of
positives that are not the same *kind* of positive.

Runs on the GRID, because that is where the source images are.

    python make_startup_sheets.py --cell coco_val:knife:0 \\
        --out /expscratch/$USER/good-mining-3267/analysis/figures

`--cell` is `dataset:category:seed`; omit it and the script picks the cell where
the arms' positive counts differ most, which is the one worth looking at.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

DATA = Path(os.environ.get("VTSEARCH_DATA_DIR", "/expscratch/sgreenberg/vts-cache/datadir"))
IMAGE_ROOTS: dict[str, list[Path]] = {
    "visual_genome_m": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_small": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_medium": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "vg_box_large": [DATA / "visual_genome/VG_100K", DATA / "visual_genome/VG_100K_2"],
    "coco_val": [Path("/exp/scale26/datasets/external/COCO/images/val2017.zip")],
}
BOX_PICKLES = {ds: DATA / f"embeddings/{ds}__siglip.pkl" for ds in IMAGE_ROOTS}

ARMS = ("prod", "top_long", "easy_med_hard", "band_wide", "incl_k", "incl_k_wide", "flat_mid", "deep_first")
THUMB = 300


class _Images:
    """Open source images from a directory tree or from COCO's zip, uniformly."""

    def __init__(self, dataset: str) -> None:
        self.roots = IMAGE_ROOTS.get(dataset, [])
        self._zips: dict[Path, zipfile.ZipFile] = {}
        self._names: dict[Path, dict[str, str]] = {}

    def open(self, file_id: str) -> "Image.Image | None":
        stem = Path(str(file_id)).stem
        for root in self.roots:
            if root.suffix == ".zip":
                if root not in self._zips:
                    if not root.exists():
                        continue
                    self._zips[root] = zipfile.ZipFile(root)
                    self._names[root] = {Path(n).stem: n for n in self._zips[root].namelist()}
                name = self._names[root].get(stem)
                if name:
                    return Image.open(io.BytesIO(self._zips[root].read(name))).convert("RGB")
            else:
                for ext in (".jpg", ".jpeg", ".png", ".JPEG"):
                    p = root / f"{stem}{ext}"
                    if p.exists():
                        return Image.open(p).convert("RGB")
        return None


def _load_meta(dataset: str) -> dict:
    """`media_id -> media dict`, for the file id and the ground-truth boxes."""
    pkl = BOX_PICKLES.get(dataset)
    if not pkl or not pkl.exists():
        return {}
    with open(pkl, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - this study own dataset pickle


def load_picks(root: Path, dataset: str, category: str, seed: int) -> pd.DataFrame:
    """Every arm's pick log for one cell, tagged with `arm`."""
    frames = []
    for arm in ARMS:
        cells = root / arm / "cells"
        if not cells.is_dir():
            continue
        for f in sorted(cells.glob("task_*__picks.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:  # noqa: BLE001, S112
                continue
            if df.empty:
                continue
            m = (df["dataset"] == dataset) & (df["category"] == category) & (df["seed"] == seed)
            if not m.any():
                continue
            g = df[m].copy()
            g["arm"] = arm
            frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pick_best_cell(root: Path) -> tuple[str, str, int]:
    """The cell whose arms disagree most about the opening's positives.

    A cell where every arm mines the same number is a cell where there is
    nothing to look at; the interesting sheet is the one with the widest spread,
    and choosing it from the data beats choosing it by hand.
    """
    rows = []
    for arm in ARMS:
        cells = root / arm / "cells"
        if not cells.is_dir():
            continue
        for f in sorted(cells.glob("task_*__picks.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:  # noqa: BLE001, S112
                continue
            if df.empty or "startup_round" not in df:
                continue
            op = df[df["startup_round"] >= 0]
            if op.empty:
                # `prod` has no schedule, so its opening is the phase machine's:
                # everything before the first trained phase.
                op = df[df["phase"].astype(str).str.startswith(("good", "bad", "s"))]
            for key, g in op.groupby(["dataset", "category", "seed"]):
                rows.append(
                    {
                        "arm": arm,
                        "dataset": key[0],
                        "category": key[1],
                        "seed": key[2],
                        "n_pos": int(g["picked_label"].sum()),
                    }
                )
    if not rows:
        raise SystemExit("no pick logs found - has the array produced cells yet?")
    d = pd.DataFrame(rows)
    spread = d.groupby(["dataset", "category", "seed"])["n_pos"].agg(lambda s: s.max() - s.min())
    ds, cat, seed = spread.idxmax()
    print(f"cell with the widest opening-positive spread: {ds}/{cat}/seed{seed} (spread {spread.max()})")
    return ds, cat, int(seed)


def _held(df: pd.DataFrame) -> pd.Series:
    """Per click: was it spent past the written schedule, held for a quorum?

    Recorded by the harness (`startup_held`). A pick log predating that column
    has no way to know, so it answers False and the caption simply says nothing
    about an overrun rather than claiming there was none.
    """
    if "startup_held" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["startup_held"].fillna(False).astype(bool)


def render_arm(
    g: pd.DataFrame,
    dataset: str,
    category: str,
    seed: int,
    meta: dict,
    images: _Images,
    out: Path,
    total_held: int = 0,
) -> Path | None:
    from vtscore.eval.labels import region_box_for_category

    g = g.sort_values("t")
    n = len(g)
    if n == 0:
        return None
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.6 * rows))
    axes = [axes] if rows * cols == 1 else list(axes.ravel())
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, g.iterrows()):
        m = meta.get(int(r["picked_id"])) or {}
        # `filename` is the key the media dicts actually carry (`000000000139.jpg`,
        # `2307.jpg`); COCO`s zip stores it under `val2017/`, which _Images
        # matches on the stem. There is no `file_id` on these medias - asking
        # for one renders a grid of "image not found" that still looks like a
        # working sheet.
        img = images.open(m.get("filename") or m.get("origin_name") or "")
        if img is not None:
            w, h = img.size
            img.thumbnail((THUMB, THUMB))
            ax.imshow(img)
            box = region_box_for_category(m, category)
            if box is not None:
                sx, sy = img.size[0] / 1.0, img.size[1] / 1.0
                x0, y0, x1, y1 = box
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0 * sx, y0 * sy),
                        (x1 - x0) * sx,
                        (y1 - y0) * sy,
                        fill=False,
                        edgecolor="#39d353",
                        linewidth=2.0,
                    )
                )
        else:
            ax.text(0.5, 0.5, "image\nnot found", ha="center", va="center", fontsize=8)
        hit = int(r["picked_label"]) == 1
        rnd = int(r["startup_round"]) if pd.notna(r["startup_round"]) else -1
        ax.set_title(
            f"t={int(r['t'])} {'GOOD' if hit else 'bad'}\n"
            f"round {rnd if rnd >= 0 else '-'} · rank {int(r['picked_seed_rank'])}"
            # `picked_seed_percentile` is a FRACTION of the sort (0 = top), not
            # a percent, whatever the name says - read it off a pick log before
            # trusting either.
            f" ({100.0 * float(r['picked_seed_percentile']):.1f}%)",
            fontsize=7,
            color=("#1a7f37" if hit else "#b91c1c"),
        )
        ax.axis("off")
    arm = str(g["arm"].iloc[0])
    sched = str(g["startup_schedule"].iloc[0])
    if not sched or sched == "nan":
        sched = "(app default: g3@top,b4@mid)"
    n_pos = int(g["picked_label"].sum())
    n_written = int((~_held(g)).sum())
    shown_held = int(_held(g).sum())
    # Say what is on the sheet AND what is not.  A starved arm is held on its
    # last round for the whole horizon, so its opening really is 200 clicks;
    # trimming that to a readable grid without saying so would make the arm
    # look merely short instead of stuck, which is the opposite of the finding.
    tail = ""
    if total_held:
        hidden = total_held - shown_held
        tail = f", then held {total_held} more clicks waiting for a first positive"
        if hidden > 0:
            tail += f" ({shown_held} shown, {hidden} not)"
    fig.suptitle(
        f"{arm} — {sched}\n{dataset} / {category} / seed {seed}: "
        f"{n_pos} positives in {n_written} opening clicks as written{tail}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"opening_{dataset}_{category.replace(' ', '-')}_s{seed}_{arm}.jpg"
    fig.savefig(p, dpi=130, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"  wrote {p.name}  ({n_pos} positive; {n_written} written, {total_held} held)")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Contact sheets of each arm's opening clicks (#3267).")
    ap.add_argument("--results", default=os.environ.get("CALIB_EXP", "") + "/results")
    ap.add_argument("--cell", default=None, help="dataset:category:seed; default = widest-spread cell")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max-extended",
        type=int,
        default=8,
        help="How many held-past-the-schedule clicks to render (0 = none); the rest are captioned.",
    )
    args = ap.parse_args(argv)

    root = Path(args.results)
    if args.cell:
        ds, cat, seed = args.cell.split(":")
        seed = int(seed)
    else:
        ds, cat, seed = pick_best_cell(root)

    picks = load_picks(root, ds, cat, seed)
    if picks.empty:
        raise SystemExit(f"no picks for {ds}/{cat}/seed{seed}")
    # The opening only.  A schedule arm tags its rounds; `prod` has no schedule,
    # so its opening is every click before the first trained phase.
    op = picks[picks["startup_round"] >= 0]
    prod = picks[picks["arm"] == "prod"]
    if not prod.empty:
        pre = prod[prod["phase"].astype(str).isin(("good", "bad"))]
        op = pd.concat([op, pre], ignore_index=True).drop_duplicates(subset=["arm", "t"])

    # Show the opening as WRITTEN in full, plus a sample of what it was reduced
    # to afterwards. A starved arm's opening is the whole 200-click horizon, and
    # 200 thumbnails is unreadable - which would hide the very cell worth
    # looking at. The caption reports how many are not shown.
    held = _held(op)
    op = pd.concat(
        [op[~held], op[held].sort_values("t").head(max(0, args.max_extended))],
        ignore_index=True,
    )

    meta = _load_meta(ds)
    images = _Images(ds)
    out = Path(args.out)
    written = []
    for arm in ARMS:
        g = op[op["arm"] == arm].sort_values("t")
        if g.empty:
            continue
        total_held = int(_held(picks[picks["arm"] == arm]).sum())
        p = render_arm(g, ds, cat, seed, meta, images, out, total_held=total_held)
        if p:
            written.append(p)
    print(f"\n{len(written)} sheets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
