#!/usr/bin/env python3
"""Size vs size (#4160): literal examples -- which test images each training size finds.

    python examples.py --exp <exp dir> --out <report dir> [--per 6]

Reads the per-band prediction dumps that ``run_examples.sh`` writes (seed 0 of
every training size for a few classes, rerun with ``VTS_DUMP_TEST_SCORES``; the
dump that survives is the final click's). Per class and TEST size it draws the
cohort images the training sizes disagree on most. Each thumbnail carries the
object's box, and under it which training sizes put it over their own cut.

These are reruns, so they differ slightly from the grid's own seed-0 rows
(float nondeterminism between runs). A sheet shows the KIND of image an arm
misses, and the report's tables carry the rates. Also writes ``examples.csv``,
every dumped cohort image with each arm's score and cut, so any sheet can be
checked against the numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pile"))
sys.path.insert(0, str(HERE.parent / "calibration"))

TRAIN = {"small": "S", "medium": "M", "large": "L", "mix-equal": "SML=", "mix-natural": "SMLn"}
ORDER = ["S", "M", "L", "SML=", "SMLn"]
BAND = {"small": "S", "medium": "M", "large": "L"}


def _thumbs_module():
    spec = importlib.util.spec_from_file_location("sota_thumbs", HERE.parent / "state_of_app" / "thumbs.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load(dumps: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(dumps.glob("*__band_*.csv")):
        tag, _, band = p.stem.rpartition("__band_")
        cls, _, arm = tag.rpartition("@")
        d = pd.read_csv(p)
        d["cls"] = cls.replace("_", " ")
        d["train"] = TRAIN[arm]
        d["test"] = BAND[band]
        rows.append(d)
    if not rows:
        raise SystemExit(f"no per-band dumps under {dumps}")
    df = pd.concat(rows, ignore_index=True)
    df["hit"] = df["score"] >= df["threshold"]
    return df


def pick(wide: pd.DataFrame, test: str, n: int) -> pd.DataFrame:
    """The images the arms disagree on: half found by the matched size and missed
    by the most other arms, half the reverse, topped up by the widest split."""
    others = [a for a in ORDER if a != test and a in wide]
    if test not in wide or not others:
        return wide.head(0)
    n_other_hits = wide[others].sum(axis=1)
    found_only_matched = wide[wide[test] & (n_other_hits < len(others))]
    found_only_matched = found_only_matched.assign(k=-n_other_hits).sort_values("k")
    missed_by_matched = wide[~wide[test] & (n_other_hits > 0)].assign(k=n_other_hits).sort_values("k", ascending=False)
    half = n // 2
    out = pd.concat([found_only_matched.head(half), missed_by_matched.head(n - half)])
    if len(out) < n:
        rest = wide.drop(out.index, errors="ignore")
        split = rest[ORDER if set(ORDER) <= set(rest) else others].sum(axis=1)
        out = pd.concat([out, rest.assign(k=-(split * (len(ORDER) - split))).sort_values("k").head(n - len(out))])
    return out.drop(columns="k", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per", type=int, default=6, help="images per (class, test size) sheet")
    ap.add_argument("--px", type=int, default=260)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    import common

    common.setup_env()
    import experiment_config as cfg
    import pile_config as pc
    from _cells_io import load_medias

    from vtscore.config import EMBEDDINGS_DIR

    df = load(args.exp / "examples" / "dumps")
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "examples.csv", index=False)
    medias = load_medias(EMBEDDINGS_DIR / cfg.pickle_name("coco_better", "siglip"))
    thumbs = _thumbs_module()
    names = thumbs._filenames({int(i) for i in df["media_id"].unique()})
    zips = [zipfile.ZipFile(z) for z in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP)]

    lines = []
    for (cls, test), g in df.groupby(["cls", "test"], sort=True):
        wide = g.pivot_table(index="media_id", columns="train", values="hit", aggfunc="first").astype(bool)
        rates = g.groupby("train")["hit"].mean()
        chosen = pick(wide, test, args.per)
        if chosen.empty:
            continue
        cols = len(chosen)
        fig, axes = plt.subplots(1, cols, figsize=(2.6 * cols, 3.4))
        axes = np.atleast_1d(axes)
        for ax, (mid, hits) in zip(axes, chosen.iterrows()):
            ax.axis("off")
            jpg = thumbs._thumb(zips, names.get(int(mid), ""), args.px)
            if jpg is None:
                ax.set_title(f"{mid}: no image", fontsize=7)
                continue
            im = Image.open(io.BytesIO(jpg))
            ax.imshow(im)
            w, h = im.size
            cell = f"{cls}@{ {'S': 'small', 'M': 'medium', 'L': 'large'}[test] }"
            for r in medias.get(int(mid), {}).get("regions") or []:
                if r.get("label") == cell:
                    x0, y0, x1, y1 = r["box"]
                    ax.add_patch(
                        patches.Rectangle(
                            (x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec="yellow", lw=1.5
                        )
                    )
            marks = "  ".join(f"{a}{'✓' if hits.get(a, False) else '✗'}" for a in ORDER if a in hits.index)
            ax.set_title(f"id {mid}\n{marks}", fontsize=7)
        fig.suptitle(
            f"{cls}, test {test}, seed 0 at click 150: found (✓) or missed (✗) by each training size.  "
            + "Recall over the whole cohort: "
            + ", ".join(f"{a} {rates[a]:.2f}" for a in ORDER if a in rates.index),
            fontsize=8,
        )
        fig.tight_layout()
        name = f"examples_{cls.replace(' ', '_')}_test_{test}.jpg"
        fig.savefig(figdir / name, dpi=100, pil_kwargs={"quality": 78, "optimize": True})
        plt.close(fig)
        lines.append(f"- `{cls}`, test {test}: [{name}](figures/{name})")
    (args.out / "examples.md").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} sheets -> {figdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
