"""Watch the app work: the images it showed, in click order, and the votes.

Every other artefact in this study is an aggregate. This one is the session --
what the simulated user actually saw, in the order Autopilot chose it, with the
vote they gave and the ground-truth box they would have dragged. One row per
mode over the SAME category and seed, so the three modes can be compared by eye
rather than by number: where they agree, where one wanders into negatives, and
what the first Good actually looked like.

Reads the per-click log `run_cells.py` writes beside each cell
(`task_*__picks.csv`), so it describes the run that happened rather than a
re-simulation of it.

    python pick_sheets.py --exp /expscratch/$USER/scale-3156-fixed \\
        --category "bird@small" --seed 0 --clicks 12 --out sheet.jpg

Green border = the user voted Good, red = Bad. The number under each tile is the
click; `s` is the item's rank in the opening text sort, so a run that keeps
picking from deep in the sort is visibly doing something different from one
working the top.

**`s` is a rank over the WHOLE dataset, not over the pool the run walks.**
``voting_iterations`` builds ``seed_rank`` from every id in ``seed_scores``
while ``pool = sorted(sim_ids)`` is the simulation split -- about half of them
under the default ``sim_fraction=0.5``. So ``s227`` is not "the app went 227
deep": roughly half of those items were never available to pick. The sheet
prints this in its footer, because a rank with an unstated denominator is
exactly the kind of number that gets quoted without it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

# This lives in calibration/ but reads the pile's dataset registry for the
# image roots, so both directories go on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pile"))

import common  # noqa: E402

common.setup_env()

import pile_config as pc  # noqa: E402

VG_DIRS = [pc.DEMO_CACHE / "visual_genome" / "VG_100K", pc.DEMO_CACHE / "visual_genome" / "VG_100K_2"]


def load_picks(exp: str, category: str, seed: str) -> dict[str, list[dict]]:
    """``mode -> [pick rows in click order]`` for one (category, seed)."""
    by_mode: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(Path(exp) / "results" / "cells" / "task_*__picks.csv"))):
        try:
            with open(path, newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r.get("category") == category and r.get("seed") == seed]
        except (OSError, csv.Error):
            continue
        for r in rows:
            mode = f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")
            by_mode.setdefault(mode, []).append(r)
    for v in by_mode.values():
        v.sort(key=lambda r: int(r["t"]))
    return by_mode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--clicks", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", default="siglip/,siglip2_l/,dinov3_patch", help="modes to report as missing if absent")
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw, ImageFont

    from vtscore.datasets import loader as _loader
    from vtscore.eval.labels import region_box_for_category

    from _cells_io import load_medias  # noqa: PLC0415

    by_mode = load_picks(args.exp, args.category, args.seed)
    if not by_mode:
        print(f"no picks for {args.category} seed {args.seed} under {args.exp}")
        return 1
    modes = sorted(by_mode)
    # Say which modes are absent. A sheet that quietly draws two rows where the
    # grid has three reads as "the third behaves like nothing" rather than
    # "the third has not finished", and the difference is invisible on the page.
    if args.expect:
        missing = [m for m in args.expect.split(",") if m.strip() and not any(m.strip() in k for k in modes)]
        if missing:
            print(f"NOTE: no picks yet for {', '.join(missing)} -- not drawn")
    medias = load_medias(_loader.EMBEDDINGS_DIR / "vg_scale__siglip.pkl")
    print(f"{args.category} seed {args.seed}: " + ", ".join(f"{m} {len(by_mode[m])} clicks" for m in modes))

    def path_of(mid: int):
        for d in VG_DIRS:
            p = d / f"{mid}.jpg"
            if p.exists():
                return p
        return None

    TILE, PAD, HDR, CAP = 132, 8, 26, 18
    cols = args.clicks
    row_h = HDR + TILE + CAP + PAD
    FOOT = 16
    sheet = Image.new("RGB", (cols * (TILE + PAD) + PAD, len(modes) * row_h + PAD + FOOT), "#fcfcfb")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 11)
        bold = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = bold = ImageFont.load_default()

    for r, mode in enumerate(modes):
        y0 = PAD + r * row_h
        picks = by_mode[mode][: args.clicks]
        goods = sum(1 for p in picks if p.get("picked_label") == "1")
        draw.text((PAD, y0 + 4), f"{mode}   —   {goods} Good of {len(picks)} clicks shown", fill="#292524", font=bold)
        for c, p in enumerate(picks):
            x0 = PAD + c * (TILE + PAD)
            mid = int(p["picked_id"])
            good = p.get("picked_label") == "1"
            colour = "#15803d" if good else "#b91c1c"
            src = path_of(mid)
            if src is None:
                draw.rectangle([x0, y0 + HDR, x0 + TILE, y0 + HDR + TILE], outline="#d6d3d1", width=1)
                continue
            img = Image.open(src).convert("RGB")
            W, H = img.size
            if good:
                # Show the box the user would have dragged -- on a region-voting
                # run that box IS the vote, and its size against the frame is
                # the thing the mode's whole advantage rests on.
                box = region_box_for_category(medias.get(mid, {}), args.category)
                if box:
                    d = ImageDraw.Draw(img)
                    d.rectangle(
                        [box[0] * W, box[1] * H, box[2] * W, box[3] * H], outline="#15803d", width=max(3, W // 120)
                    )
            img.thumbnail((TILE, TILE))
            tile = Image.new("RGB", (TILE, TILE), "#fcfcfb")
            tile.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
            sheet.paste(tile, (x0, y0 + HDR))
            draw.rectangle([x0, y0 + HDR, x0 + TILE - 1, y0 + HDR + TILE - 1], outline=colour, width=3)
            rank = p.get("picked_seed_rank") or "-"
            draw.text(
                (x0 + 2, y0 + HDR + TILE + 3),
                f"t{p['t']}  s{rank}  {p.get('phase', '')[:4]}",
                fill="#57534e",
                font=font,
            )

    draw.text(
        (PAD, sheet.height - 13),
        "s = rank in the full text sort over all medias; the run walks only the "
        "simulation split (~half), so s overstates depth-into-the-pool by ~2x",
        fill="#a8a29e",
        font=font,
    )
    sheet.save(args.out, quality=92)
    print(f"wrote {args.out} {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
