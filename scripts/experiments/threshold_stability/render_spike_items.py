"""Render the top spike-causing media items (#2790) to a JSON for the visual report.

Reads ``spike_items.json`` (from ``spike_items.py``), loads each top culprit's COCO
image via the sod dataset loader, draws the class's ground-truth boxes (green — a
*positive* image; a Bad culprit with boxes is a genuinely mislabeled / incomplete-GT
false negative), and emits a base64 JPEG thumbnail plus the item's stats. The
laptop-side report reads this JSON; keeping the render here means the COCO zips never
leave the Grid.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sod"))

from datasets import SodDataset  # noqa: E402


def _annotate(img, boxes, max_px: int = 380):
    from PIL import ImageDraw  # noqa: PLC0415

    img = img.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes or []:
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(46, 204, 113), width=max(3, w // 120))
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render top spike-causing items to JSON (#2790).")
    ap.add_argument("--items", required=True, help="spike_items.json path")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--min-box-frac", type=float, default=0.03)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument(
        "--min-spikes", type=int, default=1, help="only items that spiked in >= this many seeds (confidence)"
    )
    ap.add_argument("--per-class", type=int, default=0, help="if >0, cap items per class (breadth over depth)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.items).read_text())
    pool = [it for it in data["items"] if it["n_spikes"] >= args.min_spikes]
    if args.per_class:
        seen: dict[str, int] = {}
        capped = []
        for it in pool:  # pool is already sorted by n_spikes desc
            k = seen.get(it["cls"], 0)
            if k < args.per_class:
                capped.append(it)
                seen[it["cls"]] = k + 1
        pool = capped
    top = pool[: args.top]
    # One class_split per class → gt_boxes for that class's positives.
    classes = sorted({it["cls"] for it in top})
    gt: dict[str, dict[int, list]] = {}
    with SodDataset(args.dataset) as ds:
        # class slug in spike_items is hyphenated; map back to the sweep's class name.
        name_for = {c: c.replace("-", " ") for c in classes}
        for c in classes:
            try:
                # neg_multiple/seed are required but irrelevant here — we only use
                # gt_boxes (the class's positive boxes) and the id->file locator, both
                # independent of the negative sampling.
                split = ds.class_split(name_for[c], min_box_frac=args.min_box_frac, neg_multiple=1, seed=0)
                gt[c] = dict(split.gt_boxes)
            except Exception as e:  # noqa: BLE001
                print(f"warn: class_split {c}: {e}", file=sys.stderr)
                gt[c] = {}
        cards = []
        for it in top:
            iid = it["image_id"]
            boxes = gt.get(it["cls"], {}).get(iid)
            try:
                b64 = _annotate(ds.load_image(iid), boxes)
            except Exception as e:  # noqa: BLE001
                print(f"warn: render {it['cls']}/{iid}: {e}", file=sys.stderr)
                b64 = None
            cards.append(
                {
                    **{
                        k: it[k]
                        for k in (
                            "cls",
                            "image_id",
                            "gt_label",
                            "n_spikes",
                            "n_votes",
                            "spike_rate",
                            "mean_surface_score",
                            "mean_n_good",
                            "seeds",
                        )
                    },  # fmt: skip
                    "has_gt_box": bool(boxes),
                    "img": b64,
                }
            )
    Path(args.out).write_text(json.dumps({"summary": data["summary"], "cards": cards}))
    print(f"rendered {len(cards)} cards -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
