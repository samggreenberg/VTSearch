"""Turn per-media prediction dumps into inspectable error examples.

Aggregate fpr/fnr says how often the model is wrong. It cannot say whether the
model is wrong or the *label* is. For each dumped cell this lists the highest-
scoring false positives and the lowest-scoring false negatives, with the source
image id and everything the dataset does annotate on that image - which is what
makes "missing label" separable from "model error" by eye.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TOPK = 12


def elide(categories: str, width: int = 78) -> str:
    """Comma-list of *categories*, cut at a comma rather than mid-word.

    Cutting mid-word turns the last label into a misspelling of itself, which the
    spell-check gate then trips over in a generated file nobody would edit.
    """
    cats = categories.replace("|", ", ")
    if len(cats) <= width:
        return cats
    kept = cats[:width].rsplit(", ", 1)[0]
    return f"{kept}, ..."


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    files = sorted(Path(args.dumps).glob("*.csv"))
    if not files:
        print(f"no dumps in {args.dumps}")
        return 1

    for f in files:
        rows = load(f)
        if not rows:
            print(f"\n{f.name}: EMPTY")
            continue
        thr = float(rows[0]["threshold"])
        cat = rows[0]["target_category"]
        for r in rows:
            r["_s"] = float(r["score"])
            r["_y"] = int(r["label"])
        pos = [r for r in rows if r["_y"] == 1]
        neg = [r for r in rows if r["_y"] == 0]
        fps = sorted([r for r in neg if r["_s"] >= thr], key=lambda r: -r["_s"])
        fns = sorted([r for r in pos if r["_s"] < thr], key=lambda r: r["_s"])

        print("\n" + "=" * 104)
        print(f"{f.stem}   target='{cat}'   threshold={thr:.4f}")
        print(f"  test set: {len(rows)} medias, {len(pos)} positive ({100 * len(pos) / len(rows):.1f}%)")
        print(f"  false positives: {len(fps)}/{len(neg)} negatives   false negatives: {len(fns)}/{len(pos)} positives")
        print("=" * 104)

        print(f"\n  TOP FALSE POSITIVES (model most confident, dataset says NOT '{cat}')")
        print(f"  {'score':>8}  {'image':>12}   annotated categories on that image")
        print("  " + "-" * 100)
        for r in fps[: args.topk]:
            print(f"  {r['_s']:8.4f}  {r['filename']:>12}   {elide(r['all_categories'])}")

        print(f"\n  TOP FALSE NEGATIVES (dataset says IS '{cat}', model least confident)")
        print(f"  {'score':>8}  {'image':>12}   annotated categories on that image")
        print("  " + "-" * 100)
        for r in fns[: args.topk]:
            print(f"  {r['_s']:8.4f}  {r['filename']:>12}   {elide(r['all_categories'])}")

        # How many FPs carry a category that is plausibly the same thing?
        near = [r for r in fps if any(cat in c or c in cat for c in r["all_categories"].split("|") if c)]
        if near:
            print(
                f"\n  NOTE: {len(near)}/{len(fps)} false positives carry a category whose name "
                f"contains or is contained by '{cat}' -> likely annotation granularity, not model error"
            )
            for r in near[:6]:
                hits = [c for c in r["all_categories"].split("|") if c and (cat in c or c in cat)]
                print(f"    {r['filename']:>12}  score={r['_s']:.4f}  matched: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
