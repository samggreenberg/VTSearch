"""Is a false positive a model error, or a missing label?

Test: pick categories that cannot occur without the target (you cannot have
clouds without sky). If images the model flags are ENRICHED for those relative
to the images it correctly rejects, the "false" positives are largely
un-annotated instances of the target, not model errors.

Enrichment is measured against the model's own true negatives, so it is not
confounded by the model simply preferring outdoor scenes: both groups are
images the dataset labels negative; the only difference is what the model said.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Categories that logically entail the target: if the image carries one of
# these, the target is present whether or not it was annotated. Two kinds
# qualify, and both are entailment for this test's purpose:
#   - a part or a consequence ("clouds" cannot appear without sky; a "face" has
#     a nose),
#   - a narrower name for the same object ("sunglasses" ARE glasses), which is
#     how a free-text vocabulary splits one thing across several labels.
ENTAILS = {
    "sky": ["cloud", "clouds"],
    "bus": ["bus stop", "bus station"],
    "clock": ["clock tower"],
    "nose": ["face"],
    "glasses": ["sunglasses", "eyeglasses"],
}
# Weaker signal: the target's typical context.
CONTEXT = {
    "sky": ["tree", "building", "grass", "mountain", "roof", "water", "field", "road"],
    "clock": ["wall", "tower", "desk", "table"],
    "nose": ["eye", "eyes", "hair", "mouth", "head", "ear"],
    "glasses": ["eyes", "eye", "nose", "hair", "cap", "mouth"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    args = ap.parse_args()

    for f in sorted(Path(args.dumps).glob("*.csv")):
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        thr = float(rows[0]["threshold"])
        cat = rows[0]["target_category"]
        neg = [r for r in rows if int(r["label"]) == 0]
        fps = [r for r in neg if float(r["score"]) >= thr]
        tns = [r for r in neg if float(r["score"]) < thr]
        if not fps or not tns:
            continue

        def frac(group, terms):
            if not group:
                return 0.0
            hit = 0
            for r in group:
                cats = {c.strip().lower() for c in r["all_categories"].split("|") if c.strip()}
                if any(t in cats for t in terms):
                    hit += 1
            return hit / len(group)

        print("\n" + "=" * 96)
        print(f"{f.stem}  target='{cat}'   {len(fps)} false positives vs {len(tns)} true negatives")
        print("=" * 96)

        ent = ENTAILS.get(cat)
        if ent:
            a, b = frac(fps, ent), frac(tns, ent)
            print(f"  ENTAILING categories {ent}:")
            print(f"    present on {100 * a:5.1f}% of false positives")
            print(f"    present on {100 * b:5.1f}% of true negatives")
            if a == 0.0:
                # No evidence either way. 0/0 is not infinite enrichment - it is
                # an untestable case, and printing a conclusion here would have
                # falsely indicted COCO, whose annotation is exhaustive.
                print(
                    "    enrichment: NOT TESTABLE (the entailing categories appear on "
                    "neither group;\n       this vocabulary has no term that entails "
                    f"'{cat}')"
                )
                ratio = 0.0
            else:
                ratio = (a / b) if b else float("inf")
                print(f"    enrichment: {ratio:.1f}x")
            if a > 0.0 and ratio > 1.5:
                print(
                    f"    -> the flagged images are enriched for things that CANNOT occur without "
                    f"'{cat}';\n       these are largely MISSING LABELS, not model errors"
                )

        ctx = CONTEXT.get(cat)
        if ctx:
            a, b = frac(fps, ctx), frac(tns, ctx)
            ratio = (a / b) if b else float("inf")
            print(
                f"  CONTEXT categories (any of {len(ctx)}): "
                f"{100 * a:.1f}% of FPs vs {100 * b:.1f}% of TNs  ({ratio:.2f}x)"
            )

        # What actually co-occurs on the flagged images.
        from collections import Counter

        cnt = Counter()
        for r in fps:
            for c in r["all_categories"].split("|"):
                c = c.strip().lower()
                if c and c != cat:
                    cnt[c] += 1
        top = ", ".join(f"{c}({n})" for c, n in cnt.most_common(12))
        print(f"  most common co-annotations on false positives: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
