"""Do two VG names denote the same object? Measure it, don't guess.

VG's vocabulary is free text, so one physical object arrives under several
names. `glasses` / `sunglasses` / `reading glasses` would be a genuinely
interesting fine-grained target -- *if* the labels could be trusted. They can't
be assumed to: `glasses` might be a superset of `sunglasses`, disjoint from it,
or overlapping at each annotator's whim, and the three cases want three
different experiments.

The overview benchmark already got burned reaching for a cheap proxy: its error
report flagged false positives carrying a name that *contains* the target,
which for `bus` matched 80 images annotated **`bush`**
(`docs/experiments/overview-bench/REPORT.md`). String similarity is not evidence
about objects. Box geometry is.

**The test.** On images where both names appear, ask how often an `a` box
lands on the same pixels as a `b` box (IoU >= `--iou`). Same pixels, two names
means one object annotated twice:

* **high both ways** -- an *alias*. The names are one label split arbitrarily,
  so each one's negative pool is poisoned by the other and neither can be used
  until they are merged.
* **high one way only** -- a *subtype* (`sunglasses` boxes sit on `glasses`
  boxes, but most `glasses` are not `sunglasses`). This is the interesting
  case: a real fine-grained pair, where the broad name's negatives are sound.
* **near zero** -- *distinct* objects that merely co-occur.

**What it cannot resolve.** A pair that never co-occurs on any image is
reported as `untestable`, not as distinct. Two names systematically split
across annotators -- one says `glasses`, another says `spectacles`, neither
ever labels the same image -- produce exactly the same emptiness as two genuinely
unrelated words. Distinguishing those needs the human pass, and this script's
job is to say which pairs are worth spending it on.

Usage::

    python scan_name_overlap.py --names glasses,sunglasses,eyeglasses,spectacles
    python scan_name_overlap.py --from-shortlist shortlist.json --top 30
    python scan_name_overlap.py --names bus,bush --iou 0.5   # the false lead, refuted
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pile_config as pc

pc.setup_env()

VG_ROOT = pc.DEMO_CACHE / "visual_genome"
OBJECTS_JSON = VG_ROOT / "objects.json"

Box = tuple[float, float, float, float]


def log(msg: str) -> None:
    print(f"[overlap] {msg}", flush=True)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def scan(wanted: set[str], thresh: float) -> dict:
    """Per-name counts and per-pair box-overlap rates over the whole of VG.

    Pixel boxes are compared directly: IoU is scale-free, so unlike the area
    banding this needs no image dimensions and no dims cache.
    """
    log(f"loading {OBJECTS_JSON.name} ({OBJECTS_JSON.stat().st_size / 1e6:.0f} MB)...")
    t0 = time.time()
    with OBJECTS_JSON.open() as fh:
        records = json.load(fh)
    log(f"  parsed {len(records)} image records in {time.time() - t0:.0f}s")

    n_images: dict[str, int] = defaultdict(int)
    n_boxes: dict[str, int] = defaultdict(int)
    co_images: dict[tuple[str, str], int] = defaultdict(int)
    # (a, b) -> how many a-boxes have an on-target b-box, on co-annotated images
    hits: dict[tuple[str, str], int] = defaultdict(int)
    considered: dict[tuple[str, str], int] = defaultdict(int)

    for rec in records:
        by_name: dict[str, list[Box]] = defaultdict(list)
        for obj in rec.get("objects") or []:
            names = obj.get("names") or []
            if not names:
                continue
            name = str(names[0]).strip().lower()
            if name not in wanted:
                continue
            x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
            w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
            if w > 0 and h > 0:
                by_name[name].append((x, y, x + w, y + h))
        for name, boxes in by_name.items():
            n_images[name] += 1
            n_boxes[name] += len(boxes)
        for a, b in combinations(sorted(by_name), 2):
            co_images[(a, b)] += 1
            # Directional: what fraction of a's boxes are covered by some b box,
            # and vice versa. Asymmetry is what separates a subtype from an alias.
            for first, second in ((a, b), (b, a)):
                considered[(first, second)] += len(by_name[first])
                hits[(first, second)] += sum(
                    1 for box in by_name[first] if any(iou(box, other) >= thresh for other in by_name[second])
                )

    pairs = []
    for a, b in sorted(co_images):
        fwd = hits[(a, b)] / considered[(a, b)] if considered[(a, b)] else 0.0
        rev = hits[(b, a)] / considered[(b, a)] if considered[(b, a)] else 0.0
        pairs.append(
            {
                "a": a,
                "b": b,
                "co_images": co_images[(a, b)],
                "a_on_b": round(fwd, 4),
                "b_on_a": round(rev, 4),
                "verdict": classify(co_images[(a, b)], fwd, rev),
            }
        )
    # Pairs that never co-occur are untestable rather than absent - say so, so a
    # missing row is never read as "measured, and they are distinct".
    for a, b in combinations(sorted(wanted), 2):
        if (a, b) not in co_images:
            pairs.append({"a": a, "b": b, "co_images": 0, "a_on_b": 0.0, "b_on_a": 0.0, "verdict": "untestable"})

    return {
        "meta": {"iou_threshold": thresh, "names": sorted(wanted)},
        "names": {n: {"n_images": n_images[n], "n_boxes": n_boxes[n]} for n in sorted(wanted)},
        "pairs": sorted(pairs, key=lambda p: (-max(p["a_on_b"], p["b_on_a"]), p["a"], p["b"])),
    }


#: Below this many co-annotated images a rate is noise, not a measurement.
MIN_CO_IMAGES = 10
#: Rates above/below these separate "the same object" from "different objects".
HIGH, LOW = 0.30, 0.05


def classify(co: int, fwd: float, rev: float) -> str:
    if co == 0:
        return "untestable"
    if co < MIN_CO_IMAGES:
        return "thin"
    if fwd >= HIGH and rev >= HIGH:
        return "alias"
    if max(fwd, rev) >= HIGH and min(fwd, rev) < LOW:
        return "subtype"
    if max(fwd, rev) < LOW:
        return "distinct"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="", help="comma-separated VG names to compare")
    ap.add_argument("--from-shortlist", default="", help="shortlist_scale_classes.py --out JSON")
    ap.add_argument("--top", type=int, default=30, help="with --from-shortlist, how many classes to take")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
    ap.add_argument("--out", default="", help="also write the full result as JSON")
    args = ap.parse_args()

    wanted = {n.strip().lower() for n in args.names.split(",") if n.strip()}
    if args.from_shortlist:
        rows = json.loads(Path(args.from_shortlist).read_text())
        wanted |= {r["category"] for r in rows[: args.top]}
    if len(wanted) < 2:
        raise SystemExit("need at least two names; pass --names or --from-shortlist")
    if not OBJECTS_JSON.exists():
        raise SystemExit(f"missing {OBJECTS_JSON}")

    result = scan(wanted, args.iou)

    log("")
    log(f"=== per-name support ({len(result['names'])} names) ===")
    for name, s in sorted(result["names"].items(), key=lambda kv: -kv[1]["n_images"]):
        log(f"  {name:<24}{s['n_images']:>8} images{s['n_boxes']:>9} boxes")

    log("")
    log(f"=== box overlap at IoU >= {args.iou} ===")
    log(f"  {'a':<20}{'b':<20}{'co-imgs':>9}{'a on b':>9}{'b on a':>9}  verdict")
    for p in result["pairs"]:
        if p["verdict"] in ("distinct", "untestable") and p["co_images"] < MIN_CO_IMAGES:
            continue
        log(f"  {p['a']:<20}{p['b']:<20}{p['co_images']:>9}{p['a_on_b']:>9.3f}{p['b_on_a']:>9.3f}  {p['verdict']}")

    counts: dict[str, int] = defaultdict(int)
    for p in result["pairs"]:
        counts[p["verdict"]] += 1
    log("")
    log(f"  verdicts: {dict(sorted(counts.items()))}")
    log("  alias    -> merge the names, or neither one's negatives are sound")
    log("  subtype  -> a real fine-grained pair; the broad name's negatives are sound")
    log("  untestable -> never co-annotated; distinct and split-by-annotator look identical here")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
