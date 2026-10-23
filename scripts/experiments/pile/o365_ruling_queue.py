#!/usr/bin/env python3
"""Put the ambiguous Objects365 names (#4817) to the owner as Good/Bad crops in VTSearch.

``pile_config.O365_AMBIGUOUS`` withholds the Objects365 names that are partly a
class under its rule and partly not: Desk for `dining table`, Pot and Bakset for
`bowl`, Toiletry for `bottle`, Scooter for `motorcycle`, and so on. A ruling
moves a name to ``O365_NAMES`` or deletes it, and the designated build has no
relabel, so every ruling comes before the build. The owner chose (2026-10-16) to
rule from samples rather than from the names.

**One queue per class, not per name**, in the shape of ``class_ruling_queue.py``.
Each queue mixes the class's ambiguous names with a few boxes of its own definite
categories (``O365_NAMES``), which are the attention control: a reviewer who is
reading answers those Good, and a one-button sweep does not separate them from the
rest. The crops are shuffled and named by a hash, so nothing on screen says which
name a crop came from; ``manifest.json`` holds the mapping. The question is the
class rule's own reviewer name (``ClassRule.name``), the wording every earlier
slate of the class was voted under.

**What is sampled.** One box per image, so a classroom of thirty desks counts
once, from boxes the loader would hold (no ``iscrowd``, ``isfake`` or
``isreflected``) whose longer side is at least ``MIN_BOX`` pixels. The ruling is
about what a NAME covers, not how small an instance can still be read, and a crop
nobody can judge answers nothing.

``tally`` reads the votes back off the live app and prints each name's Good rate,
which is the evidence the ruling is made from. It writes the votes beside the
manifest every time, because a finished queue is cleared off the dashboard and the
app is then no longer a record of it.

Usage::

    python o365_ruling_queue.py emit --out /expscratch/sgreenberg/o365-4817/queues
    python load_ruling_queues.py --api http://<node>:11850 \\
        --queues /expscratch/sgreenberg/o365-4817/queues --name-template manifest
    python o365_ruling_queue.py tally --queues /expscratch/sgreenberg/o365-4817/queues
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from class_ruling_queue import outlined_crop  # noqa: E402

#: Annotation flags the loader treats as doubtful (``pilebuild.loaders.o365_better.FLAGS``).
FLAGS = ("iscrowd", "isfake", "isreflected")
#: The longer side, in pixels, a box needs before its crop can be judged. The
#: crop widens to ``MIN_WINDOW`` (480) real pixels of context, so at 48 a box was
#: an eighth of its picture: a 19x61 `Bakset` came out as an unreadable sliver.
MIN_BOX = 96
#: Questions per ambiguous name, and attention-control crops per queue.
PER_NAME = 20
PER_CONTROL = 8
SEED = 20261016


def queue_name(cls: str) -> str:
    """The dataset/detector name: class first, so the dashboard groups by class."""
    rule = pc.SCALE_CLASS_RULES[cls].name
    article = "an" if rule[:1].lower() in "aeiou" else "a"
    return f"o365_better {cls} - is the red box {article} {rule}?"


def _sample(boxes: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """*n* boxes from *boxes*, one per image, images in seeded random order."""
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for b in boxes:
        by_image[b["image_id"]].append(b)
    images = sorted(by_image)
    rng.shuffle(images)
    return [rng.choice(by_image[iid]) for iid in images[:n]]


def emit(args: argparse.Namespace) -> int:
    from PIL import Image  # noqa: PLC0415

    data = json.loads(pc.O365_VAL_ANNOTATIONS.read_text())
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    images = {int(im["id"]): im for im in data["images"]}
    wanted = {n for cls in args.classes for n in pc.O365_AMBIGUOUS[cls] + pc.o365_names_for(cls)}
    unknown = sorted(wanted - set(cat_name.values()))
    if unknown:
        raise SystemExit(f"no Objects365 category named {unknown}")
    eligible: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for ann in data["annotations"]:
        name = cat_name.get(ann["category_id"])
        if name not in wanted or any(ann.get(f) for f in FLAGS):
            continue
        x, y, w, h = ann["bbox"]
        if min(w, h) <= 0 or max(w, h) < args.min_box:
            continue
        eligible[name].append({"image_id": int(ann["image_id"]), "ann_id": int(ann["id"]), "bbox": [x, y, w, h]})
    del data

    sizes: list[float] = []
    for cls in args.classes:
        rng = random.Random(f"{args.seed}:{cls}")
        picked: list[dict[str, Any]] = []
        strata: dict[str, int] = {}
        for name in pc.O365_AMBIGUOUS[cls]:
            got = _sample(eligible[name], args.per_name, rng)
            strata[name] = len(got)
            picked += [{**b, "stratum": name, "definite": False} for b in got]
        control = [{**b, "stratum": n} for n in pc.o365_names_for(cls) for b in eligible[n]]
        for b in _sample(control, args.per_control, rng):
            strata[b["stratum"]] = strata.get(b["stratum"], 0) + 1
            picked.append({**b, "definite": True})
        rng.shuffle(picked)

        slug = cls.replace(" ", "_")
        outdir = args.out / slug / "images"
        if outdir.exists():
            raise SystemExit(f"{outdir} exists; emit writes a queue once (its votes map to these files)")
        outdir.mkdir(parents=True)
        manifest = []
        for order, b in enumerate(picked):
            im = images[b["image_id"]]
            with Image.open(pc.O365_VAL_DIR / im["file_name"]) as fh:
                crop = outlined_crop(fh.convert("RGB"), b["bbox"])
            # The filename must not say which name the crop came from.
            stem = hashlib.sha1(f"{args.seed}:{cls}:{b['ann_id']}".encode()).hexdigest()[:12]  # noqa: S324
            path = outdir / f"{slug}_{stem}.jpg"
            crop.save(path, quality=args.quality, optimize=True)
            sizes.append(path.stat().st_size / 1024)
            manifest.append({"file": path.name, "order": order, "file_name": im["file_name"], **b})
        (args.out / slug / "manifest.json").write_text(
            json.dumps(
                {
                    "class": cls,
                    "name": queue_name(cls),
                    "seed": args.seed,
                    "min_box": args.min_box,
                    "strata": strata,
                    "items": manifest,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"{cls:<24} {len(manifest):>3} crops  " + ", ".join(f"{k}={v}" for k, v in strata.items()))
    sizes.sort()
    print(f"\n{len(sizes)} crops; median {sizes[len(sizes) // 2]:.0f} KB, p90 {sizes[int(0.9 * len(sizes))]:.0f} KB")
    print("(budget from the FullMarks passes: ~124 KB median / ~169 KB p90)")
    return 0


def app_base() -> str:
    """The live app, found from the owner's ``vtsearch`` job as ``bank_verdicts.py`` does."""
    squeue = shutil.which("squeue") or "/usr/bin/squeue"
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [squeue, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-t", "R", "-o", "%N"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if not out:
        raise SystemExit("no running `vtsearch` job; pass --api")
    return f"http://{out[0]}:11850"


def votes_of(base: str, name: str) -> dict[str, str]:
    """``{filename: "good"|"bad"}`` off the live app; a file voted both ways is an error."""
    url = f"{base}/api/detectors/{urllib.parse.quote(name)}/labels-detail"
    with urllib.request.urlopen(url, timeout=120) as fh:  # noqa: S310 - our own app on the cluster
        detail = json.load(fh)
    votes: dict[str, str] = {}
    for label in ("good", "bad"):
        for row in detail.get(label, []):
            fn = row.get("filename") or row.get("name")
            if votes.get(fn, label) != label:
                raise SystemExit(f"{name}: {fn} is voted both good and bad")
            votes[fn] = label
    return votes


def tally(args: argparse.Namespace) -> int:
    base = args.api or app_base()
    today = datetime.date.today().isoformat()
    print(f"{'class':<24} {'name':<22} {'good':>4} {'bad':>4} {'open':>4}  good rate")
    for man_path in sorted(args.queues.glob("*/manifest.json")):
        man = json.loads(man_path.read_text())
        votes = votes_of(base, man["name"])
        known = {it["file"] for it in man["items"]}
        stray = sorted(set(votes) - known)
        if stray:
            raise SystemExit(f"{man['name']}: {len(stray)} voted files are not in the manifest, e.g. {stray[0]}")
        (man_path.parent / f"votes_{today}.json").write_text(json.dumps(votes, indent=1, sort_keys=True) + "\n")
        rows: dict[tuple[bool, str], collections.Counter] = collections.defaultdict(collections.Counter)
        for it in man["items"]:
            rows[(it["definite"], it["stratum"])][votes.get(it["file"], "open")] += 1
        for (definite, stratum), c in sorted(rows.items()):
            voted = c["good"] + c["bad"]
            rate = f"{c['good'] / voted:.0%}" if voted else "-"
            label = f"({stratum})" if definite else stratum
            print(f"{man['class']:<24} {label:<22} {c['good']:>4} {c['bad']:>4} {c['open']:>4}  {rate}")
    print("\n(parenthesised rows are the class's own categories: the attention control, expected Good)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="write one queue per class: crops plus manifest.json")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--classes", nargs="+", default=list(pc.O365_AMBIGUOUS), choices=list(pc.O365_AMBIGUOUS))
    e.add_argument("--per-name", type=int, default=PER_NAME)
    e.add_argument("--per-control", type=int, default=PER_CONTROL)
    e.add_argument("--min-box", type=int, default=MIN_BOX)
    e.add_argument("--seed", type=int, default=SEED)
    e.add_argument("--quality", type=int, default=85)
    t = sub.add_parser("tally", help="read the votes off the live app; Good rate per name")
    t.add_argument("--queues", type=Path, required=True)
    t.add_argument("--api", default="", help="default: the node of the running `vtsearch` job")
    args = ap.parse_args()
    return emit(args) if args.cmd == "emit" else tally(args)


if __name__ == "__main__":
    raise SystemExit(main())
