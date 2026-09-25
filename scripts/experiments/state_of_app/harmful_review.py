#!/usr/bin/env python3
"""State of the App (#4179): put the pairs the click loop flags as harmful in front of a human.

    python harmful_review.py build --pairs <analysis-binary>/harmful_pairs.csv \\
        --influence <analysis-binary>/influence.csv --z 5 --out <queues dir>
    python ../pile/load_ruling_queues.py --api http://<node>:11850 --queues <queues dir> --name-template manifest
    python harmful_review.py build --extend --z 3 ...   # deeper cut, SAME queues
    python harmful_review.py refresh --api http://<node>:11850 --queues <queues dir>
    python harmful_review.py bank --api http://<node>:11850 --queues <queues dir>

Each row of ``harmful_pairs.csv`` asks one question: *is this image really a
<label> for <class>?* This command turns each row into one VTSearch Good/Bad
question, one dataset per class (the standing review format). **Good always
means "there is a <class> here"**, whichever way COCO labelled the image, so a
reviewer never flips the meaning of the button between questions:

* a **negative** is shown WHOLE, with no box, because the question is whether
  the object is anywhere in it. Good = COCO missed one: a label error.
  Bad = a correct negative.
* a **positive** is shown whole with its banded box drawn. That is the largest
  instance, the one the build bands on and the simulated user drags (#4096).
  Good = the box holds ONE <class> by the class's rule. Bad = a label error.

The class's review rule (``ClassRule.name``) is the dataset name, because the
name is the only definition a reviewer sees while voting.

**Controls.** A queue of flagged negatives should come back mostly Bad, and a
reviewer sweeping Bad would produce the same answers. So each queue also mixes
in a few of the class's own positives, drawn from the build's banded supply and
shown whole with no box, just like the negatives. Their right answer is Good.
Nothing in a file name says which kind a question is: the stem is a hash.

Classes with fewer than ``--min-per-class`` questions share one ``mixed``
queue, and each image there carries its class rule as a banner.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import random
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pile"))

import pile_config as pc  # noqa: E402

MAX_SIDE = 900
QUALITY = 85
MIXED = "mixed"
#: The bands a control positive is drawn from (see its use).
CONTROL_BANDS = ("medium", "large")
#: Good means "there is a <class> here" for every question; the two readings of it.
VERDICT = {(0, "Good"): "label_error", (0, "Bad"): "correct_negative", (1, "Good"): "correct_positive", (1, "Bad"): "label_error"}


def _rule_name(cls: str) -> str:
    rule = pc.SCALE_CLASS_RULES.get(cls)
    return rule.name if rule else cls


#: Where the class's rule already settles a near-miss the flagged images are full
#: of, the name says so: the name is all a reviewer reads while voting.
HINT = {"keyboard": ", a laptop's own is Bad"}


def queue_name(cls: str) -> str:
    """The dataset/detector name: the class rule first, then the one question every image asks."""
    if cls == MIXED:
        return "coco_better mixed - is the class on the banner here? (if boxed, is the box one?)"
    return f"{_rule_name(cls)}{HINT.get(cls, '')} - is there one here? (if boxed, is the box one?)"


def _bands(influence: Path, keys: set[tuple[int, str, int]]) -> dict[tuple[int, str, int], str]:
    """The band each flagged pair was clicked in most often (a positive's cell)."""
    inf = pd.read_csv(influence, usecols=["image_id", "class", "label", "band"])
    inf = inf[[k in keys for k in zip(inf["image_id"], inf["class"], inf["label"], strict=True)]]
    top = inf.groupby(["image_id", "class", "label"])["band"].agg(lambda s: s.value_counts().index[0])
    return {(int(i), c, int(lab)): b for (i, c, lab), b in top.items()}


def build(args) -> int:
    from pilebuild.loaders.coco_better import lump_exclusions, read_coco_labels  # noqa: PLC0415
    from pilebuild.scale_core import band_for, largest_box  # noqa: PLC0415

    pairs = pd.read_csv(args.pairs)
    pairs = pairs[pairs["z"] < -args.z]
    keys = {(int(r.image_id), r["class"], int(r.label)) for _, r in pairs.iterrows()}
    band_of = _bands(args.influence, keys)
    labels, dims, filenames = read_coco_labels(pc.COCO_ANCHOR_DIR, pc.SCALE_CLASSES)
    lumped = lump_exclusions(labels)
    rng = random.Random(args.seed)

    by_cls: dict[str, list[dict]] = collections.defaultdict(list)
    for _, r in pairs.iterrows():
        iid, cls, lab = int(r.image_id), r["class"], int(r.label)
        item = {
            "image_id": iid,
            "class": cls,
            "label": lab,
            "arm": "flagged",
            "z": float(r.z),
            "resid": float(r.resid),
            "n_clicks": int(r.n_clicks),
            "band": band_of.get((iid, cls, lab)),
            "box": None,
            "coco_holds": sorted(labels.get(iid, {})),
        }
        if lab == 1:
            boxes = labels.get(iid, {}).get(cls)
            if not boxes:
                raise SystemExit(f"{iid} is a {cls} positive but COCO holds no {cls} box")
            item["box"] = largest_box(boxes)
            got = band_for([item["box"]], *dims[iid])
            if got != item["band"]:
                print(f"  note: {iid} {cls} clicked at {item['band']}, its box bands {got}")
        elif labels.get(iid, {}).get(cls):
            # Only the lump filter makes an image that COCO says holds the class a
            # negative for it; say so, since then the answer is known to be Good.
            item["note"] = "coco holds this class" + (" (lump-excluded)" if (iid, cls) in lumped else "")
        by_cls[cls].append(item)

    # --extend: questions already asked stay where they are, and new ones join
    # their class's existing queue, so a deeper cut lands in the SAME datasets.
    old: dict[str, list[dict]] = {}
    asked: set[tuple[int, str]] = set()
    if args.extend:
        for man in sorted(args.out.glob("*/manifest.json")):
            m = json.loads(man.read_text())
            old[m["class"]] = m["items"]
            asked |= {(it["image_id"], it["class"]) for it in m["items"]}
        by_cls = {c: [it for it in its if (it["image_id"], c) not in asked] for c, its in by_cls.items()}
        by_cls = {c: its for c, its in by_cls.items() if its}

    # Controls: the class's own positives, shown like a negative (whole, no box).
    for cls, items in by_cls.items():
        k = max(1 if args.extend else args.min_controls, round(args.control_share * len(items)))
        taken = {it["image_id"] for it in items} | {i for i, c in asked if c == cls}
        supply = sorted(
            i
            for i, m in labels.items()
            if m.get(cls)
            and i not in taken
            and (i, cls) not in lumped
            # Not `small`: shown whole with no box, a small positive is a hunt,
            # and a missed one reads as a label error that is not there (tv's
            # first control was a background screen in a luggage display).
            and band_for([largest_box(m[cls])], *dims[i]) in CONTROL_BANDS
        )
        for iid in rng.sample(supply, k):
            items.append({"image_id": iid, "class": cls, "label": 1, "arm": "control", "box": None, "band": None})

    queues: dict[str, list[dict]] = collections.defaultdict(list)
    for cls, items in by_cls.items():
        flagged = sum(it["arm"] == "flagged" for it in items)
        queues[cls if cls in old or flagged >= args.min_per_class else MIXED] += items
    return _render(queues, dims, filenames, args, old)


def _render(queues: dict[str, list[dict]], dims: dict, filenames: dict, args, old: dict | None = None) -> int:
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    members: dict[str, tuple[Path, str]] = {}
    for zp in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP):
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                if nm.lower().endswith(".jpg"):
                    members[Path(nm).name] = (zp, nm)
    zc: dict[Path, zipfile.ZipFile] = {}
    font = ImageFont.load_default(size=34)
    rng = random.Random(args.seed)
    total = 0
    for q, items in sorted(queues.items()):
        rng.shuffle(items)
        slug = q.replace(" ", "_")
        outdir = args.out / slug / "images"
        outdir.mkdir(parents=True, exist_ok=True)
        manifest = list((old or {}).get(q, []))
        for it in items:
            zp, member = members[filenames[it["image_id"]]]
            zf = zc.setdefault(zp, zipfile.ZipFile(zp))
            im = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            if it["box"] is not None:
                draw = ImageDraw.Draw(im)
                x0, y0, x1, y1 = it["box"]
                lw = max(2, int(0.006 * max(im.size)))
                draw.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=(255, 255, 255), width=lw)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 64, 0), width=lw)
            s = MAX_SIDE / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            if q == MIXED:
                banner = Image.new("RGB", (im.width, 56), (20, 20, 20))
                ImageDraw.Draw(banner).text((12, 8), _rule_name(it["class"]), fill=(255, 220, 0), font=font)
                both = Image.new("RGB", (im.width, im.height + 56))
                both.paste(banner, (0, 0))
                both.paste(im, (0, 56))
                im = both
            stem = hashlib.sha1(f"{args.seed}:{it['image_id']}:{it['class']}".encode()).hexdigest()[:12]  # noqa: S324
            p = outdir / f"{slug}_{stem}.jpg"
            im.save(p, quality=QUALITY, optimize=True)
            manifest.append({"file": p.name, **it})
        (args.out / slug / "manifest.json").write_text(
            json.dumps({"class": q, "name": queue_name(q), "seed": args.seed, "items": manifest}, indent=1)
        )
        n_flag = sum(i["arm"] == "flagged" for i in manifest)
        added = len(manifest) - len((old or {}).get(q, []))
        print(f"  {queue_name(q)}: {n_flag} flagged + {len(manifest) - n_flag} controls ({added} new)")
        total += len(manifest)
    print(f"{len(queues)} queues, {total} questions -> {args.out}")
    return 0


def _get(base: str, path: str):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=120) as fh:  # noqa: S310 - our own app
        return json.loads(fh.read().decode())


def bank(args) -> int:
    """Votes -> ``verdicts.jsonl`` beside the queues, one row per question answered.

    Reads each queue's detector through the app and matches a vote to its
    question by file name. Never writes to the app and never deletes.
    """
    dets = {d["name"]: d for d in _get(args.api, "/api/detectors/registry").get("detectors", [])}
    rows, missing = [], 0
    for man in sorted(args.queues.glob("*/manifest.json")):
        m = json.loads(man.read_text())
        if m["name"] not in dets:
            continue  # cleared, or never loaded; its banked rows stand
        live = _get(args.api, f"/api/detectors/{urllib.parse.quote(m['name'])}/labels-detail")
        votes = {r["filename"]: "Good" for r in live.get("good", [])}
        votes.update({r["filename"]: "Bad" for r in live.get("bad", [])})
        for it in m["items"]:
            v = votes.get(it["file"])
            if v is None:
                missing += 1
                continue
            rows.append({**it, "queue": m["name"], "vote": v, "verdict": VERDICT[(it["label"], v)]})
        print(f"  {m['name']}: {sum(it['file'] in votes for it in m['items'])}/{len(m['items'])} answered")
    # MERGE into the banked file, never replace it: once a finished queue is
    # cleared from the dashboard, the banked row is the only copy of its vote.
    out = args.queues / "verdicts.jsonl"
    banked = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            banked[(r["queue"], r["file"])] = r
    fresh = sum((r["queue"], r["file"]) not in banked for r in rows)
    changed = sum((r["queue"], r["file"]) in banked and banked[(r["queue"], r["file"])]["vote"] != r["vote"] for r in rows)
    banked.update({(r["queue"], r["file"]): r for r in rows})
    out.write_text("".join(json.dumps(r) + "\n" for r in banked.values()))
    print(f"{len(banked)} verdicts banked ({fresh} new, {changed} changed), {missing} unanswered live -> {out}")
    return 0


def refresh(args) -> int:
    """Grow each queue's dataset to its folder in place, keeping its name and its detector.

    The app cannot add files to a registered dataset, so the grown folder is
    imported under a temporary name, and only once it has fully landed is the
    old dataset deleted and the new one renamed into its place. The detector,
    which holds the votes, is never touched: labels key on file content, and the
    files already asked are the same bytes. A queue with no dataset yet is left
    for ``load_ruling_queues.py``, which creates the pair.
    """
    import time  # noqa: PLC0415

    from load_ruling_queues import api, count_of, datasets  # noqa: PLC0415

    rc = 0
    for man in sorted(args.queues.glob("*/manifest.json")):
        m = json.loads(man.read_text())
        folder = man.parent / "images"
        n = len(list(folder.glob("*.jpg")))
        have = datasets(args.api).get(m["name"])
        if have is None or count_of(have) >= n:
            continue
        tmp = f"{m['name']} [growing]"
        resp = api(
            args.api,
            "/api/dataset/import/server_folder",
            {"path": str(folder), "media_type": "image", "recursive": "false", "dig_archives": "false", "dataset_name": tmp},
            method="POST",
        )
        if "_error" in resp:
            print(f"  {m['name']}: IMPORT FAILED {resp['_error'][:100]}")
            rc = 1
            continue
        t0, landed = time.time(), None
        while time.time() - t0 < args.wait:
            time.sleep(3)
            d = datasets(args.api).get(tmp)
            if d and count_of(d) >= n:
                landed = d
                break
        if landed is None:
            print(f"  {m['name']}: TIMED OUT; old dataset kept, '{tmp}' left to inspect")
            rc = 1
            continue
        api(args.api, f"/api/datasets/registry/{have['id']}", method="DELETE")
        api(args.api, f"/api/datasets/registry/{landed['id']}/rename", {"name": m["name"]}, method="PUT")
        print(f"  {m['name']}: {count_of(have)} -> {count_of(landed)}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", type=Path, required=True)
    b.add_argument("--influence", type=Path, required=True)
    b.add_argument("--z", type=float, default=5.0, help="keep pairs with z < -Z")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=4179)
    b.add_argument("--min-per-class", type=int, default=5)
    b.add_argument("--control-share", type=float, default=0.1)
    b.add_argument("--min-controls", type=int, default=2)
    b.add_argument(
        "--extend",
        action="store_true",
        help="add to the queues already in --out: skip pairs already asked, render only new ones, append to manifests",
    )
    k = sub.add_parser("bank")
    k.add_argument("--api", required=True)
    k.add_argument("--queues", type=Path, required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--api", required=True)
    r.add_argument("--queues", type=Path, required=True)
    r.add_argument("--wait", type=int, default=900)
    args = ap.parse_args()
    return {"build": build, "bank": bank, "refresh": refresh}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
