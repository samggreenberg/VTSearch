#!/usr/bin/env python3
"""Find every Visual Genome image containing a given noun and box each occurrence.

Given a ``noun``, look up which images contain that object (a metadata lookup
over the annotations — *not* a pixel search) and write a copy of each matching
image to ``--out-dir`` with a colored rectangle around every occurrence of the
noun (multiple boxes if it appears multiple times).

Reads the dataset staged at ``/exp/scale26/datasets/external/VisualGenome``:

* ``derived/objects_flat.jsonl.gz``       - rows {image_id, name, synset, x0..y1}
* ``annotations/image_data.json``         - per-image width/height
* ``images/images.zip`` + ``images2.zip`` - JPEGs read in-memory (never unpacked)

Matching is synset-canonical by default: ``--match synset`` maps "dog" to
``dog.n.01`` and so groups dog/dogs; rows with an empty synset fall back to an
exact (case-insensitive) name match.  ``--match name`` / ``--match substring``
override that.

Usage::

    python scripts/annotate_vg_noun.py giraffe --out-dir /tmp/vg_giraffe --limit 5
    python scripts/annotate_vg_noun.py "traffic light" --out-dir ./vg_lights

It is I/O-bound and single-threaded.  A rare noun with a small ``--limit`` runs
fine on a login node; for a common noun (many image reads + the full extract
scan) use a compute node::

    srun --partition=cpu --cpus-per-task=2 --mem=8G --time=0:30:00 \\
        /exp/mlucio/projects/VTSearch/.venv/bin/python \\
        scripts/annotate_vg_noun.py man --out-dir /exp/$USER/vg_man --limit 200
        
matthew Usage::
    cd /exp/mlucio/projects/VTSearch
    source .venv/bin/activate
    srun --partition=cpu --cpus-per-task=2 --mem=8G --time=0:30:00 python ./scripts/vg/annotate_vg_noun.py sheep --out-dir ./data/vg/vg_sheep_200_max --limit 200
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_VG_DIR = Path("/exp/scale26/datasets/external/VisualGenome")

# High-contrast palette; the noun's box color is chosen deterministically so a
# given noun always renders in the same color.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 215, 0),  # gold
    (40, 170, 200),  # cyan
    (220, 50, 50),  # red
    (60, 200, 90),  # green
    (160, 90, 220),  # purple
    (255, 140, 0),  # orange
    (0, 160, 160),  # teal
    (230, 80, 160),  # pink
    (110, 200, 40),  # lime
    (70, 120, 240),  # blue
    (200, 170, 60),  # mustard
    (250, 100, 100),  # salmon
    (140, 140, 140),  # grey
    (180, 220, 40),  # chartreuse
)


def _color_for(name: str) -> tuple[int, int, int]:
    """Stable palette color for a noun (same word -> same color)."""
    return _PALETTE[(hash(name) & 0x7FFFFFFF) % len(_PALETTE)]


def load_image_dims(image_data_path: Path) -> dict[int, tuple[int, int]]:
    """image_id -> (width, height) from image_data.json."""
    with image_data_path.open("rb") as fh:
        data = json.load(fh)
    dims: dict[int, tuple[int, int]] = {}
    for entry in data:
        iid = entry.get("image_id")
        w = entry.get("width")
        h = entry.get("height")
        if iid is not None and w and h:
            dims[int(iid)] = (int(w), int(h))
    return dims


def build_member_index(zip_paths: list[Path]) -> dict[int, tuple[Path, str]]:
    """image_id -> (zip_path, member) over all archives.

    Keyed on the integer stem of each ``.jpg`` member so it works regardless of
    whether an archive stores flat names (``107899.jpg``) or prefixed ones
    (``VG_100K_2/2377381.jpg``).  Uses ``namelist`` (central directory only).
    """
    index: dict[int, tuple[Path, str]] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if not member.endswith(".jpg"):
                    continue
                try:
                    index[int(Path(member).stem)] = (zp, member)
                except ValueError:
                    continue
    return index


def _row_matches(row: dict, mode: str, q_syn: str, q_name: str) -> bool:
    """Whether a flattened object row matches the query under ``mode``."""
    name = str(row.get("name", "")).strip().lower()
    if mode == "name":
        return name == q_name
    if mode == "substring":
        return q_name in name
    # synset-canonical: compare the synset lemma; fall back to name when the
    # row has no synset assigned (a small fraction of VG objects).
    syn = str(row.get("synset", "")).strip().lower()
    if syn:
        return syn.split(".")[0] == q_syn
    return name == q_name


def find_matches(extract_path: Path, mode: str, q_syn: str, q_name: str) -> dict[int, list[dict]]:
    """Stream the extract, returning image_id -> matching object rows.

    Cheap pre-filter: only ``json.loads`` lines that contain the query's first
    token as a substring (the real match runs after parsing, so this only skips
    lines that cannot possibly match).
    """
    tok = q_name.split()[0] if q_name else ""
    by_image: dict[int, list[dict]] = {}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if tok and tok not in line:
                continue
            row = json.loads(line)
            if _row_matches(row, mode, q_syn, q_name):
                by_image.setdefault(int(row["image_id"]), []).append(row)
    return by_image


def annotate_image(
    raw: bytes,
    dims: tuple[int, int],
    rows: list[dict],
    color: tuple[int, int, int],
) -> Image.Image:
    """Draw a rectangle for each matched object on the image bytes."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = dims
    draw = ImageDraw.Draw(img)
    for r in rows:
        box = (r["x0"] * w, r["y0"] * h, r["x1"] * w, r["y1"] * h)
        draw.rectangle(box, outline=color, width=3)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("noun", help="object/noun to search for (e.g. 'dog', 'traffic light')")
    ap.add_argument("--vg-dir", type=Path, default=DEFAULT_VG_DIR, help="staged Visual Genome dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write annotated images (default: vg_<noun>)")
    ap.add_argument(
        "--match",
        choices=("synset", "name", "substring"),
        default="synset",
        help="how the noun matches VG object names (default: synset-canonical)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="max images to annotate; <=0 means no cap (default: 200)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed when sampling down to --limit")
    ap.add_argument("--format", choices=("png", "jpg"), default="png", help="output image format")
    args = ap.parse_args()

    q_name = args.noun.strip().lower()
    q_syn = q_name.replace(" ", "_")
    if not q_name:
        raise SystemExit("noun must be non-empty")

    out_dir = args.out_dir or Path(f"vg_{re.sub(r'[^a-z0-9]+', '_', q_name).strip('_')}")

    image_data = args.vg_dir / "annotations" / "image_data.json"
    extract = args.vg_dir / "derived" / "objects_flat.jsonl.gz"
    zips = [args.vg_dir / "images" / "images.zip", args.vg_dir / "images" / "images2.zip"]
    for p in [image_data, extract, *zips]:
        if not p.exists():
            raise SystemExit(
                f"missing {p}\nExpected the staged Visual Genome dataset under --vg-dir "
                f"({args.vg_dir}); run fetch_visual_genome.sbatch first."
            )

    print(f"searching annotations for '{args.noun}' (match={args.match})…", flush=True)
    by_image = find_matches(extract, args.match, q_syn, q_name)
    if not by_image:
        print(f"no images contain '{args.noun}'", flush=True)
        return 0
    total = len(by_image)
    print(f"  {total:,} matching images", flush=True)

    selected = sorted(by_image)
    if 0 < args.limit < total:
        selected = sorted(random.Random(args.seed).sample(selected, args.limit))
        print(f"  capping to {args.limit} (--limit); raise --limit or pass <=0 for all", flush=True)

    dims = load_image_dims(image_data)
    members = build_member_index(zips)
    color = _color_for(q_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for iid in selected:
        if iid not in members or iid not in dims:
            skipped += 1
            print(f"  [{iid}] no image/dims — skipping", flush=True)
            continue
        zp, member = members[iid]
        try:
            with zipfile.ZipFile(zp) as zf:
                raw = zf.read(member)
            img = annotate_image(raw, dims[iid], by_image[iid], color)
        except Exception as exc:  # corrupt JPEG etc. — VG has a few
            skipped += 1
            print(f"  [{iid}] render failed ({exc}) — skipping", flush=True)
            continue
        out_path = out_dir / f"{iid}.{args.format}"
        img.save(out_path)
        written += 1
        print(f"  [{iid}] {len(by_image[iid])} box(es) -> {out_path.name}", flush=True)

    print(
        f"\n'{args.noun}': {total:,} matching images, wrote {written} to {out_dir}"
        + (f" ({skipped} skipped)" if skipped else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
