"""``coco_val``: not a demo dataset, so assembled from the staged zip + annotations.

The zip and the annotation file are named through :mod:`pile_config` and read
here; :func:`check` asserts *those same* two paths. Two names for one source is
what let the rebuild canary report this dataset BROKEN against a staging area
that was present and intact (#3299), so the identity is the point.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pile_config as pc

from pilebuild.env import log


def _coco_annotations() -> tuple[dict[int, list[dict]], dict[int, str]]:
    """``({image_id: [{box, label}, ...]}, {image_id: file_name})``.

    Mirrors ``calibration/build_coco_pickle.py``: boxes already normalised to
    [0, 1], ``iscrowd`` regions kept (they are still true instances of the
    category, and positives are defined by category presence).
    """
    regions: dict[int, list[dict]] = defaultdict(list)
    filenames: dict[int, str] = {}
    with gzip.open(pc.COCO_ANNOTATIONS, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            image_id = int(row["image_id"])
            filenames[image_id] = row["file_name"]
            regions[image_id].append(
                {
                    "box": [float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])],
                    "label": row["name"],
                }
            )
    return dict(regions), filenames


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with COCO-val images (bytes read straight from the zip).

    Only images the annotation file covers are kept: an image with no category
    can be neither a positive nor a meaningful negative.
    """
    if not pc.COCO_ANNOTATIONS.exists():
        raise SystemExit(f"missing COCO annotations: {pc.COCO_ANNOTATIONS}")
    zip_path = pc.COCO_VAL_ZIP
    if not zip_path.exists():
        raise SystemExit(f"missing COCO images zip: {zip_path}")

    regions_by_image, filenames = _coco_annotations()
    log(f"  coco: {len(regions_by_image)} annotated images")

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}
        missing = 0
        for image_id in sorted(regions_by_image):
            regions = regions_by_image[image_id]
            fname = Path(filenames[image_id]).name
            member = members.get(fname)
            if member is None:
                missing += 1
                continue
            counts: dict[str, int] = defaultdict(int)
            for r in regions:
                counts[r["label"]] += 1
            # Most-annotated first, so ``category`` is the dominant object.
            ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            medias[image_id] = {
                "id": image_id,
                "media_type": "image",
                "embedder": embedder_name,
                "duration": 0,
                "file_size": 0,
                "md5": "",
                "embeddings": {},
                "media_bytes": zf.read(member),
                "media_string": None,
                "filename": fname,
                "category": ordered[0],
                "categories": ordered,
                "regions": regions,
                "origin": {"importer": "staged_coco_val", "params": {"embedder": embedder_name}},
                "origin_name": filenames[image_id],
            }
    if missing:
        log(f"  coco: WARNING {missing} annotated images absent from the zip")


def check(dataset: str) -> str:
    """What a ``coco_val`` rebuild reads. Raises ``SystemExit`` if it is absent.

    The *zip*, which is what :func:`load` opens. Checking the extracted
    directory instead reported this dataset broken against sources that were
    entirely intact (#3299) -- a canary that names a different path than the
    build is not a canary.
    """
    for path in (pc.COCO_ANNOTATIONS, pc.COCO_VAL_ZIP):
        if not path.exists():
            raise SystemExit(f"{dataset}: missing {path}")
    return "annotations + image zip present"
