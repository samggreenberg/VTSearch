"""``vg_box_{small,medium,large}``: box-size-banded Visual Genome.

One dataset per band of :data:`pile_config.BOX_BANDS`; the band's category
vocabulary is chosen by :func:`pilebuild.boxscan.band_categories` from the
full-VG scan, and every image carrying one of those categories is a candidate.
"""

from __future__ import annotations

import random

import pile_config as pc

from pilebuild.boxscan import band_categories
from pilebuild.env import cells_io, log
from pilebuild.vgsource import vg_boxes_by_name, vg_image_paths, vg_objects_json


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with full-VG images carrying this band's categories."""
    import json  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    band = pc.DATASETS[dataset]["band"]
    objects_json = vg_objects_json()
    if not objects_json.exists():
        raise SystemExit(f"missing {objects_json}")

    wanted = set(band_categories(band))
    paths = vg_image_paths()

    with objects_json.open() as fh:
        records = json.load(fh)

    # Every image carrying at least one of the band's categories.
    hits: list[tuple[int, dict]] = []
    for rec in records:
        iid = int(rec["image_id"])
        if iid not in paths:
            continue
        by_name = vg_boxes_by_name(rec, wanted)
        if by_name:
            hits.append((iid, by_name))

    rng = random.Random(0xB0FFED)  # deterministic sample, stable across rebuilds
    rng.shuffle(hits)
    hits = hits[: pc.BAND_MAX_IMAGES]
    log(f"  band {band}: {len(hits)} images carry a band category")

    for iid, by_name in hits:
        path = paths[iid]
        try:
            with Image.open(path) as im:
                W, H = im.size
            data = path.read_bytes()
        except Exception:  # noqa: BLE001 - a corrupt file just drops out
            continue
        if W <= 0 or H <= 0:
            continue
        regions = [
            {"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "label": name}
            for name, boxes in by_name.items()
            for b in boxes
        ]
        counts = {name: len(b) for name, b in by_name.items()}
        ordered = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        medias[iid] = {
            "id": iid,
            "media_type": "image",
            "embedder": embedder_name,
            "duration": 0,
            "file_size": 0,
            "md5": "",
            "embeddings": {},
            "media_bytes": data,
            "media_string": None,
            "filename": path.name,
            "category": ordered[0],
            "categories": ordered,
            "regions": regions,
            "origin": {"importer": "vg_box_band", "params": {"band": band, "embedder": embedder_name}},
            "origin_name": str(path),
        }


def check(dataset: str) -> str:
    """Really run the selection step, and ask whether it still selects *this*.

    ``--rebuildable`` on its own answers a weaker question than it looks like it
    answers: that selection *runs*, not that it selects the same thing. Those
    come apart in the direction that hurts, which is what :func:`_vocab_drift`
    is for.
    """
    chosen = band_categories(pc.DATASETS[dataset]["band"])
    drift = _vocab_drift(dataset, chosen)
    if drift:
        raise SystemExit(drift)
    return f"{len(chosen)} categories selected"


def _vocab_drift(dataset: str, chosen: list[str]) -> str:
    """Would rebuilding this band reproduce the cells that already exist?

    #3297's two candidate repairs both made the selector run again; only one of
    them kept picking the categories the published ``vg_box_*`` sets were built
    from, and taking the other would have silently redefined three datasets
    whose numbers are cited in #3129 and #3156 -- with the right media count,
    the right vectors, and nothing to look at that would say so.

    So where a cell is already built, compare its vocabulary against what the
    selector picks today. Reads the smallest present cell: every cell carries
    ``categories``, so there is no reason to page in the multi-GB patch one.
    Returns an empty string when they agree (or when nothing is built yet --
    a purged pile has nothing to reproduce, which is not a failure).
    """
    present = [(pc.cell_path(dataset, e).stat().st_size, e) for e in pc.EMBEDDERS if pc.cell_path(dataset, e).exists()]
    if not present:
        return ""
    _, emb = min(present)
    medias = cells_io().load_medias(pc.cell_path(dataset, emb))
    live = {c for m in medias.values() for c in (m.get("categories") or [])}
    gained = sorted(set(chosen) - live)
    lost = sorted(live - set(chosen))
    if not gained and not lost:
        return ""
    return (
        f"{dataset}: a rebuild would NOT reproduce the built cells -- selection now differs "
        f"from {dataset}__{emb}.pkl by {len(gained)} added and {len(lost)} dropped "
        f"categories (added {gained[:5]}, dropped {lost[:5]}). That is a dataset change, "
        f"not a rebuild; published numbers that cite this set would need re-examining."
    )
