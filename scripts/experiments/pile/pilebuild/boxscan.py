"""Reading the full-VG box scan, and choosing a band's categories from it."""

from __future__ import annotations

import json

import pile_config as pc

from pilebuild.env import log

#: The only per-category fields :func:`band_categories` reads. Named once so
#: the tolerance check below stays honest: if the selector ever starts reading
#: a field the pre-envelope scans do not carry, this list is what makes the old
#: shape fail loudly instead of selecting from partial statistics.
_SCAN_FIELDS = ("voted_area", "n_images", "union_inflation")


def load_box_scan_categories() -> dict[str, dict]:
    """Read the full-VG box scan, tolerating both eras of its file format.

    Two shapes exist in the wild, and the reader accepts either:

    * **Pre-2026-08-17** — the bare ``{category: stats}`` dict at top level.
      This is what the published ``vg_box_*`` sets were selected from.
    * **2026-08-17 and later** — ``fb4f4ec03`` wrapped that dict in
      ``{"meta": {...}, "categories": {...}}`` so per-band supply could be
      recorded alongside it.

    Tolerating both is deliberate rather than lazy. The envelope was the *only*
    incompatibility: the newer scan also carries ``bands``, ``bands_compact``,
    ``n_compact`` and ``compact_frac`` per category, and
    :func:`band_categories` reads none of them -- only ``voted_area``,
    ``n_images`` and ``union_inflation``, all three present in the old shape.
    So an old scan still selects exactly what it always selected, and the
    alternative repair (re-running ``scan_vg_boxes.py``) would *not*: the
    current scanner applies per-image compact filtering (``10239c24e``) and
    per-band supply (``fb4f4ec03``), which qualify categories differently and
    would silently redefine three datasets whose numbers are published in
    #3129 and #3156. Old scans also sit on other people's scratch, so the
    reader is the right place to absorb this rather than the file.
    """
    scan_path = pc.PILE / "vg_box_scale.json"
    if not scan_path.exists():
        raise SystemExit(f"missing {scan_path}; run scan_vg_boxes.py first")
    scan = json.loads(scan_path.read_text())
    if not isinstance(scan, dict) or not scan:
        raise SystemExit(f"{scan_path} is not a non-empty JSON object; re-run scan_vg_boxes.py")
    # Discriminate on shape, not just on the key being present: a bare scan is
    # keyed by VG's free-text vocabulary, so "categories" is a name it could in
    # principle hold. An envelope's "categories" maps to the stats dict; a
    # category of that name would map to a stats entry carrying `voted_area`.
    envelope = isinstance(scan.get("categories"), dict) and "voted_area" not in scan["categories"]
    stats = scan["categories"] if envelope else scan
    if not isinstance(stats, dict) or not stats:
        raise SystemExit(f"{scan_path} holds no categories; re-run scan_vg_boxes.py")
    # Fail here rather than with a bare KeyError deep inside the comprehension:
    # a scan missing a field the selector needs is a format problem, and saying
    # so by name is what tells the next person which era the file is from.
    missing = sorted({f for f in _SCAN_FIELDS for s in stats.values() if isinstance(s, dict) and f not in s})
    if missing:
        raise SystemExit(f"{scan_path} categories lack {', '.join(missing)}; re-run scan_vg_boxes.py")
    return stats


def band_categories(band: str) -> list[str]:
    """Pick this band's categories from the full-VG scan, stratified within it.

    Stratified on purpose: taking the N best-supported categories in a band
    would cluster them at one end of it (support correlates with size), so the
    "band" would silently be a point. Splitting the band into N slots by
    voted-area rank and taking the best-supported category in each keeps the
    band spanning its own range.
    """
    stats = load_box_scan_categories()
    lo, hi = pc.BOX_BANDS[band]

    pool = [
        (s["voted_area"], name)
        for name, s in stats.items()
        if lo <= s["voted_area"] < hi
        and s["n_images"] >= pc.BAND_MIN_IMAGES
        and s["union_inflation"] <= pc.BAND_MAX_INFLATION
        and pc.is_object_category(name)
    ]
    if not pool:
        raise SystemExit(f"no categories qualify for band {band!r}")
    pool.sort()

    if len(pool) < pc.BAND_N_CATEGORIES:
        # Say so rather than quietly returning a shorter list: a band that
        # cannot fill its quota is a real limit on what it can support.
        log(
            f"  band {band}: ONLY {len(pool)} categories qualify "
            f"(wanted {pc.BAND_N_CATEGORIES}) -- band is supply-limited"
        )
    n = min(pc.BAND_N_CATEGORIES, len(pool))
    chosen: list[str] = []
    for i in range(n):
        slot = pool[i * len(pool) // n : max((i + 1) * len(pool) // n, i * len(pool) // n + 1)]
        best = max(slot, key=lambda t: stats[t[1]]["n_images"])
        chosen.append(best[1])
    log(f"  band {band}: {len(chosen)} categories from {len(pool)} candidates")
    return sorted(set(chosen))
