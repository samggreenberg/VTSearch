#!/usr/bin/env python
"""Per-person strata for a face review (#4762): who each FHIBE identity is, for "where the app does well and poorly".

A face review's categories are people, so its strata are the people's attributes. FHIBE annotates
each subject in each photo (``subject_annotation``); this reads every one-person photo of each
listed identity and keeps, per person:

* ``n_photos`` - one-person photos in the release, and ``face_band`` - the census's modal face-size
  band (#4709);
* ``age`` (median over the photos) and ``age_bracket``; ``pronoun``; ``skin_tone`` - the modal
  ``natural_skin_color`` index (0-5, FHIBE's own scale); ``ancestry`` - the continent of the modal
  first-listed ancestry region;
* ``label`` - ``"<age bracket>, <pronoun>, skin <tone>"``, the stratum the viewer shows in the
  person's place (``viewer.py --category-map``).

The output names people (subject IDs). It stays in the release's owner-only derived directory.

Usage (on a compute node: it reads several thousand JSONs)::

    python person_strata.py --identities <runs>/2026-10-09-k4/identities.txt --out <dir>/person_strata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

RELEASE = Path(os.environ.get("VTS_FHIBE_ROOT", f"/expscratch/{os.environ.get('USER', '')}/fhibe")) / (
    "fhibe-full-resolution-674a7dcf/fhibe.20260909.u.FCXqloBs_public_fullres/data/raw/fhibe_fullres"
)
CENSUS = Path(f"/expscratch/{os.environ.get('USER', '')}/fhibe/census-674a7dcf/annotations.csv")
AGE_BRACKETS = [(0, 29, "18-29"), (30, 44, "30-44"), (45, 59, "45-59"), (60, 200, "60+")]


def _bracket(age: float) -> str:
    return next((name for lo, hi, name in AGE_BRACKETS if lo <= age <= hi), "unknown")


def _mode(values: list) -> str:
    vals = [v for v in values if v not in (None, "", "nan")]
    return Counter(vals).most_common(1)[0][0] if vals else "unknown"


def _first(x) -> str:
    if isinstance(x, list):
        return str(x[0]) if x else ""
    return str(x) if x is not None else ""


def _strip_index(s: str) -> str:
    """FHIBE's enumerated values read ``"3. [175, 148, 120]"`` or ``"0. She/her/hers"``: the text after the index."""
    head, sep, tail = s.partition(". ")
    return tail if sep and head.strip().isdigit() else s


CONTINENTS = ("Africa", "Americas", "Asia", "Europe", "Oceania")


def _continent(region: str) -> str:
    """``"Southern Europe"`` -> ``"Europe"``: FHIBE lists ancestry at mixed granularity."""
    return next((c for c in CONTINENTS if c in region), region or "unknown")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--identities", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = {line.strip() for line in open(args.identities) if line.strip() and not line.startswith("#")}
    per: dict[str, dict[str, list]] = {sid: {"age": [], "pronoun": [], "skin": [], "ancestry": []} for sid in ids}
    with open(RELEASE / "filepaths.csv") as f:
        for row in csv.DictReader(f):
            sid = row["json"].split("/", 1)[0]
            if sid not in ids:
                continue
            d = json.loads((RELEASE / row["json"]).read_text())
            if d.get("image_annotation", {}).get("humans_per_image") != 1:
                continue
            for s in d.get("subject_annotation", []):
                if s.get("subject_id") != sid:
                    continue
                if s.get("age") is not None:
                    per[sid]["age"].append(float(s["age"]))
                per[sid]["pronoun"].append(_strip_index(_first(s.get("pronoun"))))
                per[sid]["skin"].append(str(s.get("natural_skin_color", "")).split(".")[0].strip())
                per[sid]["ancestry"].append(_strip_index(_first(s.get("ancestry"))))
    census = pd.read_csv(CENSUS)
    one = census[(census["humans"] == 1) & census["subject_id"].isin(ids)].groupby("subject_id")
    n_photos, band = one.size(), one["band"].agg(lambda b: b.value_counts().idxmax())
    rows = []
    for sid in sorted(ids):
        a = per[sid]
        age = float(pd.Series(a["age"]).median()) if a["age"] else float("nan")
        pronoun, skin = _mode(a["pronoun"]), _mode(a["skin"])
        rows.append(
            {
                "category": sid,
                "n_photos": int(n_photos.get(sid, 0)),
                "face_band": band.get(sid, "unknown"),
                "age": age,
                "age_bracket": _bracket(age) if age == age else "unknown",
                "pronoun": pronoun,
                "skin_tone": skin,
                "ancestry": _continent(_mode(a["ancestry"])),
                "label": f"{_bracket(age) if age == age else 'age unknown'}, {pronoun}, skin {skin}",
            }
        )
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    os.chmod(args.out, 0o600)
    print(f"{len(out)} people -> {args.out}")
    for col in ("age_bracket", "pronoun", "skin_tone", "ancestry", "face_band"):
        print(col, out[col].value_counts().to_dict())
    print("strata (labels):", out["label"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
