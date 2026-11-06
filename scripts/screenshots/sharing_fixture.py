#!/usr/bin/env python
"""Write the user guide's sharing example into a ``shared_folder`` folder.

    python scripts/screenshots/sharing_fixture.py < spec.json

``sharing-example.mjs`` builds the spec and runs this: the folder, the
drawings corpus the labels are on, and one entry per container (its
``remote_ref``, name, owner, recipients, public flag, pinned ``updated_at``,
Good and Bad file names, and the votes someone other than the owner changed).

The containers are other people's, which no request to the app can make
(the ``shared_folder`` source stamps every write with the caller), so this
writes them as the folder would hold them had their owners saved them:
records created and last modified by the owner, and the owner's manifest.
The foreign changes then go through the source's own test hook,
:func:`~vtscore.labels.sources.shared_folder.simulate_foreign_modification`,
which writes what a misbehaving recipient would. The times are pinned
afterwards, so the folder is the same on every run.

Each label is the picture's MD5 and file name, which is how a received copy
finds its pictures in a loaded dataset. Idempotent: each container is
rewritten whole, and nothing else in the folder is touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtscore.datasets.labelset import LabeledElement, LabelSet  # noqa: E402
from vtscore.labels.sharing import build_manifest  # noqa: E402
from vtscore.labels.sources.shared_folder import FORMAT, simulate_foreign_modification  # noqa: E402
from vtscore.utils.io import atomic_write_json  # noqa: E402

#: The ``detector_meta`` every example container carries: drawings are images.
DETECTOR_META = {"media_type": "image"}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _write_container(folder: Path, corpus: Path, spec: dict[str, Any]) -> None:
    md5s = {name: _md5(corpus / name) for name in [*spec["good"], *spec["bad"]]}
    elements = [
        LabeledElement(md5=md5s[name], label=label, filename=name) for label in ("good", "bad") for name in spec[label]
    ]
    labelset = LabelSet(elements, detector_meta=DETECTOR_META)
    owner, at = spec["owner"], float(spec["updated_at"])
    path = folder / f"{spec['ref']}.json"
    atomic_write_json(
        path,
        {
            "format": FORMAT,
            "remote_ref": spec["ref"],
            "name": spec["name"],
            "owner": owner,
            "detector_meta": DETECTOR_META,
            "public": bool(spec["public"]),
            "recipients": list(spec["recipients"]),
            "archived": False,
            "created_at": at,
            "updated_at": at,
            "records": [
                {"element": el.to_dict(), "created_by": owner, "last_modified_by": owner, "modified_at": at}
                for el in elements
            ],
            "manifest": build_manifest(labelset).to_dict(),
            "manifest_modified_by": owner,
        },
    )
    foreign = spec.get("foreign")
    if foreign:
        relabel = [md5s[name] for name in foreign["relabel"]]
        simulate_foreign_modification(folder, spec["ref"], actor=foreign["actor"], relabel=relabel)
        # The hook stamps the time it ran; pin it like everything else.
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["updated_at"] = at
        for record in doc["records"]:
            record["modified_at"] = at
        atomic_write_json(path, doc)


def main() -> None:
    spec = json.load(sys.stdin)
    folder, corpus = Path(spec["folder"]), Path(spec["corpus"])
    folder.mkdir(parents=True, exist_ok=True)
    for container in spec["containers"]:
        _write_container(folder, corpus, container)


if __name__ == "__main__":
    main()
