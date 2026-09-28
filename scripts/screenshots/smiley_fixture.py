#!/usr/bin/env python
"""The user guide's worked example, drawn: the Smiley example's corpora.

    python scripts/screenshots/smiley_fixture.py drawings        # -> prints the path
    python scripts/screenshots/smiley_fixture.py drawings-new
    python scripts/screenshots/smiley_fixture.py drawing-regions

Every screenshot in the user guide is taken against these folders of cartoon
drawings (see `smiley-example.mjs`). Each is drawn by the Synthetic Media
importer's own generator, `vtscore.utils.synthetic.generate_image_dataset`,
with the importer's own `size` and `seed` (`CORPORA`), so a reader who loads
**Demo -> Synthetic Media** with those settings gets the very same pictures.

A corpus goes into `data/doc-fixtures/<name>/`, and beside it `<name>.json`:
`describe_image_dataset`'s account of every picture (its kind, background,
and the colour, expression and box of everything drawn on it), each with the
example's `category` for it and the boxes of its `yellow_smileys`, under
`pictures`, and a `fingerprint`. That is how the harness knows which drawings
are the yellow smileys - the ground truth a folder of photographs carried in
its subfolder names - and `category` is defined here, once, for the harness
and for `tests_lib/meta/test_smiley_example.py` alike.

Idempotent: the fingerprint names the generator's source and the corpus's
size and seed, and a corpus whose fingerprint still matches is not redrawn. A
corpus drawn by an older generator is deleted and drawn again, so a changed
generator can never leave last version's pictures in the folder. The account
is rewritten every run (it costs no drawing). Nothing is written into the repo.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtscore.utils.synthetic import images as generator  # noqa: E402

FIXTURES = _REPO_ROOT / "data" / "doc-fixtures"

#: The corpora, as `{name: (size, seed)}` - the Synthetic Media importer's own
#: fields, so each is exactly what that importer makes with those settings.
#:
#: `drawings` is the pile the guide's detector is trained on, and
#: `drawings-new` the one it is then run over: another seed, so the two share
#: no picture and Find ranks drawings nobody voted on. `drawing-regions` is
#: small because it is embedded with DINOv2 patch, many times the work per
#: picture, for the one region-voting shot.
CORPORA: dict[str, tuple[int, int]] = {
    "drawings": (240, 1),
    "drawings-new": (240, 2),
    "drawing-regions": (40, 3),
}


def is_yellow_smiley(obj: dict) -> bool:
    """Whether one drawn object is what the example looks for: a yellow face, smiling."""
    return obj["shape"] == "face" and obj["color"] == "yellow" and obj["smiling"]


def category(picture: dict) -> str:
    """The example's category of *picture*, from the generator's account of it.

    ``yellow-smiley`` is what the guide's detector is trained to find: a picture
    of one yellow face, smiling (a smile, a grin or a wink). The rest are named
    for the near-miss they are: ``yellow-face`` (yellow, not smiling),
    ``orange-smiley`` (the nearest colour, smiling), ``smiley`` (another colour,
    smiling), ``face`` (neither), ``yellow-shapes`` (a yellow shape, no face),
    ``shapes``, and ``scene-yellow-smiley`` / ``scene`` (several things, with or
    without a yellow smiley among them).
    """
    objects = picture["objects"]
    if picture["kind"] == "face":
        face = objects[0]
        if face["color"] == "yellow":
            return "yellow-smiley" if face["smiling"] else "yellow-face"
        if face["smiling"]:
            return "orange-smiley" if face["color"] == "orange" else "smiley"
        return "face"
    if picture["kind"] == "shapes":
        return "yellow-shapes" if any(o["color"] == "yellow" for o in objects) else "shapes"
    return "scene-yellow-smiley" if any(is_yellow_smiley(o) for o in objects) else "scene"


def pictures(name: str) -> list[dict]:
    """Corpus *name*'s pictures, as the generator describes them plus the example's reading."""
    size, seed = CORPORA[name]
    return [
        {
            **picture,
            "category": category(picture),
            "yellow_smileys": [o["box"] for o in picture["objects"] if is_yellow_smiley(o)],
        }
        for picture in generator.describe_image_dataset(size, seed=seed)
    ]


def fingerprint(name: str) -> str:
    """What decides corpus *name*'s pictures: the generator's source, its size, its seed."""
    size, seed = CORPORA[name]
    digest = hashlib.sha256(Path(generator.__file__).read_bytes())
    digest.update(f"{size}:{seed}".encode())
    return digest.hexdigest()[:16]


def ensure_corpus(name: str) -> Path:
    """Draw corpus *name* into `data/doc-fixtures/<name>/` unless it is current."""
    size, seed = CORPORA[name]
    folder = FIXTURES / name
    sidecar = FIXTURES / f"{name}.json"
    stamp = fingerprint(name)
    drawn = sidecar.exists() and folder.is_dir() and json.loads(sidecar.read_text()).get("fingerprint") == stamp
    if not drawn:
        if folder.exists():
            shutil.rmtree(folder)
        print(f"drawing {name} ({size} pictures, seed {seed}) ...", file=sys.stderr)
        generator.generate_image_dataset(folder, size, seed=seed)
    account = {"fingerprint": stamp, "size": size, "seed": seed, "pictures": pictures(name)}
    sidecar.write_text(json.dumps(account))
    return folder


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in CORPORA:
        print(f"usage: {argv[0]} {{{','.join(CORPORA)}}}", file=sys.stderr)
        return 2
    print(ensure_corpus(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
