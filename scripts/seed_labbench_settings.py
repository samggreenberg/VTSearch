"""Seed a LabBench image's data directory with its streamlined settings.

Used by ``docker/Dockerfile.labbench``, which runs it against ``/app/data``
before declaring that directory a volume. Docker copies the files into a fresh
named volume the first time it is mounted, so a new LabBench deployment comes
up streamlined with no first-run setup. They only land in a *fresh* volume;
the settings that must also hold on a redeploy onto an existing one are env
vars in the Dockerfile instead (the media-type lock and the dataset age-off).

Two files, for the media types the image was built for: one type, or every
LabBench type when the build is given no ``MEDIA_TYPE``.

``settings.json`` (server tier)
    ``hidden_plugins`` drops plugins from every picker / listing API response:
    the Downloaded Demo Media importer in every LabBench image (it needs
    outbound network access a LabBench container is not expected to have),
    plus whatever a type's slim ``requirements/labbench-<type>.txt`` leaves
    without its dependencies (:data:`HIDDEN_BY_TYPE`). An image of several
    types has no media-type lock, so it also hides the media types it does
    not install. Hidden plugins remain importable by name; this is a UI
    declutter, not a security gate.

    ``dataset_max_age_days`` gives every dataset a two-week age-off, so the
    shared volume does not accumulate stale datasets. The Dockerfile pins the
    same value in ``VTSEARCH_DATASET_MAX_AGE_DAYS``, which overrides this file.

``user_settings.json`` (the single user of a container's
``DefaultLoginProvider``, whose data directory is the data directory itself)
    ``solo_embedder_per_media_type`` maps each media type to the embedder the
    image bakes for it, so the dataset importer hides its embedder dropdown
    and uses it silently; the slim requirements only cover that embedder.
    Users can still override it in Settings > Import Defaults. The embedder is resolved
    by the same registry call ``scripts/preload_default_embedder.py`` bakes
    with, so the lock is always the baked embedder and this file names no
    model.

Run on the host to see what an image would get:
    python scripts/seed_labbench_settings.py audio /tmp/labbench-data
    python scripts/seed_labbench_settings.py audio image /tmp/labbench-data
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Plugins hidden in every LabBench image, whatever its media type.
HIDDEN_ALWAYS: dict[str, list[str]] = {"importers": ["demo"]}

#: The media types a LabBench image can be built for (one
#: ``requirements/labbench-<type>.txt`` each), and the plugins each one's slim
#: requirements leave without their dependencies, by plugin family.
HIDDEN_BY_TYPE: dict[str, dict[str, list[str]]] = {
    # The Aud2Img (spectrogram) converter needs librosa, which
    # labbench-image.txt leaves out.
    "image": {"converters": ["audio2image"]},
    # video2audio is the one converter whose output is audio, so the media-type
    # lock keeps offering it; its video input needs nothing the image lacks
    # (an ffmpeg binary, from imageio-ffmpeg).
    "audio": {},
}

#: The per-user settings file's name (``vtsearch.settings.USER_SETTINGS_FILENAME``,
#: which an app-tier test checks this against; the script stays import-clean
#: of the app so the library-tier tests can run it).
USER_SETTINGS_FILENAME = "user_settings.json"

#: The age-off every LabBench dataset gets. Keep equal to the Dockerfile's
#: ``VTSEARCH_DATASET_MAX_AGE_DAYS`` (a meta test checks they agree).
DATASET_MAX_AGE_DAYS = 14

_USAGE = (
    "usage: seed_labbench_settings.py MEDIA_TYPE [MEDIA_TYPE ...] DATA_DIR\n"
    f"Each MEDIA_TYPE is one of {sorted(HIDDEN_BY_TYPE)}: the types docker/Dockerfile.labbench "
    "installs (its MEDIA_TYPE build arg, or all of them when that is unset)."
)


def server_settings(*media_types: str) -> dict[str, Any]:
    """The seeded ``settings.json`` for a LabBench image of *media_types*.

    A plugin any one of the types hides stays hidden: a :data:`HIDDEN_BY_TYPE`
    entry says what that type's slim requirements cannot run, not what another
    type's would supply, so the union is the reading that never offers a plugin
    an entry found broken (at worst it hides one the combined requirements
    could run, as with Aud2Img, which hides itself from pickers anyway).

    An image of one type is locked to it (``VTSEARCH_SOLO_MEDIA_TYPE``), which
    already keeps the other media types out of its pickers. An image of several
    has no lock, so the media types it does not install are hidden here.
    """
    hidden = {family: list(names) for family, names in HIDDEN_ALWAYS.items()}
    for media_type in media_types:
        for family, names in HIDDEN_BY_TYPE[media_type].items():
            hidden.setdefault(family, []).extend(names)
    if len(media_types) > 1:
        from vtscore.media import all_type_ids

        hidden["media_types"] = [t for t in all_type_ids() if t not in media_types]
    return {
        "dataset_max_age_days": DATASET_MAX_AGE_DAYS,
        "hidden_plugins": {family: sorted(set(names)) for family, names in sorted(hidden.items())},
    }


def user_settings(embedders: Mapping[str, str | None]) -> dict[str, Any]:
    """The seeded ``user_settings.json``: each media type locked to its embedder in *embedders*."""
    return {"solo_embedder_per_media_type": {mt: emb for mt, emb in embedders.items() if emb}}


def baked_embedders(*media_types: str) -> dict[str, str | None]:
    """The embedder ``preload_default_embedder.py`` bakes for each of *media_types*, if any."""
    baked: dict[str, str | None] = {}
    with tempfile.TemporaryDirectory(prefix="vtsearch-seed-data-") as scratch:
        # Before any vtscore import: vtscore.config.paths reads it at import
        # time, and the registry walk must not read or write the target dir.
        os.environ["VTSEARCH_DATA_DIR"] = scratch
        from vtscore.embedding import predict_embedders_to_preload

        for media_type in media_types:
            wanted = predict_embedders_to_preload(extra_media_types=[media_type])
            baked[media_type] = wanted[0] if wanted else None
    return baked


def user_settings_path(data_dir: Path) -> Path:
    """Where the container's one user reads its settings under *data_dir*.

    Asked of the login provider rather than spelled here: the
    ``DefaultLoginProvider`` keeps its user's files in the data directory
    itself, not in ``data/default/`` (where this seed once went, and was never
    read).
    """
    from vtscore.security.login import DefaultLoginProvider

    provider = DefaultLoginProvider()
    username = provider.get_user(None)
    return provider.get_user_data_dir(username, data_dir) / USER_SETTINGS_FILENAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not all(arg.strip() for arg in argv):
        print(_USAGE, file=sys.stderr, flush=True)
        return 2
    media_types, data_dir = [arg.strip() for arg in argv[:-1]], Path(argv[-1])
    for media_type in media_types:
        if media_type not in HIDDEN_BY_TYPE:
            print(f"LabBench does not support media type {media_type!r}.\n{_USAGE}", file=sys.stderr, flush=True)
            return 2
    media_types = list(dict.fromkeys(media_types))

    # First: it is what points vtscore at a scratch data dir, which has to
    # happen before anything else here imports vtscore.
    user = user_settings(baked_embedders(*media_types))
    server = server_settings(*media_types)
    _write_json(data_dir / "settings.json", server)
    _write_json(user_settings_path(data_dir), user)
    print(f"[seed] media_types={media_types} settings={server} user_settings={user}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
