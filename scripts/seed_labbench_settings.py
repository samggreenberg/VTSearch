"""Seed a LabBench image's data directory with its streamlined settings.

Used by ``docker/Dockerfile.labbench``, which runs it against ``/app/data``
before declaring that directory a volume. Docker copies the files into a fresh
named volume the first time it is mounted, so a new LabBench deployment comes
up streamlined with no first-run setup. They only land in a *fresh* volume;
the settings that must also hold on a redeploy onto an existing one are env
vars in the Dockerfile instead (the media-type lock and the dataset age-off).

Two files, for the media type the image was built for:

``settings.json`` (server tier)
    ``hidden_plugins`` drops plugins from every picker / listing API response:
    the Downloaded Demo Media importer in every LabBench image (it needs
    outbound network access a LabBench container is not expected to have),
    plus whatever the type's slim ``requirements/labbench-<type>.txt`` leaves
    without its dependencies (:data:`HIDDEN_BY_TYPE`). Hidden plugins remain
    importable by name; this is a UI declutter, not a security gate.

    ``dataset_max_age_days`` gives every dataset a two-week age-off, so the
    shared volume does not accumulate stale datasets. The Dockerfile pins the
    same value in ``VTSEARCH_DATASET_MAX_AGE_DAYS``, which overrides this file.

``user_settings.json`` (the single user of a container's
``DefaultLoginProvider``, whose data directory is the data directory itself)
    ``solo_embedder_per_media_type`` maps the media type to the embedder the
    image bakes, so the dataset importer hides its embedder dropdown and uses
    it silently; the slim requirements only cover that embedder. Users can
    still override it in Settings > Import Defaults. The embedder is resolved
    by the same registry call ``scripts/preload_default_embedder.py`` bakes
    with, so the lock is always the baked embedder and this file names no
    model.

Run on the host to see what an image would get:
    python scripts/seed_labbench_settings.py audio /tmp/labbench-data
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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
    "usage: seed_labbench_settings.py MEDIA_TYPE DATA_DIR\n"
    f"MEDIA_TYPE is one of {sorted(HIDDEN_BY_TYPE)}: docker/Dockerfile.labbench's MEDIA_TYPE build arg."
)


def server_settings(media_type: str) -> dict[str, Any]:
    """The seeded ``settings.json`` for a LabBench image of *media_type*."""
    hidden = {family: list(names) for family, names in HIDDEN_ALWAYS.items()}
    for family, names in HIDDEN_BY_TYPE[media_type].items():
        hidden.setdefault(family, []).extend(names)
    return {
        "dataset_max_age_days": DATASET_MAX_AGE_DAYS,
        "hidden_plugins": {family: sorted(set(names)) for family, names in sorted(hidden.items())},
    }


def user_settings(media_type: str, embedder: str | None) -> dict[str, Any]:
    """The seeded ``user_settings.json``: *media_type* locked to *embedder*."""
    return {"solo_embedder_per_media_type": {media_type: embedder} if embedder else {}}


def baked_embedder(media_type: str) -> str | None:
    """The embedder ``preload_default_embedder.py`` bakes for *media_type*, if any."""
    with tempfile.TemporaryDirectory(prefix="vtsearch-seed-data-") as scratch:
        # Before any vtscore import: vtscore.config.paths reads it at import
        # time, and the registry walk must not read or write the target dir.
        os.environ["VTSEARCH_DATA_DIR"] = scratch
        from vtscore.embedding import predict_embedders_to_preload

        wanted = predict_embedders_to_preload(extra_media_types=[media_type])
    return wanted[0] if wanted else None


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
    if len(argv) != 2 or not argv[0].strip() or not argv[1].strip():
        print(_USAGE, file=sys.stderr, flush=True)
        return 2
    media_type, data_dir = argv[0].strip(), Path(argv[1])
    if media_type not in HIDDEN_BY_TYPE:
        print(f"LabBench does not support media type {media_type!r}.\n{_USAGE}", file=sys.stderr, flush=True)
        return 2

    embedder = baked_embedder(media_type)
    server = server_settings(media_type)
    user = user_settings(media_type, embedder)
    _write_json(data_dir / "settings.json", server)
    _write_json(user_settings_path(data_dir), user)
    print(f"[seed] media_type={media_type} settings={server} user_settings={user}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
