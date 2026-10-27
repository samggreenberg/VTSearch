"""Bake one media type's default embedder weights into ``$VTSEARCH_MODELS_DIR``.

Used by ``docker/Dockerfile.solo`` and ``docker/Dockerfile.labbench``, which
lock an image to a single media type (``VTSEARCH_SOLO_MEDIA_TYPE``). On its
first boot such a container preloads that type's default embedder
(``initialize_server`` in ``app.py``), so without this step every fresh
container would download the weights from Hugging Face before it could serve.

The build step runs the very call the server makes at startup,
:func:`vtscore.embedding.preload_predicted_embedders` with
``extra_media_types=[MEDIA_TYPE]``, so the image holds exactly what the
container would otherwise fetch: whichever embedder the registry calls that
type's default, through that embedder's own ``load_models()``. Nothing here
names a model, so a change of default (or of how an embedder loads) cannot
leave the image baking the wrong files.

That includes the half media types, which name a pipeline: ``document``
bakes the image default its pages embed with.  ``face`` is the exception: the
app downloads no face models (#5320), a deployment mounts its own, so a face
image built without them has nothing to bake, and Face turns on once they are
mounted (``docs/DEPLOYMENT.md``).

``preload_predicted_embedders`` logs a load failure and carries on, which is
right for a server and wrong for a build, so the result is checked here and a
missing embedder fails the build. A type the server would warm nothing for has
nothing to bake and exits 0.

The registry walk is pointed at an empty, throwaway data directory: the build
has no datasets or detectors to predict from, and this step must not write
anything into the image's ``/app/data``.

Run on the host to debug:
    VTSEARCH_MODELS_DIR=/tmp/m python -u scripts/preload_default_embedder.py audio
"""

from __future__ import annotations

import os
import sys
import tempfile

_USAGE = (
    "usage: preload_default_embedder.py MEDIA_TYPE\n"
    "In docker/Dockerfile.solo, pass it as a build arg, e.g.\n"
    "  docker build -f docker/Dockerfile.solo --build-arg MEDIA_TYPE=audio -t vtsearch:audio ."
)


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0].strip():
        print(f"No media type given.\n{_USAGE}", file=sys.stderr, flush=True)
        return 2
    media_type = argv[0].strip()

    if not os.environ.get("VTSEARCH_MODELS_DIR"):
        print("VTSEARCH_MODELS_DIR is not set", file=sys.stderr, flush=True)
        return 1

    with tempfile.TemporaryDirectory(prefix="vtsearch-preload-data-") as scratch:
        # Before any vtscore import: vtscore.config.paths reads it at import time.
        os.environ["VTSEARCH_DATA_DIR"] = scratch

        from vtscore.embedding import predict_embedders_to_preload, preload_predicted_embedders
        from vtscore.media import all_type_ids

        valid = sorted(all_type_ids())
        if media_type not in valid:
            print(f"Unknown media type {media_type!r}. Valid values: {valid}\n{_USAGE}", file=sys.stderr, flush=True)
            return 2

        wanted = predict_embedders_to_preload(extra_media_types=[media_type])
        if not wanted:
            print(f"[preload] no embedder is registered for {media_type!r}; nothing to bake.", flush=True)
            return 0

        from vtscore.media.face import imagekit_bridge

        if imagekit_bridge.EMBEDDER_NAME in wanted and not imagekit_bridge.face_models().available:
            wanted = [name for name in wanted if name != imagekit_bridge.EMBEDDER_NAME]
            print(
                "[preload] no face models to bake: the app downloads none, and Face turns on when "
                "a deployment mounts its own (docs/DEPLOYMENT.md).",
                flush=True,
            )
            if not wanted:
                return 0

        cache = os.environ["VTSEARCH_MODELS_DIR"]
        print(f"[preload] media_type={media_type} embedders={wanted} cache={cache}", flush=True)
        loaded = preload_predicted_embedders(extra_media_types=[media_type])

    missing = [name for name in wanted if name not in loaded]
    if missing:
        print(
            f"[preload] failed to bake {missing} for {media_type!r} (see the warning above).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"[preload] baked {loaded}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
