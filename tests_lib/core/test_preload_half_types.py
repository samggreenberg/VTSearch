"""Startup preload resolves a half media type to what its media embed as.

``predict_embedders_to_preload(extra_media_types=[solo])`` is what warms the
solo-mediaType server at boot, and what ``docker/Dockerfile.solo`` bakes into
the image (issue #4868). A half type names a pipeline rather than an
embedding space:

* ``document`` (convert-out) has no embedder; a document container means
  document input converted to page images, so the image default is warmed.
* ``face`` (convert-in) has its own embedder; a face container means image
  input with faces cropped out, so the face embedder is warmed.

Before this, ``document`` resolved to nothing: the server warmed no model and
the Docker build baked no weights.
"""

from __future__ import annotations

from vtscore.embedding import predict_embedders_to_preload
from vtscore.media import embedders_for_type
from vtscore.media import get as get_media_type


def test_convert_out_type_warms_its_default_targets_default() -> None:
    document = get_media_type("document")
    assert not embedders_for_type("document"), "document grew an embedder; this test's premise is gone"
    target = document.converts_to[0]
    assert predict_embedders_to_preload(extra_media_types=["document"]) == [embedders_for_type(target)[0].name]


def test_convert_in_type_warms_its_own_embedder() -> None:
    assert predict_embedders_to_preload(extra_media_types=["face"]) == [embedders_for_type("face")[0].name]


def test_full_types_are_unchanged() -> None:
    for media_type in ("audio", "image", "text", "video"):
        assert predict_embedders_to_preload(extra_media_types=[media_type]) == [embedders_for_type(media_type)[0].name]


def test_unknown_type_warms_nothing() -> None:
    assert predict_embedders_to_preload(extra_media_types=["no_such_type"]) == []
