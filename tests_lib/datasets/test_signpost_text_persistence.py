"""Library-tier tests for signpost texts surviving the pickle round-trip.

The signpost text is the sign pipeline's only full-corpus model cost, and the
ingest stage computes it *before* the registry save specifically so it rides
along in the dataset container.  ``export_dataset_to_file`` writes an explicit
field list, so the three fields the text layer owns
(``signpost_texts.PERSISTED_FIELDS``) have to be named there — and restored on
load — or every reload silently re-runs the caption/tag models and drops the
item's "AI Caption" / "AI Tags" metadata row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.loader_pickle import load_dataset_from_pickle
from vtscore.media.audio.media_type import AudioMediaType
from vtscore.projection import signpost_texts as st


def _media(**extra) -> dict:
    rng = np.random.default_rng(11)
    return {
        "id": 1,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": 64,
        "md5": "abc",
        "embedder": "clap",
        "embeddings": {"clap": rng.standard_normal(16).astype(np.float32)},
        "filename": "a.wav",
        "category": "test",
        "media_bytes": b"\x00" * 64,
        **extra,
    }


def _round_trip(media: dict, tmp_path: Path) -> dict:
    container = export_dataset_to_file({1: media}, embedder="clap", media_type="audio")
    pkl = tmp_path / "ds.pkl"
    pkl.write_bytes(container)
    loaded: dict = {}
    load_dataset_from_pickle(pkl, loaded)
    return loaded[1]


class TestSignpostTextSurvivesPickleRoundTrip:
    def test_text_source_and_kind_persist(self, tmp_path: Path):
        media = _media(
            **{
                st.TEXT_FIELD: "Rain, Thunderstorm, Wind",
                st.SOURCE_FIELD: "tags:audioset527:clap",
                st.KIND_FIELD: st.KIND_TAGS,
            }
        )
        out = _round_trip(media, tmp_path)

        assert out[st.TEXT_FIELD] == "Rain, Thunderstorm, Wind"
        assert out[st.SOURCE_FIELD] == "tags:audioset527:clap"
        assert out[st.KIND_FIELD] == st.KIND_TAGS

    def test_metadata_row_survives_the_round_trip(self, tmp_path: Path):
        media = _media(
            **{
                st.TEXT_FIELD: "birds chirping in a forest",
                st.SOURCE_FIELD: "caption:whisper-audio",
                st.KIND_FIELD: st.KIND_CAPTION,
            }
        )
        out = _round_trip(media, tmp_path)

        assert AudioMediaType().display_metadata(out)["AI Caption"] == "birds chirping in a forest"

    def test_unprepped_media_gains_no_fields(self, tmp_path: Path):
        out = _round_trip(_media(), tmp_path)
        assert not [field for field in st.PERSISTED_FIELDS if field in out]
