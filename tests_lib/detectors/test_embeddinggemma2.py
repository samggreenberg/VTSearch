"""EmbeddingGemma 2 embedders (#4837): registration, the pooled forward, precision, audio windowing.

Offline: nothing here downloads the checkpoint or needs a ``transformers`` new
enough to hold ``EmbeddingGemma2Model``.  The forward pass is exercised against
a stand-in model and processor, which is enough to pin the parts this module
owns: the masked mean over the backbone's last hidden state, the retrieval
prefix on a query, the processor kwargs, and the audio window.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestRegistration:
    def test_one_checkpoint_one_entry_per_media_type(self):
        from vtscore.media import get_embedder

        img = get_embedder("embeddinggemma2")
        aud = get_embedder("embeddinggemma2_audio")
        assert (img.media_type_id, aud.media_type_id) == ("image", "audio")
        assert img.model_id == aud.model_id == "google/embeddinggemma-2"
        assert img.embedding_dim == aud.embedding_dim == 768

    def test_image_half_is_eval_only_audio_half_selectable(self):
        """The owner kept the image half out of the app as too slow (#4837); audio stays a choice."""
        from vtscore.media import all_embedders_dict, embedders_for_type, get_embedder

        image = embedders_for_type("image")
        audio = embedders_for_type("audio")
        assert get_embedder("embeddinggemma2").eval_only is True
        assert "embeddinggemma2" not in [e.name for e in image]
        assert "embeddinggemma2" not in {d["name"] for d in all_embedders_dict()}
        assert get_embedder("embeddinggemma2_audio").eval_only is False
        assert "embeddinggemma2_audio" in [e.name for e in audio]
        assert image[0].name == "siglip"
        assert audio[0].name == "clap_general"

    def test_text_side_and_no_prompt_ensemble(self):
        from vtscore.media import get_embedder

        for name in ("embeddinggemma2", "embeddinggemma2_audio"):
            emb = get_embedder(name)
            assert emb.supports_text is True
            assert emb.supports_patch_regions is False
            assert emb.description_wrappers == []

    def test_image_to_dict(self):
        from vtscore.media.image.embedders.embedder_embeddinggemma2 import ImageEmbeddingGemma2Embedder

        assert ImageEmbeddingGemma2Embedder().to_dict() == {
            "name": "embeddinggemma2",
            "display_name": "EmbeddingGemma 2 (images)",
            "model_id": "google/embeddinggemma-2",
            "media_type_id": "image",
            "is_default": False,
            "supports_text": True,
            "supports_patch_regions": False,
            "supports_geometric_verification": False,
            "license_notice": None,
        }

    def test_image_batch_defaults_small_but_env_wins(self, monkeypatch):
        from vtscore.media.image.embedders.embedder_embeddinggemma2 import ImageEmbeddingGemma2Embedder

        monkeypatch.delenv("VTSEARCH_EMBED_BATCH_SIZE", raising=False)
        assert ImageEmbeddingGemma2Embedder().embed_batch_size == 8
        monkeypatch.setenv("VTSEARCH_EMBED_BATCH_SIZE", "32")
        assert ImageEmbeddingGemma2Embedder().embed_batch_size == 32


class _FakeProcessor:
    """Records what ``apply_chat_template`` was given; returns a fixed padded batch."""

    def __init__(self, attention_mask: torch.Tensor):
        self.mask = attention_mask
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        n, t = self.mask.shape
        return {"input_ids": torch.zeros((n, t), dtype=torch.long), "attention_mask": self.mask}


class _FakeModel(torch.nn.Module):
    def __init__(self, hidden: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.hidden = hidden

    def forward(self, **_kwargs):
        return types.SimpleNamespace(last_hidden_state=self.hidden)


class TestPooledForward:
    def test_masked_mean_then_unit_length(self):
        from vtscore.media._embeddinggemma2 import pooled_forward

        hidden = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]],  # last token is padding
                [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        out = pooled_forward(_FakeModel(hidden), _FakeProcessor(mask), [[], []])
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)

    def test_pads_the_batch_and_forwards_processor_kwargs(self):
        from vtscore.media._embeddinggemma2 import pooled_forward

        proc = _FakeProcessor(torch.ones((1, 2), dtype=torch.long))
        pooled_forward(_FakeModel(torch.ones((1, 2, 3))), proc, [[]], sampling_rate=16000)
        call = proc.calls[0]
        assert call["tokenize"] is True and call["return_dict"] is True
        assert call["processor_kwargs"] == {"padding": True, "sampling_rate": 16000}

    def test_query_carries_the_search_prefix_and_media_none(self):
        from vtscore.media._embeddinggemma2 import media_messages, query_messages

        (q,) = query_messages(["a red bus"])
        assert q[0]["content"] == [{"type": "text", "text": "task: search result | query: a red bus"}]
        (m,) = media_messages("image", ["PIL"])
        assert m[0]["content"] == [{"type": "image", "image": "PIL"}]

    def test_image_text_query_goes_through_the_prefix(self):
        from vtscore.media.image.embedders.embedder_embeddinggemma2 import ImageEmbeddingGemma2Embedder

        emb = ImageEmbeddingGemma2Embedder()
        emb._model = _FakeModel(torch.ones((1, 2, 4)))
        emb._processor = _FakeProcessor(torch.ones((1, 2), dtype=torch.long))
        vec = emb.embed_text("a dog")
        assert vec is not None and vec.shape == (4,)
        text = emb._processor.calls[0]["messages"][0][0]["content"][0]["text"]
        assert text == "task: search result | query: a dog"


class TestPrecision:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("fp32", None),
            ("bf16", torch.bfloat16),
            ("autocast_bf16", torch.bfloat16),
            ("fp16", None),  # overflows fp16: the model card says NaN or silent degradation
            ("autocast_fp16", None),
        ],
    )
    def test_never_fp16(self, mode, expected):
        from vtscore.media import _embeddinggemma2 as eg2

        with patch.object(eg2, "embed_precision", return_value=mode):
            assert eg2._weight_dtype() is expected


class TestTransformersFloor:
    def test_old_transformers_names_the_fix(self, monkeypatch):
        from vtscore.media import _embeddinggemma2 as eg2

        # Stands in for a transformers that predates EmbeddingGemma2Model: a version and no class.
        monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.17.0"))
        with pytest.raises(ImportError, match=r"transformers>=5\.18\.0.*5\.17\.0"):
            eg2._require_transformers()


class TestAudioWindow:
    def _embed(self, n_samples: int) -> int:
        from vtscore.media.audio.embedder_embeddinggemma2_audio import AudioEmbeddingGemma2Embedder

        emb = AudioEmbeddingGemma2Embedder()
        emb._model = MagicMock()
        emb._processor = MagicMock()
        seen = {}

        def fake_forward(_model, _proc, messages, **kwargs):
            seen["len"] = len(messages[0][0]["content"][0]["audio"])
            seen["kwargs"] = kwargs
            return np.ones((1, 768), dtype=np.float32)

        wave = np.zeros(n_samples, dtype=np.float32)
        with (
            patch("vtscore.media.audio.decode.decode_audio", return_value=(wave, 16000)),
            patch("vtscore.media.audio.embedder_embeddinggemma2_audio.pooled_forward", side_effect=fake_forward),
        ):
            vec = emb.embed_media({"media_path": "/x.wav"})
        assert vec is not None and vec.shape == (768,)
        assert seen["kwargs"] == {"sampling_rate": 16000}
        return seen["len"]

    def test_long_clip_keeps_its_first_30_seconds(self):
        assert self._embed(16000 * 45) == 16000 * 30

    def test_short_clip_is_padded_to_a_second(self):
        assert self._embed(4000) == 16000

    def test_ordinary_clip_is_untouched(self):
        assert self._embed(16000 * 5) == 16000 * 5
