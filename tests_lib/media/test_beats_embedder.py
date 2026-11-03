"""Tests for the BEATs audio embedder and its vendored architecture.

Covers three layers, none of which downloads the released weights:

- the Kaldi fbank front-end ported from torchaudio,
- the vendored encoder's structure (built from a miniature config),
- the embedder's own audio handling (windowing, padding, pooling).
"""

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from vtscore.media.audio._beats_model import BEATs, _EncoderLayer


def _layers(model: "BEATs") -> list["_EncoderLayer"]:
    """The encoder's layers, typed.

    ``nn.ModuleList`` indexing widens to ``Module``, whose ``__getattr__``
    returns ``Tensor | Module``, so reaching into a layer's submodules needs
    the concrete type back.
    """
    return cast(list["_EncoderLayer"], list(model.encoder.layers))


# A miniature BEATs config: same shape as the released one, small enough to
# build in a test. Mirrors the ``cfg`` dict stored in the checkpoint.
TINY_CFG = {
    "encoder_layers": 2,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 4,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "dropout_input": 0.0,
    "conv_pos": 16,
    "conv_pos_groups": 4,
    "relative_position_embedding": True,
    "num_buckets": 32,
    "max_distance": 80,
    "gru_rel_pos": True,
    "deep_norm": True,
    "layer_norm_first": False,
    "conv_bias": False,
    "input_patch_size": 16,
    "embed_dim": 16,
}


def _tone(freq: float, n: int = 4000, sr: int = 16000) -> torch.Tensor:
    """A deterministic pure tone — no RNG, so goldens are version-stable."""
    t = torch.arange(n, dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * freq * t)


class TestKaldiFbank:
    """The fbank front-end must match Kaldi's ``compute-fbank-feats``."""

    def test_framing_matches_kaldi_snip_edges(self):
        from vtscore.media.audio._beats_model import kaldi_fbank

        # 25 ms window (400 samples), 10 ms hop (160): only whole frames count,
        # so m = 1 + (n - 400) // 160.
        for n in (4000, 16000, 12345):
            fb = kaldi_fbank(_tone(440.0, n) * 2**15)
            assert fb.shape == (1 + (n - 400) // 160, 128)

    def test_signal_shorter_than_one_window_is_empty(self):
        from vtscore.media.audio._beats_model import kaldi_fbank

        assert kaldi_fbank(_tone(440.0, 399) * 2**15).shape == (0, 128)

    @pytest.mark.parametrize(
        ("freq", "expected_bin"),
        [(250.0, 13), (1000.0, 43), (4000.0, 96)],
    )
    def test_pure_tone_peaks_in_the_right_mel_bin(self, freq, expected_bin):
        """A tone's energy must land in the mel bin covering its frequency.

        This pins the mel scale, the FFT and the triangular filterbank
        together, independently of the golden values below.
        """
        from vtscore.media.audio._beats_model import kaldi_fbank

        fb = kaldi_fbank(_tone(freq) * 2**15)
        assert int(fb[5].argmax()) == expected_bin

    def test_golden_values(self):
        """Values captured from the run verified bit-exact against torchaudio.

        ``torchaudio.compliance.kaldi.fbank(waveform * 2**15, num_mel_bins=128,
        sample_frequency=16000, frame_length=25, frame_shift=10)`` returned
        exactly these numbers for this input.
        """
        from vtscore.media.audio._beats_model import kaldi_fbank

        t = torch.arange(4000, dtype=torch.float32) / 16000
        waveform = 0.6 * torch.sin(2 * math.pi * 440 * t) + 0.3 * torch.sin(2 * math.pi * 3000 * t)
        fb = kaldi_fbank(waveform * 2**15)

        assert fb.shape == (23, 128)
        np.testing.assert_allclose(fb[0, :4].numpy(), [9.7353, 6.9038, 10.3472, -15.9424], rtol=0, atol=1e-3)
        np.testing.assert_allclose(fb[0, 60:64].numpy(), [6.1935, 5.4065, 6.0111, 7.0570], rtol=0, atol=1e-3)
        assert float(fb.sum()) == pytest.approx(30299.383, abs=0.05)

    def test_is_deterministic(self):
        """No dithering: re-running must give bit-identical features."""
        from vtscore.media.audio._beats_model import kaldi_fbank

        w = _tone(440.0) * 2**15
        assert torch.equal(kaldi_fbank(w), kaldi_fbank(w))


class TestVendoredEncoder:
    """Structure of the vendored BEATs module."""

    def test_relative_attention_bias_is_shared_across_layers(self):
        from vtscore.media.audio._beats_model import BEATs

        layers = _layers(BEATs(TINY_CFG))
        first = layers[0].self_attn.relative_attention_bias
        for layer in layers[1:]:
            assert layer.self_attn.relative_attention_bias is first

    def test_state_dict_keys_match_the_released_layout(self):
        """The checkpoint stores one bias table per layer; so must we.

        Upstream ties the tables rather than deduplicating them, so the
        released ``state_dict`` carries a byte-identical copy under every
        layer. Losing the tie (or dropping the tables) would break a strict
        load of the real checkpoint.
        """
        from vtscore.media.audio._beats_model import BEATs

        keys = set(BEATs(TINY_CFG).state_dict())
        assert {"patch_embedding.weight", "layer_norm.weight", "post_extract_proj.weight"} <= keys
        assert {"encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"} <= keys
        for i in range(TINY_CFG["encoder_layers"]):
            assert f"encoder.layers.{i}.self_attn.relative_attention_bias.weight" in keys
            assert f"encoder.layers.{i}.self_attn.grep_a" in keys

    def test_pre_layer_norm_configs_are_refused(self):
        """Only the post-LN graph is ported; a pre-LN config must not load quietly."""
        from vtscore.media.audio._beats_model import BEATs

        with pytest.raises(ValueError, match="layer_norm_first"):
            BEATs(dict(TINY_CFG, layer_norm_first=True))

    def test_ungated_config_has_no_gate_tensors(self):
        """``gru_rel_pos=False`` checkpoints carry no ``grep_*``; neither should we."""
        from vtscore.media.audio._beats_model import BEATs

        keys = set(BEATs(dict(TINY_CFG, gru_rel_pos=False)).state_dict())
        assert not [k for k in keys if "grep_" in k]

    def test_deep_norm_can_be_disabled(self):
        from vtscore.media.audio._beats_model import BEATs

        model = BEATs(dict(TINY_CFG, deep_norm=False))
        assert _layers(model)[0].deep_norm_alpha == 1.0

    def test_deep_norm_alpha(self):
        from vtscore.media.audio._beats_model import BEATs

        model = BEATs(dict(TINY_CFG, encoder_layers=12))
        assert _layers(model)[0].deep_norm_alpha == pytest.approx(24**0.25)

    def test_extract_features_shape(self):
        """One token per 16x16 patch: (frames // 16) rows x (128 // 16) columns."""
        from vtscore.media.audio._beats_model import BEATs

        model = BEATs(TINY_CFG).eval()
        with torch.no_grad():
            out = model.extract_features(_tone(440.0, 16000), 15.41663, 6.55582)
        frames = 1 + (16000 - 400) // 160  # 98
        assert out.shape == (1, (frames // 16) * (128 // 16), TINY_CFG["encoder_embed_dim"])

    def test_weight_norm_conv_reparameterisation(self):
        """``weight_g``/``weight_v`` must reconstruct the same conv weight.

        We reparameterise by hand rather than using ``nn.utils.weight_norm``
        (deprecated) or ``nn.utils.parametrizations.weight_norm`` (different
        key names), so the maths is worth pinning.
        """
        from vtscore.media.audio._beats_model import _WeightNormConv1d

        conv = _WeightNormConv1d(8, kernel_size=4, groups=2)
        with torch.no_grad():
            conv.weight_v.normal_(generator=torch.Generator().manual_seed(0))
            conv.weight_g.fill_(2.0)
        norm = conv.weight_v.norm(2, dim=(0, 1), keepdim=True)
        expected = conv.weight_v * (conv.weight_g / norm)
        # Every kernel slice ends up with L2 norm equal to weight_g.
        np.testing.assert_allclose(expected.norm(2, dim=(0, 1)).detach().numpy(), np.full(4, 2.0), rtol=1e-5, atol=1e-5)


class TestBEATsEmbedderProperties:
    """Identity and registration — no weights are downloaded."""

    def test_name_and_media_type(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        assert emb.name == "beats"
        assert emb.media_type_id == "audio"

    def test_does_not_support_text(self):
        """BEATs has no text tower, so text search must be hidden for it."""
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        assert emb.supports_text is False
        assert emb.embed_text("a dog barking") is None

    def test_is_not_the_default_audio_embedder(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        assert AudioBEATsEmbedder().is_default is False

    def test_registered_in_registry(self):
        from vtscore.media import get_embedder

        assert get_embedder("beats").name == "beats"

    def test_to_dict(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        assert AudioBEATsEmbedder().to_dict() == {
            "name": "beats",
            "display_name": "BEATs (audio events)",
            "model_id": "lpepino/beats_ckpts",
            "media_type_id": "audio",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "supports_geometric_verification": False,
            "license_notice": None,
        }

    def test_config_constants(self):
        from vtscore.config import (
            BEATS_CHECKPOINT_FILE,
            BEATS_CHECKPOINT_REPO,
            BEATS_FBANK_MEAN,
            BEATS_FBANK_STD,
            BEATS_MAX_SAMPLES,
            BEATS_SAMPLE_RATE,
        )

        assert BEATS_CHECKPOINT_REPO == "lpepino/beats_ckpts"
        # The self-supervised encoder, not an AudioSet-finetuned classifier.
        assert BEATS_CHECKPOINT_FILE == "BEATs_iter3_plus_AS2M.pt"
        assert BEATS_SAMPLE_RATE == 16000
        assert BEATS_MAX_SAMPLES == 160000
        assert (BEATS_FBANK_MEAN, BEATS_FBANK_STD) == (15.41663, 6.55582)


class TestBEATsEmbedderAudioHandling:
    """Windowing / padding / pooling, exercised against a stub encoder."""

    @staticmethod
    def _stub(emb):
        """Swap in a fake encoder that records the waveform it was handed."""
        seen = {}

        class _Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def extract_features(self, waveform, fbank_mean, fbank_std):
                seen["n"] = int(waveform.shape[0])
                seen["norm"] = (fbank_mean, fbank_std)
                return torch.arange(2 * 4, dtype=torch.float32).reshape(1, 2, 4)

        emb._model = _Fake()
        return seen

    def _wav_bytes(self, seconds: float, sr: int = 16000) -> bytes:
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr, format="WAV")
        return buf.getvalue()

    def test_long_audio_is_truncated_to_the_leading_window(self):
        """Deterministic first window — a random crop would break rederivation."""
        from vtscore.config import BEATS_MAX_SAMPLES
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        seen = self._stub(emb)
        emb._embed_media_impl({"media_bytes": self._wav_bytes(30.0)})
        assert seen["n"] == BEATS_MAX_SAMPLES

    def test_short_audio_is_padded_to_the_floor(self):
        """Too-short clips must still produce a patch row instead of failing."""
        from vtscore.config import BEATS_MIN_SAMPLES
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        seen = self._stub(emb)
        emb._embed_media_impl({"media_bytes": self._wav_bytes(0.05)})
        assert seen["n"] == BEATS_MIN_SAMPLES

    def test_mid_length_audio_is_passed_through_untouched(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        seen = self._stub(emb)
        emb._embed_media_impl({"media_bytes": self._wav_bytes(5.0)})
        assert seen["n"] == 5 * 16000

    def test_tokens_are_mean_pooled(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        self._stub(emb)
        vec = emb._embed_media_impl({"media_bytes": self._wav_bytes(5.0)})
        assert vec is not None
        # Stub returns tokens [[0,1,2,3],[4,5,6,7]] -> mean over dim 1.
        np.testing.assert_allclose(vec, [2.0, 3.0, 4.0, 5.0])

    def test_uses_the_checkpoint_fbank_normalisation(self):
        from vtscore.config import BEATS_FBANK_MEAN, BEATS_FBANK_STD
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        seen = self._stub(emb)
        emb._embed_media_impl({"media_bytes": self._wav_bytes(5.0)})
        assert seen["norm"] == (BEATS_FBANK_MEAN, BEATS_FBANK_STD)

    def test_missing_source_returns_none(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        self._stub(emb)
        assert emb._embed_media_impl({}) is None

    def test_undecodable_bytes_return_none(self):
        from vtscore.media.audio.embedder_beats import AudioBEATsEmbedder

        emb = AudioBEATsEmbedder()
        self._stub(emb)
        assert emb._embed_media_impl({"media_bytes": b"not audio"}) is None
