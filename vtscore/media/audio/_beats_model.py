"""Vendored BEATs architecture + Kaldi-compatible fbank front-end.

BEATs ("Audio Pre-Training with Acoustic Tokenizers", Chen et al. 2022) has no
``transformers`` implementation, so - exactly as with
:mod:`vtscore.media.audio._paraspeechclap_model` - the architecture is
reconstructed here and the released checkpoint is overlaid onto it.  The
underscore-prefixed filename keeps this module out of the embedder
auto-discovery scan in :mod:`vtscore.media` (only ``embedder*.py`` files are
imported as plugins); :mod:`vtscore.media.audio.embedder_beats` imports from
here.

Two things are vendored:

**The encoder** (:class:`BEATs`) - a ported subset of ``microsoft/unilm``'s
``BEATs.py`` / ``backbone.py`` (MIT).  A 16x16 patch embedding over the
log-mel fbank feeds a 12-layer, 768-d Transformer that differs from a stock
Transformer in three ways, all of which the released weights depend on:

- **DeepNorm** residual scaling (``x = residual * alpha + f(x)`` with
  ``alpha = (2 * layers) ** 0.25``) and post-layer-norm ordering.
- **Gated relative position bias** - a T5-style bucketed bias, computed once
  in layer 0 and shared by every layer, gated per-layer by ``grep_linear`` /
  ``grep_a`` from the query.
- A grouped **convolutional position embedding** (``pos_conv``) applied before
  the stack, stored weight-normalised as ``weight_g`` / ``weight_v``.

Only the inference path is ported.  Training-only machinery (layer-drop,
gradient scaling, the masked-prediction head, the acoustic tokenizer) is
omitted, and the fine-tuned classifier variants are not supported - the
embedder loads the self-supervised checkpoint and mean-pools the encoder
output.

**The front-end** (:func:`kaldi_fbank`) - BEATs consumes Kaldi
``compute-fbank-feats`` features, which upstream obtains from
``torchaudio.compliance.kaldi.fbank``.  We do not depend on torchaudio (its
wheels are built against a pinned torch and would constrain ours), so the
fbank is ported here from torchaudio's own pure-PyTorch implementation
(BSD-2), specialised to the settings BEATs uses.  The port was checked against
torchaudio's output and agrees bit-for-bit at every signal length tried;
``tests_lib/detectors/test_beats_embedder.py`` pins values captured from that
verified run so later refactors cannot drift away from it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# ----------------------------------------------------------------------
# Kaldi-compatible log-mel filterbank
#
# Ported from ``torchaudio.compliance.kaldi`` (BSD-2), fixed to the options
# BEATs uses: 128 mel bins, 25 ms / 10 ms framing, Povey window, power
# spectrum, log output, snip_edges, no dither, no energy column.  The
# remaining Kaldi defaults are inlined as the constants below.
# ----------------------------------------------------------------------

_PREEMPHASIS = 0.97
_LOW_FREQ = 20.0


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def _mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def _mel_banks(num_bins: int, window_length_padded: int, sample_freq: float, low_freq: float) -> Tensor:
    """Kaldi triangular mel filterbank, shape ``(num_bins, padded // 2)``."""
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    high_freq = nyquist  # Kaldi's ``high_freq <= 0`` means "offset from Nyquist"

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _mel_scale_scalar(low_freq)
    mel_high_freq = _mel_scale_scalar(high_freq)
    # +1 because the end bins spread out past the edges.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin_idx = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin_idx * mel_freq_delta
    center_mel = mel_low_freq + (bin_idx + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_idx + 2.0) * mel_freq_delta

    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    # left < center < right, so the min of the two slopes clamped at 0 is the triangle.
    return torch.max(torch.zeros(1), torch.min(up_slope, down_slope))


def kaldi_fbank(
    waveform: Tensor,
    *,
    num_mel_bins: int = 128,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
) -> Tensor:
    """Log-mel filterbank matching Kaldi's ``compute-fbank-feats``.

    Args:
        waveform: 1-D tensor of samples, already scaled to Kaldi's int16-ish
            range (BEATs multiplies its ``[-1, 1]`` float audio by ``2 ** 15``).
        num_mel_bins: Number of triangular mel bins.
        sample_frequency: Sample rate of *waveform*, in Hz.
        frame_length: Window length in milliseconds.
        frame_shift: Hop between windows in milliseconds.

    Returns:
        Tensor of shape ``(num_frames, num_mel_bins)``.
    """
    device, dtype = waveform.device, waveform.dtype
    epsilon = torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)

    window_shift = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded_window_size = _next_power_of_2(window_size)

    # Frames that completely fit in the signal (Kaldi's snip_edges=True).
    num_samples = waveform.size(0)
    if num_samples < window_size:
        return torch.empty((0, num_mel_bins), dtype=dtype, device=device)
    m = 1 + (num_samples - window_size) // window_shift
    strides = (window_shift * waveform.stride(0), waveform.stride(0))
    strided = waveform.as_strided((m, window_size), strides)

    # remove_dc_offset: subtract each frame's own mean.
    strided = strided - torch.mean(strided, dim=1).unsqueeze(1)

    # Pre-emphasis, with the first sample replicated so frame[0] keeps its edge.
    offset = F.pad(strided.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    strided = strided - _PREEMPHASIS * offset[:, :-1]

    # Povey window: a Hann window raised to 0.85 so it reaches zero at the edges.
    window = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    strided = strided * window.unsqueeze(0)

    if padded_window_size != window_size:
        strided = F.pad(strided.unsqueeze(0), (0, padded_window_size - window_size), mode="constant", value=0).squeeze(
            0
        )

    spectrum = torch.fft.rfft(strided).abs().pow(2.0)

    banks = _mel_banks(num_mel_bins, padded_window_size, sample_frequency, _LOW_FREQ).to(device=device, dtype=dtype)
    # The Nyquist column is not covered by any triangle; pad it with zeros.
    banks = F.pad(banks, (0, 1), mode="constant", value=0)

    return torch.max(torch.mm(spectrum, banks.T), epsilon).log()


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------


class _SamePad(nn.Module):
    """Trim the trailing column an even-width 'same' convolution adds."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: Tensor) -> Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class _WeightNormConv1d(nn.Module):
    """``Conv1d`` with the weight stored weight-normalised over ``dim=2``.

    Upstream builds this with ``nn.utils.weight_norm(conv, dim=2)``, which is
    deprecated in current torch and whose replacement
    (``nn.utils.parametrizations.weight_norm``) renames the stored tensors to
    ``parametrizations.weight.original{0,1}``.  The released checkpoint holds
    the legacy ``weight_g`` / ``weight_v`` names, so the reparameterisation is
    done explicitly here: it pins the key names and is version-proof.
    """

    def __init__(self, channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.groups = groups
        self.weight_g = nn.Parameter(torch.ones(1, 1, kernel_size))
        self.weight_v = nn.Parameter(torch.zeros(channels, channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        # weight_norm over dim=2: normalise across every dim except the last.
        norm = self.weight_v.norm(2, dim=(0, 1), keepdim=True)
        weight = self.weight_v * (self.weight_g / norm)
        return F.conv1d(x, weight, self.bias, padding=self.padding, groups=self.groups)


class _MultiheadAttention(nn.Module):
    """Self-attention with a gated, bucketed relative position bias.

    One relative-position table is shared by the whole stack: only layer 0 ever
    computes the bias, and it threads the *raw* bias through to the layers
    above, each of which applies its own gate (``grep_linear`` / ``grep_a``) to
    it.  ``_TransformerEncoder`` ties the tables the way upstream does, which
    is why the released checkpoint carries twelve byte-identical copies.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        has_relative_attention_bias: bool,
        num_buckets: int,
        max_distance: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.grep_linear = nn.Linear(self.head_dim, 8)
        self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def _relative_positions_bucket(self, relative_positions: Tensor) -> Tensor:
        """T5-style bidirectional bucketing: exact when near, log-spaced when far."""
        num_buckets = self.num_buckets // 2
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        if_large = torch.min(if_large, torch.full_like(if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_positions, if_large)

    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        bucket = self._relative_positions_bucket(memory_position - context_position)
        bucket = bucket.to(self.relative_attention_bias.weight.device)
        return self.relative_attention_bias(bucket).permute([2, 0, 1])

    def forward(self, x: Tensor, position_bias: Optional[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
        """Args: *x* is ``(T, B, C)``; returns ``(output, raw position bias)``."""
        tgt_len, bsz, _ = x.size()

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_mask = None
        if position_bias is not None:
            # Gate the shared bias on this layer's queries.
            query_layer = x.transpose(0, 1).view(bsz, tgt_len, self.num_heads, -1).transpose(1, 2)
            _b, _h, _l, _ = query_layer.size()
            gate_a, gate_b = torch.sigmoid(
                self.grep_linear(query_layer).view(_b, _h, _l, 2, 4).sum(-1, keepdim=False)
            ).chunk(2, dim=-1)
            gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
            attn_mask = gate.view(bsz * self.num_heads, tgt_len, 1) * position_bias
            attn_mask = attn_mask.view((-1, tgt_len, tgt_len))

        out, _ = F.multi_head_attention_forward(
            x,
            x,
            x,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None,
            None,
            False,
            self.dropout_module.p,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
        return out, position_bias


class _EncoderLayer(nn.Module):
    """Post-LN Transformer layer with DeepNorm residual scaling."""

    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        has_relative_attention_bias: bool,
        num_buckets: int,
        max_distance: int,
        encoder_layers: int,
    ) -> None:
        super().__init__()
        self.self_attn = _MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)

    def forward(self, x: Tensor, pos_bias: Optional[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
        residual = x
        x, pos_bias = self.self_attn(x, position_bias=pos_bias)
        x = self.dropout1(x)
        x = residual * self.deep_norm_alpha + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x).float()).type_as(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual * self.deep_norm_alpha + x
        x = self.final_layer_norm(x)
        return x, pos_bias


class _TransformerEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        embed_dim = cfg["encoder_embed_dim"]
        layers = cfg["encoder_layers"]
        self.dropout = cfg["dropout"]
        self.pos_conv = nn.Sequential(
            _WeightNormConv1d(embed_dim, cfg["conv_pos"], cfg["conv_pos_groups"]),
            _SamePad(cfg["conv_pos"]),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=cfg["encoder_ffn_embed_dim"],
                    num_attention_heads=cfg["encoder_attention_heads"],
                    dropout=self.dropout,
                    attention_dropout=cfg["attention_dropout"],
                    activation_dropout=cfg["activation_dropout"],
                    has_relative_attention_bias=cfg["relative_position_embedding"],
                    num_buckets=cfg["num_buckets"],
                    max_distance=cfg["max_distance"],
                    encoder_layers=layers,
                )
                for _ in range(layers)
            ]
        )
        if cfg["relative_position_embedding"]:
            # Tie every layer's table to layer 0's, as upstream does. Only
            # layer 0 actually evaluates it (the bias is computed once and
            # passed up the stack), but the tie is what makes the checkpoint's
            # twelve identical copies load.
            for i in range(1, layers):
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        # layer_norm_first is False for every released BEATs config, so the
        # norm sits before the stack and there is none after it.
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        pos_bias = None
        for layer in self.layers:
            x, pos_bias = layer(x, pos_bias)
        return x.transpose(0, 1)


class BEATs(nn.Module):
    """The self-supervised BEATs encoder, from fbank patches to token features.

    Instantiate from the ``cfg`` dict stored alongside the weights in the
    released ``.pt`` file, then load ``checkpoint["model"]`` into it.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = cfg["embed_dim"]
        patch = cfg["input_patch_size"]
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=patch, stride=patch, bias=cfg["conv_bias"])
        self.layer_norm = nn.LayerNorm(self.embed)
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg["encoder_embed_dim"]) if self.embed != cfg["encoder_embed_dim"] else None
        )
        self.dropout_input = nn.Dropout(cfg["dropout_input"])
        self.encoder = _TransformerEncoder(cfg)

    def preprocess(self, waveform: Tensor, fbank_mean: float, fbank_std: float) -> Tensor:
        """Waveform in ``[-1, 1]`` -> normalised fbank of shape ``(T, 128)``."""
        fbank = kaldi_fbank(waveform * 2**15, num_mel_bins=128, sample_frequency=16000)
        return (fbank - fbank_mean) / (2 * fbank_std)

    def extract_features(self, waveform: Tensor, fbank_mean: float, fbank_std: float) -> Tensor:
        """Embed a single mono waveform into ``(1, num_tokens, encoder_embed_dim)``."""
        fbank = self.preprocess(waveform, fbank_mean, fbank_std).unsqueeze(0).unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        return self.encoder(self.dropout_input(features))
