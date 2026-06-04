"""Vendored ParaSpeechCLAP model definition (dual-encoder speech↔text CLAP).

ParaSpeechCLAP (`github.com/ajd12342/paraspeechclap`, MIT-licensed) maps speech
and rich textual *style* descriptions ("a deep, raspy voice", "a whispered,
anxious style") into a shared 768-dim space.  It pairs a WavLM-Large speech
encoder with a Granite text encoder via two small residual projection heads.

The upstream ``paraspeechclap`` package is not on PyPI and pulls in
training-only deps (hydra, wandb, audtorch).  Rather than vendor the whole
package, we reconstruct just the *inference* architecture here so VTSearch can
load the released ``.pth.tar`` checkpoints with the dependencies it already
has (torch + transformers).  Attribute names mirror upstream exactly so the
checkpoint ``state_dict`` keys line up on ``load_state_dict``.

This module imports torch/transformers at module scope, so it must only ever
be imported lazily (from inside ``embedder_paraspeechclap._load_models_impl``),
never at plugin-discovery time.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# A loader hook lets the embedder inject ``load_pretrained_local_first`` so the
# base-encoder downloads honour VTSearch's offline-first / retry policy.  The
# default just calls the function directly, keeping this module usable stand-alone.
Loader = Callable[..., Any]


def _default_loader(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)


class Projection(torch.nn.Module):
    """Residual MLP head projecting an encoder output into the shared space."""

    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear2 = torch.nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_out)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        return self.layer_norm(embed1 + embed2)


class SpeechEncoder(torch.nn.Module):
    """Speech encoder (WavLM-Large) mean-pooled over the time axis."""

    def __init__(self, model_name: str, loader: Loader = _default_loader) -> None:
        super().__init__()
        self.model_name = model_name
        self.is_wavlm = "wavlm" in model_name.lower()
        config = loader(AutoConfig.from_pretrained, model_name)
        # LayerDrop is a training-time regulariser; disable it so inference is
        # deterministic (and so the module graph matches the checkpoint).
        config.layerdrop = 0.0
        self.base = loader(AutoModel.from_pretrained, model_name, config=config)
        self.hidden_size = self.base.config.hidden_size

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.is_wavlm:
            # return_dict=False skips the (unused) extract_features tensor.
            last_hidden_state = self.base(x, attention_mask=attention_mask, return_dict=False)[0]
        else:
            last_hidden_state = self.base(x, attention_mask=attention_mask).last_hidden_state
        return torch.mean(last_hidden_state, dim=1)


class TextEncoder(torch.nn.Module):
    """Text encoder (Granite) using the CLS token of the last hidden state."""

    def __init__(self, model_name: str, loader: Loader = _default_loader) -> None:
        super().__init__()
        self.base = loader(AutoModel.from_pretrained, model_name)

    def forward(self, x: dict) -> torch.Tensor:
        return self.base(**x).last_hidden_state[:, 0, :]


class CLAP(torch.nn.Module):
    """Dual-encoder CLAP: speech + text projected into a shared embedding space.

    Only the inference getters (:meth:`get_audio_embedding`,
    :meth:`get_text_embedding`) are kept from upstream; the contrastive loss
    and logit-scale gradient path are training-only and omitted.  The
    ``log_logit_scale`` parameter is still declared so the checkpoint loads
    with ``strict=True``.
    """

    def __init__(
        self,
        speech_name: str,
        text_name: str,
        embedding_dim: int = 768,
        projection_dropout: float = 0.5,
        loader: Loader = _default_loader,
    ) -> None:
        super().__init__()
        self.audio_branch = SpeechEncoder(speech_name, loader=loader)
        self.text_branch = TextEncoder(text_name, loader=loader)
        self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim, p=projection_dropout)
        self.text_projection = Projection(self.text_branch.base.config.hidden_size, embedding_dim, p=projection_dropout)
        self.log_logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def get_audio_embedding(
        self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None, normalize: bool = True
    ) -> torch.Tensor:
        emb = self.audio_projection(self.audio_branch(audio, attention_mask=attention_mask))
        return F.normalize(emb, dim=-1) if normalize else emb

    def get_text_embedding(self, text_input: dict, normalize: bool = True) -> torch.Tensor:
        emb = self.text_projection(self.text_branch(text_input))
        return F.normalize(emb, dim=-1) if normalize else emb
