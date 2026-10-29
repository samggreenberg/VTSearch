"""Torch tensor/device helpers shared by every :class:`MediaEmbedder`.

Small, stateless adapters that sit between a ``transformers``-style model and
the fp32-vector contract the rest of VTSearch is written against: pull a plain
tensor out of whatever dataclass a model returned, move a freshly loaded model
onto the resolved compute device, follow it with the processor's inputs, and
upcast the result back to fp32 before it leaves the GPU.

Split out of :mod:`vtscore.media.embedder` so the ABC that 20-odd embedder
modules subclass is not read through a wall of unrelated helpers; every name
here is still re-exported from that module for third-party embedders.
"""

from __future__ import annotations

import contextlib
from typing import Any

__all__ = [
    "embed_autocast",
    "extract_tensor",
    "to_compute_device",
    "to_float32",
    "to_model_inputs",
]


def extract_tensor(output: object):
    """Extract a plain tensor from a model output.

    Depending on the ``transformers`` version, methods like
    ``get_image_features()`` / ``get_text_features()`` /
    ``get_video_features()`` may return either a raw :class:`torch.Tensor`
    or a ``BaseModelOutputWithPooling`` dataclass.  This helper handles
    both cases transparently.
    """
    import torch  # noqa: PLC0415

    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "video_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


def to_compute_device(model: Any, allow_half: bool = False) -> Any:
    """Move a freshly loaded embedding *model* onto the resolved compute device.

    Replaces the hardcoded ``model.to("cpu")`` every embedder used to run at
    load time.  The target device comes from
    :func:`vtscore.config.resolve_device`, which honours ``VTSEARCH_DEVICE`` and
    smoke-tests CUDA before returning a CUDA device - falling back to ``"cpu"``
    when the installed torch wheel can't actually launch a kernel on the visible
    GPU.  The move is therefore always safe:

    * On a CPU-only host (or one whose GPU the wheel can't drive) this is exactly
      the old ``.to("cpu")``, still materialising any ``meta``-device tensors
      left behind by ``low_cpu_mem_usage=True``.
    * On a working CUDA / MPS host the model lands on the accelerator, and every
      embedder's forward pass follows it automatically: each reads
      ``next(self._model.parameters()).device`` and copies its inputs there,
      pulling results back with ``.detach().cpu().numpy()``.

    Returns the moved model so callers can write ``self._model = to_compute_device(self._model)``.

    Pass ``allow_half=True`` to additionally honour ``VTSEARCH_EMBED_PRECISION``
    by casting the weights (see :func:`vtscore.config.embed_weight_dtype`).  It
    is opt-in per embedder rather than global because the precision measurement
    behind it (#3143) covers the **image** encoders only: the audio, video and
    face backbones share this helper, and casting a model whose numerics nobody
    has measured would be a silent change to what it produces.  Half precision
    is off by default in any case, so an embedder that has not opted in is
    unaffected either way.
    """
    from vtscore.config import embed_weight_dtype, resolve_device  # noqa: PLC0415

    model = model.to(resolve_device())
    if allow_half:
        dtype = embed_weight_dtype()
        if dtype is not None:
            model = model.to(dtype)
    return model


@contextlib.contextmanager
def embed_autocast() -> Any:
    """Wrap an embedding forward in ``torch.autocast`` when so configured.

    A no-op unless ``VTSEARCH_EMBED_PRECISION`` names an ``autocast_*`` mode.
    Unlike a weight cast, autocast keeps fp32 master weights and lets torch
    choose per op, holding the reduction-heavy ones (softmax, layer norm) in
    fp32 — numerically the safer half of the two, and the slower one.

    Both paths still return **fp32** vectors: only the compute is half, never
    the stored embedding (see :func:`to_float32`).  Storing half vectors would
    change every downstream matrix's dtype, which is a different change with a
    different blast radius than the one #3143 measured.
    """
    from vtscore.config import embed_autocast_dtype, resolve_device  # noqa: PLC0415

    dtype = embed_autocast_dtype()
    if dtype is None:
        yield
        return
    import torch  # noqa: PLC0415

    device_type = resolve_device().split(":")[0]
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def to_model_inputs(inputs: Any, model: Any) -> dict:
    """Move processor *inputs* onto *model*'s device **and** floating dtype.

    Replaces the ``{k: v.to(device) for ...}`` every embedder wrote by hand.
    The dtype half is what a weight cast needs: fp16 weights fed fp32
    ``pixel_values`` raise ``expected scalar type Half but found Float`` in the
    patch-embedding conv, so the pixels have to follow the weights.

    Only floating tensors are cast.  ``input_ids`` / ``attention_mask`` are
    integer and casting them to half would corrupt token ids outright — quietly,
    for ids above 2048, which is most of a real vocabulary.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = {}
    for key, val in inputs.items():
        if hasattr(val, "is_floating_point") and val.is_floating_point():
            out[key] = val.to(device=device, dtype=dtype)
        elif hasattr(val, "to"):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def to_float32(tensor: Any) -> Any:
    """Upcast a possibly-half tensor back to fp32 before it leaves the GPU.

    Half precision is a *compute* choice; the embedding contract is fp32.
    Numpy would otherwise carry a ``float16`` array straight into the dataset
    pickles and the embedding matrices, silently halving the precision of
    everything stored rather than only of the forward pass.
    """
    return tensor.float() if hasattr(tensor, "float") else tensor
