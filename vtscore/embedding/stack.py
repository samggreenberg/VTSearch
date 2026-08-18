"""What produced these vectors: the software and hardware stack, as a dict.

Two datasets embedded by the same *embedder name* are not necessarily embedded
the same way, and the differences are large enough to matter:

- **The image processor changed under a version range we allow.** transformers
  v5 moved the plain ``SiglipImageProcessor`` name onto the torchvision
  implementation and renamed the PIL one to ``SiglipImageProcessorPil``. On the
  shipped default embedder the two disagree on **58% of preprocessed pixels**,
  by up to two 8-bit levels (issue #3160, measured on 64 VG images).
- **CPU kernel dispatch changes the resize.** An AVX2 host and an AVX-512 host
  disagree on 12.3% of pixels at 384px, each by exactly one 8-bit level, which
  is where #3143's "cross-GPU drift" actually came from. 224px is unaffected.
- **The device matters a little**, though far less than either of the above.

None of this makes a dataset wrong. It makes two datasets *not bit-comparable*,
which is invisible unless somebody writes it down — and vectors are compared
across datasets routinely: a detector's labeled origins are re-embedded on the
current host (``populate_label_embeddings``) and scored against a gallery
embedded wherever that pickle was made.

So: record it. This is provenance, not configuration — nothing reads these
fields to make a decision, and a missing or unknown value is written as ``None``
rather than guessed.
"""

from __future__ import annotations

from typing import Any


def _version(module: str) -> str | None:
    try:
        return getattr(__import__(module), "__version__", None)
    except Exception:  # noqa: BLE001 -- an optional dep that is absent is a fact, not an error
        return None


def _config_value(name: str) -> str | None:
    """Read a vtscore.config setting without making this module import-heavy."""
    try:
        from vtscore import config  # noqa: PLC0415

        value = getattr(config, name, None)
    except Exception:  # noqa: BLE001
        return None
    return str(value) if value is not None else None


def _cpu_capability() -> str | None:
    """The CPU kernel ISA torch dispatches to (``AVX512`` / ``AVX2`` / ``DEFAULT``).

    Read back from torch rather than from ``ATEN_CPU_CAPABILITY``: that variable
    is a *request*, is read once at import, and is ignored in silence when it
    names something the host cannot do.
    """
    try:
        import torch  # noqa: PLC0415

        getter = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
        return str(getter()) if getter else None
    except Exception:  # noqa: BLE001
        return None


def _device_name() -> str | None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
    return None


def _processor_classes(embedder: Any) -> dict[str, str | None]:
    """The preprocessing classes this embedder actually resolved to, if any."""
    proc = getattr(embedder, "_processor", None)
    image_proc = getattr(proc, "image_processor", None)
    return {
        "processor_class": type(proc).__name__ if proc is not None else None,
        "image_processor_class": type(image_proc).__name__ if image_proc is not None else None,
    }


def embedding_stack(embedder: Any = None) -> dict[str, Any]:
    """Describe the stack that is embedding right now.

    *embedder* is optional: pass the loaded embedder to capture the processor
    classes it resolved (the field that distinguishes the two implementations
    the transformers range admits). Everything here is best-effort — provenance
    must never be the reason an import fails.
    """
    stack: dict[str, Any] = {
        # What was *asked for* (#3146's knob) next to what actually resolved
        # (the class name below). A request that did not land is the failure
        # mode both #3146 and #3160 hit, so recording only one of the two is
        # what made those confounds invisible.
        "image_processor_backend": _config_value("IMAGE_PROCESSOR_BACKEND"),
        "image_processor_device": _config_value("IMAGE_PROCESSOR_DEVICE"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "torchvision": _version("torchvision"),
        "pillow": _version("PIL"),
        "cpu_capability": _cpu_capability(),
        "device": _device_name(),
    }
    if embedder is not None:
        try:
            stack.update(_processor_classes(embedder))
        except Exception:  # noqa: BLE001
            stack.update({"processor_class": None, "image_processor_class": None})
    return stack
