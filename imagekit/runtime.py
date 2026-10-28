"""Runtime options shared by every ``initialize()`` in :mod:`imagekit`.

:class:`RuntimeOptions` is a plain value object: which execution provider
ONNX Runtime should use, which GPU, and how many intra-op threads.  Every
detector, extractor and pipeline takes one at ``initialize()`` time and never
reads the environment itself, so a caller (a CLI, a server) decides once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnxruntime as ort

#: ONNX Runtime's CUDA execution provider.
CUDA_PROVIDER = "CUDAExecutionProvider"

#: ONNX Runtime's CPU execution provider, the fallback behind every GPU session.
CPU_PROVIDER = "CPUExecutionProvider"

#: ``SessionOptions.log_severity_level`` for every session: errors only.  The
#: InsightFace recognisers declare a batch of 1 on their output while the
#: extractor embeds a photo's faces as one batch, so at the default (warning)
#: level ONNX Runtime prints a ``VerifyOutputSizes`` warning for every photo
#: with more than one face, which floods a server's console.
LOG_SEVERITY_ERROR = 3


@dataclass(frozen=True)
class RuntimeOptions:
    """Where and how wide a model runs.

    Args:
        use_gpu: Run on the CUDA execution provider.  ``initialize()`` raises
            when it is requested but the installed ``onnxruntime`` build does
            not offer it, rather than silently running on the CPU.
        gpu_device_id: Which CUDA device, when ``use_gpu`` is set.
        num_threads: Intra-op thread count for CPU inference.  ``0`` leaves
            ONNX Runtime's own default (one thread per physical core).
    """

    use_gpu: bool = False
    gpu_device_id: int = 0
    num_threads: int = 0

    def providers(self) -> list[Any]:
        """The execution-provider list for ``onnxruntime.InferenceSession``."""
        if self.use_gpu:
            return [(CUDA_PROVIDER, {"device_id": int(self.gpu_device_id)}), CPU_PROVIDER]
        return [CPU_PROVIDER]

    def check_available(self) -> None:
        """Raise :class:`RuntimeError` when a requested provider is not installed."""
        import onnxruntime as ort  # noqa: PLC0415

        if self.use_gpu and CUDA_PROVIDER not in ort.get_available_providers():
            raise RuntimeError(
                f"RuntimeOptions(use_gpu=True) but this onnxruntime build has no {CUDA_PROVIDER} "
                f"(available: {ort.get_available_providers()}); install onnxruntime-gpu or run on the CPU"
            )

    def session_options(self) -> "ort.SessionOptions":
        """A fresh ``SessionOptions`` carrying the thread settings, logging errors only."""
        import onnxruntime as ort  # noqa: PLC0415

        opts = ort.SessionOptions()
        opts.log_severity_level = LOG_SEVERITY_ERROR
        if self.num_threads > 0:
            opts.intra_op_num_threads = int(self.num_threads)
            opts.inter_op_num_threads = 1
        return opts

    def make_session(self, model_path: str) -> "ort.InferenceSession":
        """Open *model_path* as an ``InferenceSession`` on these options.

        Raises whatever ``onnxruntime`` raises on a bad model, after
        :meth:`check_available` has refused an uninstalled provider.
        """
        import onnxruntime as ort  # noqa: PLC0415

        self.check_available()
        return ort.InferenceSession(model_path, sess_options=self.session_options(), providers=self.providers())
