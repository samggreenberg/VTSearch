"""imagekit - face localisation and embedding behind a pluggable provider seam.

The package has two halves:

* :mod:`imagekit.face` is the public face API: :class:`~imagekit.face.FaceRecord`
  (the only face type an application sees), the
  :class:`~imagekit.face.FacePipeline` protocol, the provider registry
  (:func:`~imagekit.face.get_pipeline`), and the open-weights ONNX detector and
  embedder behind the ``oss`` provider.
* :func:`read_image_tensor` and :class:`RuntimeOptions` are the two things every
  caller needs before touching a face: an image in the package's one tensor
  convention (``(H, W, 3)`` ``uint8`` RGB; see :mod:`imagekit.util`) and where
  the models should run.

See ``imagekit/README.md`` for model sources, the JSON config schema, the
provider architecture and the CLIs.
"""

from __future__ import annotations

from imagekit.runtime import RuntimeOptions
from imagekit.util import ImageTensor, decode_image_buffer, read_image_tensor

__all__ = ["ImageTensor", "RuntimeOptions", "decode_image_buffer", "read_image_tensor"]
