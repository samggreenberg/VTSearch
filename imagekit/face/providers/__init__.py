"""The built-in providers: ``oss`` (open weights, this package) and ``vista`` (proprietary, lazy)."""

from imagekit.face.providers.oss import OssFacePipeline
from imagekit.face.providers.vista import VistaFacePipeline, VistaProviderUnavailable

__all__ = ["OssFacePipeline", "VistaFacePipeline", "VistaProviderUnavailable"]
