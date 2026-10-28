"""The built-in providers: ``oss`` (open weights, this package) and ``ext`` (an external backend, lazy)."""

from imagekit.face.providers.oss import OssFacePipeline
from imagekit.face.providers.ext import ExtFacePipeline, ExtProviderUnavailable

__all__ = ["ExtFacePipeline", "ExtProviderUnavailable", "OssFacePipeline"]
