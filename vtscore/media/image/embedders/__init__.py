"""The image embedders, one ``embedder_<name>.py`` module each, and their shared bases.

Auto-discovered like a flat ``embedder_*.py`` module in ``vtscore/media/image/``
itself: :func:`vtscore.media._discover_embedders_in` treats an ``embedders/``
sub-package as a container and scans inside it.
"""
