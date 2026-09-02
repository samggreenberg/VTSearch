"""Contract tests for the two ``MediaType`` hooks the app calls but the ABC
used to leave undeclared.

``image_response`` was, until #3401, implemented by four shipped types and
reached only through ``getattr(mt, "image_response", None)`` in two route
helpers.  Duck-typing an ABC's own hook makes it invisible three ways: a
third-party author reading ``MediaType`` cannot find it, the extension guides
cannot document it (``scripts/check-extension-docs.py`` rejects a table naming
a member the class does not define), and a typo in an override degrades to
"this type has no picture" instead of failing.  These tests pin the declared
shape: a ``None`` default that keeps a minimal type working, an override that
the callers see, and the ``load_demo_source`` parameters that used to hide
inside ``**kwargs``.
"""

from __future__ import annotations

import inspect

from vtscore.media.base import MediaResponse, MediaType


class _MinimalMediaType(MediaType):
    """The smallest legal media type: every abstract member, nothing else.

    Stands in for a third-party type written against the ABC alone, which is
    exactly the plugin the ``return None`` default has to keep working.
    """

    @property
    def type_id(self) -> str:
        return "minimal"

    @property
    def name(self) -> str:
        return "Minimal"

    @property
    def icon(self) -> str:
        return "file"

    @property
    def file_extensions(self) -> list[str]:
        return ["*.min"]

    @property
    def loops(self) -> bool:
        return False

    @property
    def demo_datasets(self) -> list:
        return []

    def load_media_data(self, file_path, media_bytes=None) -> dict:
        return {"duration": 0}

    def media_response(self, media: dict) -> MediaResponse:
        return MediaResponse(data=b"", mimetype="application/octet-stream", download_name="m")


class TestImageResponseDefault:
    def test_declared_on_the_abc(self):
        """The hook is a real member, not something four subclasses invented.

        The extension guides tabulate ``image_response`` under the
        ``MediaType`` contract, and ``check-extension-docs.py`` resolves every
        tabulated name against the class — so an undeclared hook would fail
        that gate rather than ship undocumented again.
        """
        assert "image_response" in vars(MediaType)

    def test_default_returns_none(self):
        assert _MinimalMediaType().image_response({"id": 1}) is None

    def test_default_is_not_abstract(self):
        """A type with no visual form must instantiate without overriding it."""
        assert _MinimalMediaType() is not None
        assert "image_response" not in MediaType.__abstractmethods__

    def test_text_type_keeps_the_default(self):
        """Text has no picture; it inherits ``None`` rather than overriding."""
        from vtscore.media.text.media_type import TextMediaType

        assert TextMediaType().image_response({"id": 1, "media_string": "hello"}) is None

    def test_override_is_seen_through_the_base_class(self):
        class _Painted(_MinimalMediaType):
            def image_response(self, media: dict) -> MediaResponse | None:
                return MediaResponse(data=b"PNG", mimetype="image/png", download_name="x.png")

        resp = _Painted().image_response({"id": 2})
        assert resp is not None
        assert resp.mimetype == "image/png"


class TestEnsureThumbnailBytesDefault:
    def test_default_is_a_pure_read(self):
        """No generation, no mutation: a type with no cheap thumbnail says so."""
        mt = _MinimalMediaType()
        assert mt.ensure_thumbnail_bytes({"id": 1}) is None
        assert mt.ensure_thumbnail_bytes({"id": 1, "thumbnail_bytes": b"PNG"}) == b"PNG"

    def test_default_does_not_invent_a_key(self):
        media = {"id": 1}
        _MinimalMediaType().ensure_thumbnail_bytes(media)
        assert "thumbnail_bytes" not in media


class TestLoadDemoSourceSignature:
    """``vtscore.datasets.loader_demo`` passes four keyword arguments that the
    ABC used to swallow into ``**kwargs``, so the declared signature told a
    plugin author nothing about what their override would receive."""

    def test_declares_every_argument_the_loader_passes(self):
        params = inspect.signature(MediaType.load_demo_source).parameters
        for expected in ("embedder", "slice_frac_start", "slice_frac_end", "skip_embedding"):
            assert expected in params, f"{expected} is passed by the demo loader but undeclared"

    def test_keeps_a_kwargs_tail(self):
        """Additive by construction: an override predating the change, or one
        accepting only what it uses, still satisfies the contract."""
        params = inspect.signature(MediaType.load_demo_source).parameters
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def test_new_arguments_are_optional(self):
        params = inspect.signature(MediaType.load_demo_source).parameters
        for expected in ("embedder", "slice_frac_start", "slice_frac_end", "skip_embedding"):
            assert params[expected].default is not inspect.Parameter.empty

    def test_base_still_rejects_an_unknown_source(self):
        mt = _MinimalMediaType()
        try:
            mt.load_demo_source("nope", [], 0, None, {})
        except ValueError as exc:
            assert "minimal" in str(exc)
        else:  # pragma: no cover - the base must not silently succeed
            raise AssertionError("expected ValueError for an unrecognised source")
