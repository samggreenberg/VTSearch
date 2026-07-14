"""Direct tests for the importer base-class validation / error paths.

``vtscore/datasets/importers/base/core.py`` (:class:`ImporterBase`) and its
richer subclass :class:`DatasetImporter` are normally exercised only through
concrete importers, so a regression in the base machinery surfaces as a
*confusing subclass failure* rather than a clear base-class error.  These
tests pin the base-class contracts directly, covering the three failure
themes that had no coverage:

- **Bad params** — the ``NotImplementedError`` "you forgot to override this"
  surfaces, the ``_ingest_spec_stream`` unknown-converter guard, and the
  ``run()`` fallback that swallows a ``ValueError`` from
  ``effective_source_specs``.
- **Partial-batch failures** — ``None`` raw pairs and ``None`` converter
  outputs are skipped mid-stream without disturbing the IDs of the survivors.
- **Cancellation propagation** — a :class:`CancelledError` raised from any
  subclass hook (``list_records`` / ``fetch_record`` / ``fetch_source_media``
  / a converter) propagates cleanly out of ``run()`` and is **not** swallowed
  by the ``except ValueError`` fallback.

Everything here is library tier (no Flask / app imports), so it lives under
``tests_lib/``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from vtscore.concurrency.progress import CancelledError
from vtscore.datasets.importers.base import (
    DatasetImporter,
    ImporterBase,
    PluginField,
    SourceSpec,
)


def _media_type_field() -> PluginField:
    """A ``media_type`` select so ``effective_source_specs`` resolves specs."""
    return PluginField(
        "media_type",
        "Media Type",
        "select",
        options=["image", "video", "audio", "text"],
        default="image",
    )


# ---------------------------------------------------------------------------
# Bad params: "you didn't override this" NotImplementedError surfaces
# ---------------------------------------------------------------------------


class TestNotImplementedSurface:
    """The base raises a *helpful* NotImplementedError, not an opaque one."""

    def test_thin_base_run_points_at_dataset_importer(self):
        """``ImporterBase.run`` (the thin base) tells the author what to do."""

        class _Thin(ImporterBase):
            name = "thin_err"
            display_name = "Thin"
            description = "."
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError) as exc:
            _Thin().run({}, {})
        msg = str(exc.value)
        assert "run()" in msg
        # It nudges the author toward the two supported override strategies.
        assert "DatasetImporter" in msg

    def test_get_field_options_default_names_the_field(self):
        """The default ``get_field_options`` echoes the offending field key."""

        class _Thin(ImporterBase):
            name = "thin_opts"
            display_name = "Thin"
            description = "."
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError) as exc:
            _Thin().get_field_options("some_field", {})
        assert "some_field" in str(exc.value)

    def test_dataset_importer_list_records_hints_at_source_specs(self):
        """``list_records`` explains the media_type / source_specs fallback."""

        class _Imp(DatasetImporter):
            name = "lr_err"
            display_name = "LR"
            description = "."
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError) as exc:
            _Imp().list_records({})
        msg = str(exc.value)
        assert "list_records" in msg
        assert "media_type" in msg

    def test_dataset_importer_fetch_record_default_raises(self):
        class _Imp(DatasetImporter):
            name = "fr_err"
            display_name = "FR"
            description = "."
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError, match="fetch_record"):
            _Imp().fetch_record({"any": "record"}, {})

    def test_bare_run_falls_through_to_list_records(self):
        """A DatasetImporter overriding nothing raises the list_records error
        from ``run`` (no media_type → spec resolution fails → fallback path)."""

        class _Bare(DatasetImporter):
            name = "bare_err"
            display_name = "Bare"
            description = "."
            fields: list[PluginField] = []

        with pytest.raises(NotImplementedError, match="list_records"):
            _Bare().run({}, {})


# ---------------------------------------------------------------------------
# Bad params: _ingest_spec_stream unknown-converter guard
# ---------------------------------------------------------------------------


class TestIngestSpecStreamBadConverter:
    """``_ingest_spec_stream`` has its own converter-resolution guard.

    ``effective_source_specs`` validates converters up front, so in the normal
    ``run`` flow an unknown converter is caught earlier.  The guard inside
    ``_ingest_spec_stream`` is the last line of defence for callers that build
    a stream by hand (spec-aware importers driving their own loop); it must
    raise a clear ``ValueError`` rather than crash on ``None.convert``.
    """

    def _make_importer(self) -> DatasetImporter:
        class _Imp(DatasetImporter):
            name = "ingest_bad_conv"
            display_name = "Ingest"
            description = "."
            fields = [_media_type_field()]

        return _Imp()

    def test_unknown_converter_raises_value_error(self):
        imp = self._make_importer()
        spec = SourceSpec(source_type="video", converter="does_not_exist", params={})
        stream = iter([(spec, {"filename": "v.mp4", "media_bytes": b"VID"})])
        medias: dict = {}
        with pytest.raises(ValueError, match="Unknown converter"):
            imp._ingest_spec_stream(stream, medias, {"importer": "x", "params": {}}, 1)
        # Nothing should have been ingested before the guard fired.
        assert medias == {}


# ---------------------------------------------------------------------------
# Bad params: run() swallows ValueError from effective_source_specs
# ---------------------------------------------------------------------------


class TestRunSwallowsSpecValueError:
    """``run`` catches ``ValueError`` from ``effective_source_specs`` and falls
    back to the per-record path.  This is *why* a malformed ``source_specs``
    value surfaces as a confusing ``list_records`` NotImplementedError instead
    of a "bad source_specs" message — a characterization test that pins the
    fallback branch so a regression there is caught at the base level.
    """

    def test_invalid_source_specs_json_falls_back_to_list_records(self):
        # Only fetch_source_media is implemented (the spec path); no
        # list_records.  A bad source_specs value makes effective_source_specs
        # raise ValueError, which run() swallows → fallback → list_records
        # NotImplementedError.
        class _SpecOnly(DatasetImporter):
            name = "spec_only"
            display_name = "Spec Only"
            description = "."
            fields = [_media_type_field()]

            def fetch_source_media(self, spec, field_values, thin=False):
                yield {"filename": "unreached.png", "media_bytes": b"X"}

        imp = _SpecOnly()
        with pytest.raises(NotImplementedError, match="list_records"):
            imp.run({"media_type": "image", "source_specs": "{not valid json"}, {})

    def test_non_value_error_is_not_swallowed(self):
        """Only ``ValueError`` is caught; any other error from
        ``effective_source_specs`` propagates (so a real bug isn't masked)."""

        class _Imp(DatasetImporter):
            name = "not_swallowed"
            display_name = "NS"
            description = "."
            fields = [_media_type_field()]

            def effective_source_specs(self, field_values):
                raise RuntimeError("boom from spec resolution")

        with pytest.raises(RuntimeError, match="boom from spec resolution"):
            _Imp().run({"media_type": "image"}, {})


# ---------------------------------------------------------------------------
# Partial-batch failures: None raws / None converter outputs are skipped
# ---------------------------------------------------------------------------


class TestPartialBatchSkips:
    """Mid-stream ``None`` values are dropped without disturbing survivors."""

    def _spec_importer(self, records: list[Any]) -> DatasetImporter:
        class _Imp(DatasetImporter):
            name = "partial_spec"
            display_name = "Partial"
            description = "."
            fields = [_media_type_field()]

            def fetch_source_media(self, spec, field_values, thin=False):
                yield from iter(records)

        return _Imp()

    def test_none_raw_pairs_skipped_direct_spec(self):
        """``(spec, None)`` pairs are skipped; IDs stay contiguous over survivors."""
        records = [
            {"filename": "a.png", "media_bytes": b"A"},
            None,
            {"filename": "b.png", "media_bytes": b"B"},
        ]
        imp = self._spec_importer(records)
        medias: dict = {}
        imp.run({"media_type": "image"}, medias)

        assert list(medias.keys()) == [1, 2]
        assert [medias[i]["filename"] for i in (1, 2)] == ["a.png", "b.png"]

    def test_none_converter_output_skipped(self, monkeypatch):
        """A converter that returns ``[media, None]`` has the ``None`` dropped."""
        from vtscore.converters import get_converter

        v2i = get_converter("video2image")
        assert v2i is not None

        def fake_convert(media, params):
            return [{"filename": "frame_0.png", "media_bytes": b"PNG", "duration": 0}, None]

        monkeypatch.setattr(v2i, "convert", fake_convert)

        imp = self._spec_importer([{"filename": "clip.mp4", "media_bytes": b"VID"}])
        medias: dict = {}
        source_specs = json.dumps([{"source_type": "video", "converter": "video2image", "params": {}}])
        imp.run({"media_type": "image", "source_specs": source_specs}, medias)

        # The good frame lands; the None is silently dropped.
        assert list(medias.keys()) == [1]
        assert medias[1]["filename"] == "frame_0.png"

    def test_ingest_spec_stream_skips_none_directly(self):
        """Direct call: a hand-built stream with a ``None`` raw is skipped and
        ``next_id`` advances only for the ingested media."""

        class _Imp(DatasetImporter):
            name = "ingest_none"
            display_name = "Ingest"
            description = "."
            fields = [_media_type_field()]

        imp = _Imp()
        spec = SourceSpec(source_type="image", converter=None, params={})
        # A partial batch: two failed fetches (None) around one survivor.
        pairs: list[tuple[SourceSpec, Any]] = [
            (spec, None),
            (spec, {"filename": "kept.png", "media_bytes": b"K"}),
            (spec, None),
        ]
        stream = iter(pairs)
        medias: dict = {}
        next_id = imp._ingest_spec_stream(stream, medias, {"importer": "x", "params": {}}, 1)

        assert list(medias.keys()) == [1]
        assert medias[1]["filename"] == "kept.png"
        # Only one media was ingested, so the returned id is 2 (1 consumed).
        assert next_id == 2

    def test_fetch_all_source_media_default_skips_none(self):
        """The default ``fetch_all_source_media`` drops ``None`` yields from
        ``fetch_source_media`` before pairing them with the spec."""

        class _Imp(DatasetImporter):
            name = "fasm_none"
            display_name = "FASM"
            description = "."
            fields = [_media_type_field()]

            def fetch_source_media(self, spec, field_values, thin=False):
                yield {"filename": "a.png"}
                yield None
                yield {"filename": "b.png"}

        imp = _Imp()
        spec = SourceSpec(source_type="image", converter=None, params={})
        pairs = list(imp.fetch_all_source_media([spec], {"media_type": "image"}))

        assert [raw["filename"] for _s, raw in pairs] == ["a.png", "b.png"]
        assert all(s is spec for s, _raw in pairs)

    def test_fallback_path_skips_none_records(self):
        """Sanity check on the per-record fallback: ``None`` fetch_record
        results are dropped (mirrors the spec-path skip for the other flow)."""

        class _Imp(DatasetImporter):
            name = "fallback_none"
            display_name = "Fallback"
            description = "."
            fields: list[PluginField] = []  # no media_type → per-record path

            def list_records(self, field_values):
                return ["keep", "drop", "keep2"]

            def fetch_record(self, record, field_values, thin=False):
                if record == "drop":
                    return None
                return {"media_type": "audio", "filename": record, "embeddings": {}}

        medias: dict = {}
        _Imp().run({}, medias)
        assert [medias[i]["filename"] for i in medias] == ["keep", "keep2"]


# ---------------------------------------------------------------------------
# Cancellation propagation: CancelledError from any hook escapes run()
# ---------------------------------------------------------------------------


class TestCancellationPropagation:
    """A cooperative ``CancelledError`` raised inside a subclass hook must
    propagate out of ``run`` untouched — the base must never swallow it (in
    particular the ``except ValueError`` fallback must not catch it, since
    ``CancelledError`` is not a ``ValueError``)."""

    def test_cancel_in_list_records_propagates(self):
        class _Imp(DatasetImporter):
            name = "cancel_lr"
            display_name = "C"
            description = "."
            fields: list[PluginField] = []  # per-record fallback path

            def list_records(self, field_values):
                raise CancelledError("Operation cancelled by user")

        with pytest.raises(CancelledError):
            _Imp().run({}, {})

    def test_cancel_in_fetch_record_propagates(self):
        class _Imp(DatasetImporter):
            name = "cancel_fr"
            display_name = "C"
            description = "."
            fields: list[PluginField] = []

            def list_records(self, field_values):
                return ["a", "b"]

            def fetch_record(self, record, field_values, thin=False):
                raise CancelledError("Operation cancelled by user")

        with pytest.raises(CancelledError):
            _Imp().run({}, {})

    def test_cancel_in_fetch_source_media_propagates(self):
        """Spec path: cancellation mid-stream escapes ``_ingest_spec_stream``."""

        class _Imp(DatasetImporter):
            name = "cancel_fsm"
            display_name = "C"
            description = "."
            fields = [_media_type_field()]

            def fetch_source_media(self, spec, field_values, thin=False):
                yield {"filename": "first.png", "media_bytes": b"A"}
                raise CancelledError("Operation cancelled by user")

        with pytest.raises(CancelledError):
            _Imp().run({"media_type": "image"}, {})

    def test_cancel_in_converter_propagates(self, monkeypatch):
        """Cancellation raised inside a converter escapes the ingest loop."""
        from vtscore.converters import get_converter

        v2i = get_converter("video2image")
        assert v2i is not None

        def cancelling_convert(media, params):
            raise CancelledError("Operation cancelled by user")

        monkeypatch.setattr(v2i, "convert", cancelling_convert)

        class _Imp(DatasetImporter):
            name = "cancel_conv"
            display_name = "C"
            description = "."
            fields = [_media_type_field()]

            def fetch_source_media(self, spec, field_values, thin=False):
                yield {"filename": "clip.mp4", "media_bytes": b"VID"}

        source_specs = json.dumps([{"source_type": "video", "converter": "video2image", "params": {}}])
        with pytest.raises(CancelledError):
            _Imp().run({"media_type": "image", "source_specs": source_specs}, {})

    def test_cancel_in_effective_source_specs_not_swallowed(self):
        """The ``except ValueError`` fallback must not catch a CancelledError
        raised while resolving specs."""

        class _Imp(DatasetImporter):
            name = "cancel_ess"
            display_name = "C"
            description = "."
            fields = [_media_type_field()]

            def effective_source_specs(self, field_values):
                raise CancelledError("Operation cancelled by user")

        with pytest.raises(CancelledError):
            _Imp().run({"media_type": "image"}, {})
