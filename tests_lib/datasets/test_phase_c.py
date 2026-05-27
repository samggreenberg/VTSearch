"""Phase C plugin-interface streamlines; coverage for the three landed surfaces.

Three independent candidates:

#9; :meth:`MediaConverter.convert_normalized` validates and default-fills
``params`` before dispatch, so :meth:`convert` receives a fully-populated
non-``None`` dict.

#13; :mod:`vtscore.plugins.uploads` defines :class:`UploadedFile` plus the
:class:`CliUploadedFile` / :class:`BytesIOUploadedFile` adapters; the
``DatasetImporter`` / ``LabelImporter`` base classes' default ``run_cli``
wraps path strings so plugin bodies see one shape regardless of ingress.

#2; :meth:`DatasetImporter.yield_precomputed` collapses the three
precomputed-dict writes into one helper call so a single miskeyed entry
cannot land in only one or two of the parallel dicts.
"""

from __future__ import annotations

from typing import Any

import pytest

from vtscore.converters.base import MediaConverter
from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.plugins import PluginField
from vtscore.plugins.uploads import (
    BytesIOUploadedFile,
    CliUploadedFile,
    UploadedFile,
    wrap_cli_file_fields,
)


# ---------------------------------------------------------------------------
# #9: convert_normalized
# ---------------------------------------------------------------------------


class _NClipsConverter(MediaConverter):
    """Throwaway converter declaring one bounded integer field."""

    # ``MediaConverter.name`` is normally derived from source_type /
    # target_type, but we override it here so tests get a stable name
    # the marshmallow schema cache can key on.
    display_name = "Fake Video → Image"
    description = ""
    fields = [
        PluginField(
            key="n_clips",
            label="Frames",
            field_type="number",
            default="10",
            required=False,
            min="1",
            max="100",
            step="1",
        ),
    ]

    last_params: dict[str, Any] | None = None

    @property
    def name(self) -> str:  # type: ignore[override]
        return "fakevideo2image"

    @property
    def source_type(self) -> str:
        return "video"

    @property
    def target_type(self) -> str:
        return "image"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.last_params = dict(params) if params is not None else None
        return [{"filename": "out.png", "media_bytes": b""}]


class TestConvertNormalized:
    def test_none_params_get_default_filled(self):
        c = _NClipsConverter()
        c.convert_normalized({"filename": "v.mp4"}, None)
        assert c.last_params == {"n_clips": 10}

    def test_string_value_coerced_to_int(self):
        c = _NClipsConverter()
        c.convert_normalized({"filename": "v.mp4"}, {"n_clips": "5"})
        assert c.last_params == {"n_clips": 5}

    def test_empty_string_falls_back_to_default(self):
        c = _NClipsConverter()
        c.convert_normalized({"filename": "v.mp4"}, {"n_clips": ""})
        assert c.last_params == {"n_clips": 10}

    def test_out_of_range_value_raises_value_error(self):
        c = _NClipsConverter()
        with pytest.raises(ValueError, match="n_clips"):
            c.convert_normalized({"filename": "v.mp4"}, {"n_clips": 9999})

    def test_unknown_keys_are_dropped(self):
        c = _NClipsConverter()
        c.convert_normalized({"filename": "v.mp4"}, {"n_clips": 3, "ignored": "garbage"})
        assert c.last_params == {"n_clips": 3}

    def test_convert_directly_still_receives_raw_params(self):
        """Backwards-compat: third-party callers that invoke convert() directly
        bypass normalization, so the raw params reach the body."""
        c = _NClipsConverter()
        c.convert({"filename": "v.mp4"}, {"n_clips": "raw-string"})
        assert c.last_params == {"n_clips": "raw-string"}


# ---------------------------------------------------------------------------
# #13: UploadedFile + CLI wrapping
# ---------------------------------------------------------------------------


class TestUploadedFileAdapters:
    def test_cli_uploaded_file_exposes_filename(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello")
        u = CliUploadedFile(p)
        assert u.filename == "data.bin"
        assert u.read() == b"hello"

    def test_cli_uploaded_file_save_copies(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dst = tmp_path / "dst.bin"
        CliUploadedFile(src).save(dst)
        assert dst.read_bytes() == b"payload"

    def test_bytesio_uploaded_file_exposes_filename_and_name(self):
        u = BytesIOUploadedFile(b"abc", filename="upload.pkl")
        assert u.filename == "upload.pkl"
        # Legacy ``.name`` shim; pre-Phase-C readers expected it.
        assert u.name == "upload.pkl"
        assert u.read() == b"abc"

    def test_bytesio_uploaded_file_save_writes_bytes(self, tmp_path):
        dst = tmp_path / "out.bin"
        BytesIOUploadedFile(b"abc", filename="x").save(dst)
        assert dst.read_bytes() == b"abc"

    def test_cli_and_bytesio_satisfy_uploaded_file_protocol(self):
        assert isinstance(CliUploadedFile("/tmp/x"), UploadedFile)
        assert isinstance(BytesIOUploadedFile(b"", "x"), UploadedFile)


class TestWrapCliFileFields:
    def test_wraps_string_paths(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"x")
        fields = [PluginField(key="file", label="F", field_type="file")]
        wrapped = wrap_cli_file_fields(fields, {"file": str(p)})
        assert isinstance(wrapped["file"], CliUploadedFile)
        assert wrapped["file"].filename == "data.bin"

    def test_passes_uploaded_file_through(self):
        existing = BytesIOUploadedFile(b"", "x")
        fields = [PluginField(key="file", label="F", field_type="file")]
        wrapped = wrap_cli_file_fields(fields, {"file": existing})
        assert wrapped["file"] is existing

    def test_leaves_non_file_fields_alone(self):
        fields = [
            PluginField(key="file", label="F", field_type="file"),
            PluginField(key="name", label="N", field_type="text"),
        ]
        wrapped = wrap_cli_file_fields(fields, {"file": None, "name": "/some/path"})
        assert wrapped["file"] is None
        assert wrapped["name"] == "/some/path"

    def test_returns_a_copy(self):
        fields: list[PluginField] = []
        original = {"k": "v"}
        wrapped = wrap_cli_file_fields(fields, original)
        wrapped["k"] = "mutated"
        assert original["k"] == "v"


class _FileImporter(DatasetImporter):
    """In-memory importer with a single ``file`` field.

    ``run`` records the type / surface of the value it receives so tests
    can assert that ``run_cli`` wraps path strings in
    :class:`CliUploadedFile` before dispatching.
    """

    name = "_file_importer"
    display_name = "_file_importer"
    description = ""
    fields = [ImporterField(key="file", label="F", field_type="file")]

    last_value: Any = None

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        self.last_value = field_values["file"]


class TestRunCliWrapsFileFields:
    def test_string_path_arrives_as_uploaded_file(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"data")
        imp = _FileImporter()
        imp.run_cli({"file": str(p)}, {})
        assert isinstance(imp.last_value, UploadedFile)
        assert imp.last_value.filename == "x.bin"
        assert imp.last_value.read() == b"data"

    def test_already_uploaded_file_passes_through(self):
        u = BytesIOUploadedFile(b"x", "x")
        imp = _FileImporter()
        imp.run_cli({"file": u}, {})
        assert imp.last_value is u


# ---------------------------------------------------------------------------
# #2: yield_precomputed
# ---------------------------------------------------------------------------


class _PrecomputedImporter(DatasetImporter):
    name = "_pre"
    display_name = "_pre"
    description = ""
    fields = []


class TestYieldPrecomputed:
    def test_routes_to_three_underlying_dicts(self):
        imp = _PrecomputedImporter()
        imp.yield_precomputed(
            "tone.wav",
            embedding=[0.1, 0.2],
            md5="deadbeef",
            metadata={"source": "test"},
        )
        assert imp.content_vectors == {"tone.wav": [0.1, 0.2]}
        assert imp.content_md5s == {"tone.wav": "deadbeef"}
        assert imp.custom_metadata_map == {"tone.wav": {"source": "test"}}

    def test_omitted_args_dont_touch_their_dicts(self):
        imp = _PrecomputedImporter()
        imp.yield_precomputed("a.wav", embedding=[1.0])
        assert imp.content_vectors == {"a.wav": [1.0]}
        assert imp.content_md5s == {}
        assert imp.custom_metadata_map == {}

    def test_all_three_optional(self):
        imp = _PrecomputedImporter()
        imp.yield_precomputed("noop.wav")
        assert imp.content_vectors == {}
        assert imp.content_md5s == {}
        assert imp.custom_metadata_map == {}

    def test_subsequent_calls_accumulate(self):
        imp = _PrecomputedImporter()
        imp.yield_precomputed("a.wav", embedding=[1.0])
        imp.yield_precomputed("b.wav", embedding=[2.0])
        assert imp.content_vectors == {"a.wav": [1.0], "b.wav": [2.0]}

    def test_legacy_direct_dict_writes_still_work(self):
        """Back-compat: external importers writing to the dicts directly
        keep working unchanged."""
        imp = _PrecomputedImporter()
        imp.content_vectors["legacy.wav"] = [0.0]
        imp.yield_precomputed("new.wav", embedding=[1.0])
        assert imp.content_vectors == {"legacy.wav": [0.0], "new.wav": [1.0]}
