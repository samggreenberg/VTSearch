"""A demo loaded as several linked datasets: each output names its converter in ``source_specs``.

``/api/dataset/load-demo`` with ``outputs`` hands the multi-dataset pipeline one
:class:`OutputSpec` per dataset (the photos, and the faces cut out of them).
The pipeline narrows each into the field values :meth:`DemoDatasetImporter.run`
receives, which carry the converter in ``source_specs``, and records each
output's origin from them, which carries the rows as JSON.  Both must reach
``load_demo_dataset`` (and a later reload) as the converter the output asked for.
"""

from __future__ import annotations

from unittest.mock import patch

from vtscore.datasets.importers.base import OutputSpec, SourceSpec
from vtscore.datasets.importers.demo import DemoDatasetImporter, _demo_converter

_DEMO = "celebrity_together_s"
_PHOTOS = OutputSpec(media_type="image", source_specs=[SourceSpec(source_type="image")])
_FACES = OutputSpec(media_type="face", source_specs=[SourceSpec(source_type="image", converter="image2face")])


class TestDemoConverter:
    def test_the_converter_field_wins(self):
        assert (
            _demo_converter({"converter": "image2text", "source_specs": [{"converter": "image2face"}]}) == "image2text"
        )

    def test_a_source_row_names_it(self):
        assert _demo_converter(_FACES.narrow({"name": _DEMO})) == "image2face"

    def test_an_origin_carries_the_rows_as_json(self):
        assert (
            _demo_converter({"source_specs": '[{"source_type": "image", "converter": "image2face"}]'}) == "image2face"
        )

    def test_a_direct_row_or_nothing_is_no_converter(self):
        assert _demo_converter(_PHOTOS.narrow({"name": _DEMO})) == ""
        assert _demo_converter({}) == ""
        assert _demo_converter({"source_specs": "not json"}) == ""


class TestDemoOutputs:
    def test_each_output_loads_the_demo_with_its_own_converter(self):
        importer = DemoDatasetImporter()
        with patch("vtscore.datasets.importers.demo.load_demo_dataset") as load:
            produced = list(importer.run_outputs({"name": _DEMO}, [_PHOTOS, _FACES]))

        assert [output for output, _medias in produced] == [_PHOTOS, _FACES]
        assert [c.args[0] for c in load.call_args_list] == [_DEMO, _DEMO]
        assert [c.kwargs["converter_name"] for c in load.call_args_list] == ["", "image2face"]

    def test_an_output_reloads_as_the_same_conversion(self):
        importer = DemoDatasetImporter()
        origin = importer.build_origin(_FACES.narrow({"name": _DEMO}))

        assert importer.reload_from_origin(origin) == {"name": _DEMO, "converter": "image2face"}
        assert importer.reload_from_origin(importer.build_origin(_PHOTOS.narrow({"name": _DEMO}))) == {"name": _DEMO}
