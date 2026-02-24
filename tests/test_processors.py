"""Tests for the Processor base class hierarchy (Processor, Detector, Localizer, Extractor)."""

import pytest

from vtsearch.media.base import Detector, Extractor, Localizer, Processor


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing
# ---------------------------------------------------------------------------


class StubDetector(Detector):
    """Trivial detector that returns a fixed boolean."""

    def __init__(self, name="stub-det", media_type="audio", result=True):
        self._name = name
        self._media_type = media_type
        self._result = result

    @property
    def name(self):
        return self._name

    @property
    def media_type(self):
        return self._media_type

    def detect(self, clip):
        return self._result


class StubLocalizer(Localizer):
    """Trivial localizer that returns a fixed list of bounding boxes."""

    def __init__(self, name="stub-loc", media_type="image", results=None):
        self._name = name
        self._media_type = media_type
        self._results = results if results is not None else []

    @property
    def name(self):
        return self._name

    @property
    def media_type(self):
        return self._media_type

    def localize(self, clip):
        return self._results


class StubExtractor(Extractor):
    """Trivial extractor that returns a fixed list of results."""

    def __init__(self, name="stub-ext", media_type="image", results=None):
        self._name = name
        self._media_type = media_type
        self._results = results if results is not None else []

    @property
    def name(self):
        return self._name

    @property
    def media_type(self):
        return self._media_type

    def extract(self, clip):
        return self._results


# ---------------------------------------------------------------------------
# Processor ABC
# ---------------------------------------------------------------------------


class TestProcessorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Processor()

    def test_detector_is_processor(self):
        det = StubDetector()
        assert isinstance(det, Processor)

    def test_localizer_is_processor(self):
        loc = StubLocalizer()
        assert isinstance(loc, Processor)

    def test_extractor_is_processor(self):
        ext = StubExtractor()
        assert isinstance(ext, Processor)

    def test_load_model_is_noop_by_default(self):
        det = StubDetector()
        det.load_model()  # should not raise

    def test_to_dict(self):
        det = StubDetector(name="my-proc", media_type="video")
        d = det.to_dict()
        assert d == {"name": "my-proc", "media_type": "video"}


# ---------------------------------------------------------------------------
# Detector ABC
# ---------------------------------------------------------------------------


class TestDetectorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Detector()

    def test_detect_returns_bool(self):
        det = StubDetector(result=True)
        assert det.detect({}) is True

        det2 = StubDetector(result=False)
        assert det2.detect({}) is False

    def test_process_delegates_to_detect(self):
        det = StubDetector(result=True)
        assert det.process({}) is True

        det2 = StubDetector(result=False)
        assert det2.process({}) is False

    def test_name_and_media_type(self):
        det = StubDetector(name="dog_barks", media_type="audio")
        assert det.name == "dog_barks"
        assert det.media_type == "audio"

    def test_to_dict(self):
        det = StubDetector(name="test-det", media_type="paragraph")
        d = det.to_dict()
        assert d["name"] == "test-det"
        assert d["media_type"] == "paragraph"


# ---------------------------------------------------------------------------
# Localizer ABC
# ---------------------------------------------------------------------------


class TestLocalizerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Localizer()

    def test_localize_returns_bboxes(self):
        boxes = [
            {"confidence": 0.95, "bbox": [10, 20, 200, 300]},
            {"confidence": 0.73, "bbox": [400, 50, 600, 250]},
        ]
        loc = StubLocalizer(results=boxes)
        assert loc.localize({}) == boxes

    def test_localize_returns_empty_list(self):
        loc = StubLocalizer(results=[])
        assert loc.localize({}) == []

    def test_process_delegates_to_localize(self):
        boxes = [{"confidence": 0.88, "bbox": [0, 0, 100, 100]}]
        loc = StubLocalizer(results=boxes)
        assert loc.process({}) == boxes

    def test_process_returns_empty_list(self):
        loc = StubLocalizer(results=[])
        assert loc.process({}) == []

    def test_name_and_media_type(self):
        loc = StubLocalizer(name="face_regions", media_type="image")
        assert loc.name == "face_regions"
        assert loc.media_type == "image"

    def test_to_dict(self):
        loc = StubLocalizer(name="test-loc", media_type="video")
        d = loc.to_dict()
        assert d["name"] == "test-loc"
        assert d["media_type"] == "video"


# ---------------------------------------------------------------------------
# Extractor as Processor
# ---------------------------------------------------------------------------


class TestExtractorAsProcessor:
    def test_process_delegates_to_extract(self):
        hits = [{"confidence": 0.9, "label": "cat"}]
        ext = StubExtractor(results=hits)
        assert ext.process({}) == hits

    def test_process_returns_empty_list(self):
        ext = StubExtractor(results=[])
        assert ext.process({}) == []

    def test_extractor_to_dict(self):
        ext = StubExtractor(name="my-ext", media_type="image")
        d = ext.to_dict()
        assert d["name"] == "my-ext"
        assert d["media_type"] == "image"
