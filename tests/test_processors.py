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

    def detect(self, media):
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

    def localize(self, media):
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

    def extract(self, media):
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
        det = StubDetector(name="test-det", media_type="text")
        d = det.to_dict()
        assert d["name"] == "test-det"
        assert d["media_type"] == "text"


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


# ---------------------------------------------------------------------------
# OCR Extractor
# ---------------------------------------------------------------------------


class TestOCRExtractor:
    def test_identity(self):
        from vtsearch.media.image.ocr_extractor import OCRExtractor

        ext = OCRExtractor(name="test-ocr", language="en", threshold=0.6)
        assert ext.name == "test-ocr"
        assert ext.media_type == "image"
        assert ext.language == "en"
        assert ext.threshold == 0.6
        assert isinstance(ext, Extractor)

    def test_to_dict(self):
        from vtsearch.media.image.ocr_extractor import OCRExtractor

        ext = OCRExtractor(name="ocr1", language="fr", threshold=0.7)
        d = ext.to_dict()
        assert d["name"] == "ocr1"
        assert d["media_type"] == "image"
        assert d["extractor_type"] == "ocr"
        assert d["config"]["language"] == "fr"
        assert d["config"]["threshold"] == 0.7

    def test_from_config(self):
        from vtsearch.media.image.ocr_extractor import OCRExtractor

        config = {"language": "de", "threshold": 0.8}
        ext = OCRExtractor.from_config("rebuilt", config)
        assert ext.name == "rebuilt"
        assert ext.language == "de"
        assert ext.threshold == 0.8

    def test_from_config_defaults(self):
        from vtsearch.media.image.ocr_extractor import OCRExtractor

        ext = OCRExtractor.from_config("default", {})
        assert ext.language == "en"
        assert ext.threshold == 0.5

    def test_extract_returns_empty_when_no_clip_bytes(self):
        from vtsearch.media.image.ocr_extractor import OCRExtractor

        ext = OCRExtractor(name="test", language="en")
        ext._model = True  # Skip model loading
        result = ext.extract({"id": 1})
        assert result == []


# ---------------------------------------------------------------------------
# Speech Extractor
# ---------------------------------------------------------------------------


class TestSpeechExtractor:
    def test_identity(self):
        from vtsearch.media.audio.speech_extractor import SpeechExtractor

        ext = SpeechExtractor(name="test-speech", model_size="tiny", language="en")
        assert ext.name == "test-speech"
        assert ext.media_type == "audio"
        assert ext.model_size == "tiny"
        assert ext.language == "en"
        assert isinstance(ext, Extractor)

    def test_to_dict(self):
        from vtsearch.media.audio.speech_extractor import SpeechExtractor

        ext = SpeechExtractor(name="speech1", model_size="base", language="fr")
        d = ext.to_dict()
        assert d["name"] == "speech1"
        assert d["media_type"] == "audio"
        assert d["extractor_type"] == "speech"
        assert d["config"]["model_size"] == "base"
        assert d["config"]["language"] == "fr"

    def test_from_config(self):
        from vtsearch.media.audio.speech_extractor import SpeechExtractor

        config = {"model_size": "small", "language": "es"}
        ext = SpeechExtractor.from_config("rebuilt", config)
        assert ext.name == "rebuilt"
        assert ext.model_size == "small"
        assert ext.language == "es"

    def test_from_config_defaults(self):
        from vtsearch.media.audio.speech_extractor import SpeechExtractor

        ext = SpeechExtractor.from_config("default", {})
        assert ext.model_size == "tiny"
        assert ext.language is None

    def test_extract_returns_empty_when_no_clip_bytes(self):
        from vtsearch.media.audio.speech_extractor import SpeechExtractor

        ext = SpeechExtractor(name="test")
        ext._model = True  # Skip model loading
        result = ext.extract({"id": 1})
        assert result == []


# ---------------------------------------------------------------------------
# Face Localizer
# ---------------------------------------------------------------------------


class TestFaceLocalizer:
    def test_identity(self):
        from vtsearch.media.image.face_localizer import FaceLocalizer

        loc = FaceLocalizer(name="test-face", threshold=0.6, model_selection=0)
        assert loc.name == "test-face"
        assert loc.media_type == "image"
        assert loc.threshold == 0.6
        assert loc.model_selection == 0
        assert isinstance(loc, Localizer)

    def test_to_dict(self):
        from vtsearch.media.image.face_localizer import FaceLocalizer

        loc = FaceLocalizer(name="face1", threshold=0.7, model_selection=1)
        d = loc.to_dict()
        assert d["name"] == "face1"
        assert d["media_type"] == "image"
        assert d["localizer_type"] == "face"
        assert d["config"]["threshold"] == 0.7
        assert d["config"]["model_selection"] == 1

    def test_from_config(self):
        from vtsearch.media.image.face_localizer import FaceLocalizer

        config = {"threshold": 0.3, "model_selection": 0}
        loc = FaceLocalizer.from_config("rebuilt", config)
        assert loc.name == "rebuilt"
        assert loc.threshold == 0.3
        assert loc.model_selection == 0

    def test_from_config_defaults(self):
        from vtsearch.media.image.face_localizer import FaceLocalizer

        loc = FaceLocalizer.from_config("default", {})
        assert loc.threshold == 0.5
        assert loc.model_selection == 1

    def test_localize_returns_empty_when_no_clip_bytes(self):
        from vtsearch.media.image.face_localizer import FaceLocalizer

        loc = FaceLocalizer(name="test")
        loc._detector = True  # Skip model loading
        result = loc.localize({"id": 1})
        assert result == []


# ---------------------------------------------------------------------------
# Autorun localizer state management
# ---------------------------------------------------------------------------


class TestAutorunLocalizerState:
    def test_add_and_get(self):
        from vtsearch.utils.state import add_autorun_localizer, get_autorun_localizers

        add_autorun_localizer("face1", "face", "image", {"threshold": 0.5})
        locs = get_autorun_localizers()
        assert "face1" in locs
        assert locs["face1"]["localizer_type"] == "face"
        assert locs["face1"]["media_type"] == "image"

    def test_remove(self):
        from vtsearch.utils.state import add_autorun_localizer, get_autorun_localizers, remove_autorun_localizer

        add_autorun_localizer("face1", "face", "image", {"threshold": 0.5})
        assert remove_autorun_localizer("face1") is True
        assert remove_autorun_localizer("face1") is False
        assert get_autorun_localizers() == {}

    def test_rename(self):
        from vtsearch.utils.state import add_autorun_localizer, get_autorun_localizers, rename_autorun_localizer

        add_autorun_localizer("old", "face", "image", {"threshold": 0.5})
        assert rename_autorun_localizer("old", "new") is True
        locs = get_autorun_localizers()
        assert "new" in locs
        assert "old" not in locs

    def test_get_by_media(self):
        from vtsearch.utils.state import add_autorun_localizer, get_autorun_localizers_by_media

        add_autorun_localizer("face1", "face", "image", {"threshold": 0.5})
        add_autorun_localizer("audio_loc", "face", "audio", {"threshold": 0.5})
        image_locs = get_autorun_localizers_by_media("image")
        assert "face1" in image_locs
        assert "audio_loc" not in image_locs


# ---------------------------------------------------------------------------
# Pregen processors API route
# ---------------------------------------------------------------------------


class TestPregenProcessorsRoute:
    def test_list_pregen_processors(self, client):
        res = client.get("/api/pregen-processors")
        assert res.status_code == 200
        data = res.get_json()
        assert "processors" in data
        names = [p["name"] for p in data["processors"]]
        assert "OCR (PaddleOCR)" in names
        assert "Speech (Whisper Tiny)" in names
        assert "Face (MediaPipe)" in names

    def test_add_pregen_processors(self, client):
        res = client.post("/api/pregen-processors/add")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert len(data["added"]) == 3
        assert "OCR (PaddleOCR)" in data["added"]
        assert "Speech (Whisper Tiny)" in data["added"]
        assert "Face (MediaPipe)" in data["added"]

    def test_pregen_adds_to_autorun(self, client):
        from vtsearch.utils.state import autorun_extractors, autorun_localizers

        client.post("/api/pregen-processors/add")
        assert "OCR (PaddleOCR)" in autorun_extractors
        assert "Speech (Whisper Tiny)" in autorun_extractors
        assert "Face (MediaPipe)" in autorun_localizers


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestExtractorFactoryRegistration:
    def test_ocr_factory_registered(self):
        from vtsearch.routes.detectors import _EXTRACTOR_FACTORIES, _ensure_extractor_factories

        _EXTRACTOR_FACTORIES.clear()
        _ensure_extractor_factories()
        assert "ocr" in _EXTRACTOR_FACTORIES
        assert "speech" in _EXTRACTOR_FACTORIES
        assert "image_class" in _EXTRACTOR_FACTORIES

    def test_build_ocr_extractor(self):
        from vtsearch.routes.detectors import _EXTRACTOR_FACTORIES, _build_extractor, _ensure_extractor_factories

        _EXTRACTOR_FACTORIES.clear()
        _ensure_extractor_factories()
        ext = _build_extractor("test-ocr", "ocr", {"language": "en", "threshold": 0.5})
        assert ext.name == "test-ocr"
        assert ext.media_type == "image"

    def test_build_speech_extractor(self):
        from vtsearch.routes.detectors import _EXTRACTOR_FACTORIES, _build_extractor, _ensure_extractor_factories

        _EXTRACTOR_FACTORIES.clear()
        _ensure_extractor_factories()
        ext = _build_extractor("test-speech", "speech", {"model_size": "tiny"})
        assert ext.name == "test-speech"
        assert ext.media_type == "audio"


class TestLocalizerFactoryRegistration:
    def test_face_factory_registered(self):
        from vtsearch.routes.detectors import _LOCALIZER_FACTORIES, _ensure_localizer_factories

        _LOCALIZER_FACTORIES.clear()
        _ensure_localizer_factories()
        assert "face" in _LOCALIZER_FACTORIES

    def test_build_face_localizer(self):
        from vtsearch.routes.detectors import _LOCALIZER_FACTORIES, _build_localizer, _ensure_localizer_factories

        _LOCALIZER_FACTORIES.clear()
        _ensure_localizer_factories()
        loc = _build_localizer("test-face", "face", {"threshold": 0.5})
        assert loc.name == "test-face"
        assert loc.media_type == "image"
