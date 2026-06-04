"""Whitelist for vulture's dead-code detector.

Vulture finds defined-but-never-referenced names. This file lists symbols
that VTSearch DOES use, but only reflectively - so static analysis can't
see the reference. Run vulture with this file as an extra argument:

    vulture vtsearch/ app.py tests/ .vulture-whitelist.py --min-confidence 60 \\
      --exclude '*/vtsearch/schemas/*,*/vtsearch/settings_models.py' \\
      --ignore-decorators '@*.route,@*.before_request,@*.after_request,@*.errorhandler,@*.teardown_request,@*.context_processor,@bp.*,@app.*,@pytest.fixture,@pytest.mark.*,@fixture,@*.fixture' \\
      --ignore-names 'Meta,model_config,_keys_to_ignore_on_load_unexpected,test_*,Test*,setup_method,teardown_method,setup_class,teardown_class,pytest_*,pytestmark,__enter__,__exit__,__package__'

Why the excludes / ignores:

* ``vtsearch/schemas/*`` and ``vtsearch/settings_models.py`` are pure
  framework-managed declaration files. Every field assignment in a
  marshmallow ``Schema`` subclass or a pydantic ``BaseModel`` looks
  "unused" to vulture because both frameworks collect fields via
  metaclass at class-creation time. There is no way to tell vulture
  about that linkage short of listing every field by hand, so we just
  skip the directories. Dead schemas, if any, will surface as unused
  imports in the route files that consume them.
* ``--ignore-names Meta,model_config`` covers the same metaclass-managed
  inner-class / config-attribute patterns for any stragglers that aren't
  in the excluded paths.
* ``--ignore-names _keys_to_ignore_on_load_unexpected`` is HuggingFace's
  convention for telling ``transformers.PreTrainedModel`` to skip
  certain weight keys when loading; the attribute is read by the base
  class via reflection.
* Flask route handlers are decorated with ``@bp.route(...)`` etc., which
  vulture doesn't follow back to a call site - the decorator filter
  covers ``@*.route``, ``@*.before_request``, ``@*.errorhandler``,
  ``@bp.*``, ``@app.*``, and the rest of the Flask lifecycle hooks.
* Test-only names: ``test_*`` and ``Test*`` are pytest-discovered, the
  ``setup_method``/``teardown_method``/``setup_class``/``teardown_class``
  hooks are pytest fixture lifecycle, ``pytest_*`` and ``pytestmark`` are
  framework reserved.
* Python protocol dunders (``__enter__``, ``__exit__``, ``__package__``)
  are called by the runtime, not by user code - vulture sees the
  assignment (e.g. on a ``MagicMock`` instance for a context-manager
  test) but no read.

Whitelist entries below cover the remaining individual symbols that
vulture flags but are actually used reflectively or as the public API
surface of a module.
"""

# ---------------------------------------------------------------------------
# Plugin sentinels - each ``<FAMILY> = SomeClass`` line at the bottom of a
# plugin module is discovered by the ``PluginRegistry`` scanner via the
# matching attribute name. Vulture sees the assignment but no reference;
# discovery happens at import time through ``getattr``.
# ---------------------------------------------------------------------------
IMPORTER  # noqa: F821
EXPORTER  # noqa: F821
CONVERTER  # noqa: F821
SOURCE  # noqa: F821
PROCESSOR  # noqa: F821
PICKER  # noqa: F821
MEDIA_TYPE  # noqa: F821
SETTINGS_IMPORTER  # noqa: F821
SETTINGS_EXPORTER  # noqa: F821
SETTINGS_SOURCE  # noqa: F821
LABEL_IMPORTER  # noqa: F821
LABELSET_SOURCE  # noqa: F821
PickerView  # noqa: F821

# ---------------------------------------------------------------------------
# argparse.Action.__call__ requires the ``option_string`` parameter even
# when the action ignores it.
# ---------------------------------------------------------------------------
option_string  # noqa: F821

# ---------------------------------------------------------------------------
# Flask reads ``app.secret_key`` via attribute access for session signing.
# Setting it is the side-effecting "use".
# ---------------------------------------------------------------------------
secret_key  # noqa: F821

# ---------------------------------------------------------------------------
# Public-API forwarders in ``vtscore.concurrency.progress`` that mirror
# their ``update_<tracker>_progress`` partners. The corresponding
# trackers (``sort_progress``, ``find_progress``, ``dataset_progress``)
# are imported and read directly in tests / routes; the helper wrappers
# stay for API symmetry and are documented in CLAUDE.md.
# ---------------------------------------------------------------------------
check_dataset_cancelled  # noqa: F821
get_sort_progress  # noqa: F821
get_find_progress  # noqa: F821

# ---------------------------------------------------------------------------
# Public module constants - referenced by callers via ``module.NAME`` or
# settings lookup, which vulture treats as the only assignment.
# ---------------------------------------------------------------------------
SAVED_DATASETS_DIR  # noqa: F821 - vtscore.datasets.registry default dir
DETECTORS_DIR  # noqa: F821 - vtscore.detectors.store default dir
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB  # noqa: F821 - downloader size budget constant

# ---------------------------------------------------------------------------
# Hardware-derived defaults imported and called from the excluded
# ``vtsearch/settings_models.py`` (pydantic ``default_factory`` for the
# two ``max_concurrent_dataset_*`` keys). The exclude hides those uses
# from vulture; the functions themselves are real.
# ---------------------------------------------------------------------------
default_concurrent_downloads  # noqa: F821
default_concurrent_embeddings  # noqa: F821

# ---------------------------------------------------------------------------
# Public APIs exposed for external consumers (CLI scripts, future tests,
# extension authors). Documented in CLAUDE.md but not currently called
# from within vtsearch/ or tests/.
# ---------------------------------------------------------------------------
find_by_pkl_path  # noqa: F821 - vtscore.datasets.registry
recreate_model_at_time  # noqa: F821 - vtscore.detectors.labeling_progress
update_cache_for_cid  # noqa: F821 - vtscore.detectors.labelset_training
collect_media_origins  # noqa: F821 - vtscore.detectors.training
train_detector_from_origins  # noqa: F821 - vtscore.detectors.training

# ---------------------------------------------------------------------------
# Public context managers exported from ``vtsearch.state`` for callers
# that need explicit, scoped switching of the active dataset/detector
# without going through the per-request middleware.
# ---------------------------------------------------------------------------
with_dataset_context  # noqa: F821
with_detector_context  # noqa: F821

# ---------------------------------------------------------------------------
# TYPE_CHECKING-only accessor stubs in ``vtsearch.settings``. The actual
# runtime functions are generated by ``make_accessors`` at module-bottom;
# the ``if TYPE_CHECKING:`` block exists so pyright can resolve
# ``settings.get_<key>()`` from other modules. Vulture sees the stub but
# can't connect it to the dynamic definition or the runtime callers.
# ---------------------------------------------------------------------------
get_audio_playing  # noqa: F821
get_swipe_animation  # noqa: F821
get_hide_autopilot  # noqa: F821
get_autopilot_resort_interval  # noqa: F821
get_browse_minimap_visible  # noqa: F821
get_browse_minimap_width  # noqa: F821
get_browse_minimap_height  # noqa: F821
set_dataset_max_age_days  # noqa: F821

# ---------------------------------------------------------------------------
# DetectorContext attributes written in ``vtsearch/routes/detectors/registry.py``
# and read in ``vtscore/detectors/dataset_sync.py`` /
# ``vtscore/detectors/label_sync.py``. The reader lives in ``vtscore/``,
# which the standard vulture command does not scan - so the write sites in
# ``vtsearch/`` look "unused" even though both attributes are real and
# heavily used.
# ---------------------------------------------------------------------------
cached_labelset_mtime  # noqa: F821
votes_dataset_id  # noqa: F821

# ---------------------------------------------------------------------------
# ``tests/helpers.py`` exports test helpers; ``make_wav_file`` is consumed
# by ``tests_lib/`` (the library-tier test suite mirrored from ``tests/``),
# which the default vulture command does not scan.
# ---------------------------------------------------------------------------
make_wav_file  # noqa: F821

# ---------------------------------------------------------------------------
# Stub / mock methods defined in test classes that override real abstract
# methods on ``MediaEmbedder`` (``_load_models_impl``, ``_embed_media_impl``)
# or mimic third-party API surfaces - ``PaddleOCR`` / ``ocr`` shadow
# ``paddleocr.PaddleOCR``, ``transcribe`` shadows the Whisper / Faster-Whisper
# API, ``_detect_speech_intervals`` shadows a VAD helper, ``_processor``
# shadows a HuggingFace processor attribute. The framework / production
# code calls them by name; vulture sees the test override but not the
# caller.
# ---------------------------------------------------------------------------
_load_models_impl  # noqa: F821
_embed_media_impl  # noqa: F821
_do_peek_version  # noqa: F821 - SyncSource subclass hook called by base class
PaddleOCR  # noqa: F821
ocr  # noqa: F821
transcribe  # noqa: F821
_detect_speech_intervals  # noqa: F821
_processor  # noqa: F821

# ---------------------------------------------------------------------------
# PEP 562 module-level ``__getattr__`` in ``vtsearch/settings.py`` exposes
# ``_SERVER_DEFAULTS`` / ``_USER_DEFAULTS`` / ``_DEFAULTS`` lazily without
# eagerly instantiating the pydantic defaults. Python calls it via the
# attribute-lookup protocol; vulture sees the def but no caller.
# ---------------------------------------------------------------------------
__getattr__  # noqa: F821

# ---------------------------------------------------------------------------
# Mock attributes that mimic the shape of third-party return values:
# ``conf`` / ``xyxy`` look like an ultralytics YOLO ``Result`` row;
# ``relative_bounding_box`` / ``location_data`` look like a MediaPipe
# detection. The production reader pulls them by name off the mock.
# ---------------------------------------------------------------------------
conf  # noqa: F821
xyxy  # noqa: F821
relative_bounding_box  # noqa: F821
location_data  # noqa: F821

# ---------------------------------------------------------------------------
# Test-internal attributes assigned for later access through fixtures or
# saved-state replay. ``_detector`` holds the unit under test inside
# clipper / processor tests; ``_extract_dir`` / ``dir_key`` mirror the
# corresponding production attribute on a fake media source; ``text_sort``
# / ``learned_sort`` are stub-result dataclass attributes for the eval CLI
# tests.
# ---------------------------------------------------------------------------
_detector  # noqa: F821
_extract_dir  # noqa: F821
dir_key  # noqa: F821
text_sort  # noqa: F821
learned_sort  # noqa: F821

# ---------------------------------------------------------------------------
# Pytest fixture parameters whose side effects are the point - the body
# never references the argument name, but the fixture must be requested
# by name in the signature. Vulture flags each parameter at 100 %
# confidence because the local name has no read.
# ---------------------------------------------------------------------------
stub_run_eval  # noqa: F821
stub_pipeline  # noqa: F821
stub_extractor_factory  # noqa: F821
stub_localizer_factory  # noqa: F821
stubbed_resolver  # noqa: F821
reset_state  # noqa: F821

# ---------------------------------------------------------------------------
# Test-local names that exist only to document the shape of an unpacked
# tuple, a captured Flask view function, or a fixture-yielded id. They
# are unused after binding by design.
# ---------------------------------------------------------------------------
view  # noqa: F821
first_chunk_medias  # noqa: F821
demo_id  # noqa: F821
folder_path_for_origin  # noqa: F821
output_type  # noqa: F821
