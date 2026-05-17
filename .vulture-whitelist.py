"""Whitelist for vulture's dead-code detector.

Vulture finds defined-but-never-referenced names. This file lists symbols
that VTSearch DOES use, but only reflectively — so static analysis can't
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
  vulture doesn't follow back to a call site — the decorator filter
  covers ``@*.route``, ``@*.before_request``, ``@*.errorhandler``,
  ``@bp.*``, ``@app.*``, and the rest of the Flask lifecycle hooks.
* Test-only names: ``test_*`` and ``Test*`` are pytest-discovered, the
  ``setup_method``/``teardown_method``/``setup_class``/``teardown_class``
  hooks are pytest fixture lifecycle, ``pytest_*`` and ``pytestmark`` are
  framework reserved.
* Python protocol dunders (``__enter__``, ``__exit__``, ``__package__``)
  are called by the runtime, not by user code — vulture sees the
  assignment (e.g. on a ``MagicMock`` instance for a context-manager
  test) but no read.

Whitelist entries below cover the remaining individual symbols that
vulture flags but are actually used reflectively or as the public API
surface of a module.
"""

# ---------------------------------------------------------------------------
# Plugin sentinels — each ``<FAMILY> = SomeClass`` line at the bottom of a
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
# Public-API forwarders in ``vtsearch.concurrency.progress`` that mirror
# their ``update_<tracker>_progress`` partners. The corresponding
# trackers (``sort_progress``, ``find_progress``, ``dataset_progress``)
# are imported and read directly in tests / routes; the helper wrappers
# stay for API symmetry and are documented in CLAUDE.md.
# ---------------------------------------------------------------------------
check_dataset_cancelled  # noqa: F821
get_sort_progress  # noqa: F821
get_find_progress  # noqa: F821

# ---------------------------------------------------------------------------
# Public module constants — referenced by callers via ``module.NAME`` or
# settings lookup, which vulture treats as the only assignment.
# ---------------------------------------------------------------------------
SAVED_DATASETS_DIR  # noqa: F821 — vtsearch.datasets.registry default dir
DETECTORS_DIR  # noqa: F821 — vtsearch.detectors.store default dir
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB  # noqa: F821 — downloader size budget constant

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
find_by_pkl_path  # noqa: F821 — vtsearch.datasets.registry
recreate_model_at_time  # noqa: F821 — vtsearch.detectors.labeling_progress
update_cache_for_cid  # noqa: F821 — vtsearch.detectors.labelset_training
collect_media_origins  # noqa: F821 — vtsearch.detectors.training
train_detector_from_origins  # noqa: F821 — vtsearch.detectors.training

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
