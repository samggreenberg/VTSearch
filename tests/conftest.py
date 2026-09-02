from pathlib import Path
from unittest.mock import patch

import pytest

import vtscore.config as config
from tests_shared.embedding_stubs import fake_embed_audio as _fake_embed_audio, make_stub_embedding_models_fixture
from tests_shared.pytest_plumbing import add_group_markers, print_summary_and_exit
from tests_shared.state_reset import (
    allow_test_tmp_paths as _allow_test_tmp_paths,  # noqa: F401  (autouse fixture)
    freeze_startup_heap,
    install_startup_contexts,
    pin_training_budget,
    reset_shared_state,
)

# Everything below that the two suites share — the group-marker hook, the fake
# embedders, the tmp-path widener, the embedder stub fixture, the bulk of the
# reset fixture and the end-of-run summary printer — lives in ``tests_shared``.
# What stays here is app-tier only: the Flask ``client``, settings isolation,
# the Angular bundle fixture and the autorun-processor reset.


def pytest_collection_modifyitems(items, config):
    """Auto-assign group markers based on the test file's parent directory."""
    add_group_markers(items, root_dir_name="tests")


pin_training_budget()

# Patch embed_audio_file so init_medias() never triggers CLAP model loading.
_EMBED_AUDIO_FILE_TARGET = "tests.fixtures.medias.embed_audio_file"
_patch_embed_audio = patch(_EMBED_AUDIO_FILE_TARGET, side_effect=_fake_embed_audio)
_patch_embed_audio.start()

install_startup_contexts()

# Importing ``app`` registers the Flask routes and builds the module-level
# ``app_module.app`` the ``client`` fixture below serves.  Test modules do NOT
# need to repeat this import: conftest is imported first, so ``app`` is already
# in ``sys.modules`` (and its import side effects have already run) by the time
# any test module is collected.
import app as app_module

from tests.fixtures.medias import init_medias
from vtscore.embedding import initialize_models
from vtsearch.state import medias

# Initialize models and medias
initialize_models()
init_medias()

# Save the test medias so we can replay them into each test's fresh context.
_test_medias_snapshot = {k: dict(v) for k, v in medias.items()}

freeze_startup_heap()

# Stop the module-level patch (init_medias is done); the per-test autouse
# fixture below re-applies the patches for every test so that /api/sort and
# other routes that call embed_text don't trigger CLAP loading either.
_patch_embed_audio.stop()

# Stub every registered media type and embedder for the whole session.
_stub_embedding_models = make_stub_embedding_models_fixture(_EMBED_AUDIO_FILE_TARGET)


# ---------------------------------------------------------------------------
# Angular bundle (static/) on demand, safe under pytest-xdist.
#
# `./run-tests.sh` builds the bundle before pytest starts, so this is a no-op
# there.  Bare `pytest tests/ -n auto` invocations on a fresh clone need the
# bundle built exactly once: without cross-worker coordination, the worker
# that collects tests/core/test_frontend.py would build while workers running
# tests/api/test_dashboard.py read static/ before it exists (and several
# workers could kick off concurrent ~16s npm builds).  A cross-process flock
# (vtscore.io.file_lock) elects one builder; the others block until the build
# finishes, then see the bundle and return.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATIC_DIR = _REPO_ROOT / "static"
_FRONTEND_DIR = _REPO_ROOT / "frontend"


def _angular_bundle_built() -> bool:
    return (_STATIC_DIR / "index.html").exists() and (_STATIC_DIR / "main.js").exists()


@pytest.fixture(scope="session")
def angular_bundle() -> None:
    """Ensure the Angular bundle exists in ``static/`` (build once per run).

    Tests that serve the SPA shell or read bundle artefacts request this
    fixture.  Skips (cached for the whole session) when the bundle is absent
    and cannot be built here (no npm / node_modules / failing build).
    """
    import shutil
    import subprocess

    if _angular_bundle_built():
        return
    npm = shutil.which("npm")
    if npm is None or not (_FRONTEND_DIR / "node_modules").exists():
        pytest.skip(
            "Angular bundle not built and cannot build it here "
            f"(npm={'found' if npm else 'missing'}, "
            f"node_modules={'present' if (_FRONTEND_DIR / 'node_modules').exists() else 'missing'}). "
            "Run: cd frontend && npm install && npm run build:prod"
        )

    from vtscore.io import file_lock

    # flock-based: releases automatically if the building process dies, so a
    # crashed builder can't wedge the other workers.
    with file_lock(config.DATA_DIR / "angular-bundle-build"):
        if _angular_bundle_built():
            return  # another xdist worker built it while we waited
        try:
            subprocess.run(  # noqa: S603  # npm resolved via shutil.which, args are constant
                [npm, "run", "build:prod"],
                cwd=_FRONTEND_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            pytest.skip(f"Angular build failed: {exc.stderr.strip() or exc.stdout.strip() or exc}")
    if not _angular_bundle_built():
        pytest.skip("Angular build completed but static/main.js or static/index.html still missing")


import vtscore.state.core as _core
from vtsearch.auth import DefaultLoginProvider as _DefaultLoginProvider, set_login_provider as _set_login_provider


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all mutable global state before each test.

    The library-tier half (contexts, medias replay, progress trackers,
    registries, the lru_cache and TTL-cache clears) lives in
    :func:`tests_shared.state_reset.reset_shared_state`, which the library
    suite calls too.  Only the app-tier extras are spelled out here.
    """
    from vtsearch.autorun_processors import clear_all_autorun

    clear_all_autorun()

    reset_shared_state(_test_medias_snapshot)

    _set_login_provider(_DefaultLoginProvider())

    # ``test_torch_config.py`` reloads ``vtscore.config`` to test env-var
    # behaviour, which wipes the module-level ``_core_config_builder``
    # installed at app startup.  Re-register defensively so any later test
    # that calls ``CoreConfig.from_settings()`` (e.g. via ``get_inclusion()``)
    # still has a backing implementation.
    from vtsearch.shim import register_app_config_builder

    register_app_config_builder()


class _MergedSettingsPath:
    """Path-like helper that bridges the two settings tiers in tests.

    Settings used to live in a single ``data/settings.json`` file, and
    test code reads/writes it via ``isolated_settings.read_text()`` /
    ``.write_text(...)``.  With the two-tier layout, per-user keys live
    in ``users/default/user_settings.json`` while server-tier keys stay
    in ``settings.json``. This wrapper preserves the legacy API by:

    * ``read_text()``: returns the merged JSON of both files (per-user
      values win over server values on key collisions, which matches the
      runtime behaviour of ``settings.get_all()``).
    * ``write_text(text)``: parses *text* as JSON, splits the keys by
      tier, and writes each subset to the right file. Non-JSON text is
      treated as a "corrupt the file" probe and goes to the server file
      so the existing corruption tests still see invalid JSON on load.
    * ``exists()`` / ``__fspath__()`` / ``__str__()`` resolve to the
      server-tier file for the rare tests that pass the fixture to APIs
      expecting a real path.
    """

    def __init__(self, server_path, user_default_path) -> None:
        self._server = server_path
        self._user = user_default_path

    def read_text(self, *args, **kwargs) -> str:
        import json as _json

        merged: dict = {}
        for p in (self._server, self._user):
            if p.exists():
                try:
                    data = _json.loads(p.read_text(*args, **kwargs))
                    if isinstance(data, dict):
                        merged.update(data)
                except Exception:
                    pass
        return _json.dumps(merged)

    def write_text(self, content: str, *args, **kwargs) -> int:
        import json as _json
        from vtsearch import settings as _settings

        try:
            data = _json.loads(content)
        except Exception:
            # Non-JSON content: simulate a corrupt server-tier file.
            self._server.parent.mkdir(parents=True, exist_ok=True)
            return self._server.write_text(content, *args, **kwargs)

        if not isinstance(data, dict):
            self._server.parent.mkdir(parents=True, exist_ok=True)
            return self._server.write_text(content, *args, **kwargs)

        server_data = {k: v for k, v in data.items() if k in _settings._SERVER_KEYS}
        user_data = {k: v for k, v in data.items() if k not in _settings._SERVER_KEYS}

        self._server.parent.mkdir(parents=True, exist_ok=True)
        self._server.write_text(_json.dumps(server_data, indent=2) + "\n")
        self._user.parent.mkdir(parents=True, exist_ok=True)
        self._user.write_text(_json.dumps(user_data, indent=2) + "\n")
        # Force the in-memory caches to re-read on next access.
        _settings.reset()
        return len(content)

    def exists(self) -> bool:
        return self._server.exists() or self._user.exists()

    def __fspath__(self) -> str:
        return str(self._server)

    def __str__(self) -> str:
        return str(self._server)


@pytest.fixture(autouse=True)
def _no_signpost_pipeline(monkeypatch):
    """Keep the Toponymy signpost pipeline out of the app-tier suite.

    The projection build paths run signpost prep best-effort whenever the
    ``toponymy`` library is installed, which would drag a real (numba-heavy)
    clustering fit into otherwise-fast build tests — and make the suite's
    behavior depend on whether the optional library is present.  Reporting it
    unavailable makes every build path skip prep deterministically; the prep
    glue itself is covered by ``tests_lib/projection`` with the fit seam
    stubbed, and the real library by the ``slow``-marked smoke test.  Tests
    that exercise the label-serving paths re-patch what they need.
    """
    from vtscore.projection import signpost_prep

    # The build paths gate on require_signposting; the serve / signature paths
    # on the quiet probe.  Stub both off so every app-tier path skips prep
    # deterministically — and stubbing require_signposting (rather than letting
    # it compute False) also keeps its one-time "install broken" error out of
    # the suite, since nothing is actually broken here.
    monkeypatch.setattr(signpost_prep, "signposting_available", lambda: False)
    monkeypatch.setattr(signpost_prep, "require_signposting", lambda: False)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect both settings tiers to a temp directory for each test.

    Without this, tests that write settings (inclusion, volume, etc.) would mutate the shared ``data/settings.json`` and the
    per-user files under ``data/<user>/user_settings.json``, leaking
    values into subsequent tests that lazy-load from those files.

    Yields a :class:`_MergedSettingsPath` so legacy tests that call
    ``isolated_settings.read_text()`` / ``.write_text(...)`` continue to
    see a merged view of both tiers.
    """
    from vtsearch import settings as settings_mod

    test_settings_path = tmp_path / "settings.json"
    user_default_path = tmp_path / "users" / "default" / "user_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", test_settings_path)
    # Redirect per-user settings files under tmp_path/users/<username>/.
    settings_mod.set_user_data_dir_override(tmp_path / "users")
    settings_mod.reset()

    # Also redirect dataset and detector registries to temp paths
    from vtscore.datasets import registry as ds_reg_mod
    from vtscore.detectors import registry as det_reg_mod

    monkeypatch.setattr(ds_reg_mod, "REGISTRY_PATH", tmp_path / "dataset_registry.json")
    monkeypatch.setattr(det_reg_mod, "REGISTRY_PATH", tmp_path / "detector_registry.json")

    # Redirect storage directories to temp paths via settings
    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved_datasets"))
    settings_mod.set_detectors_dir(str(tmp_path / "detectors"))

    ds_reg_mod.reset_for_tests()
    det_reg_mod.reset_for_tests()

    yield _MergedSettingsPath(test_settings_path, user_default_path)
    settings_mod.reset()
    settings_mod.set_user_data_dir_override(None)
    ds_reg_mod.reset_for_tests()
    det_reg_mod.reset_for_tests()


def _active_id_for_tests(thread_local_getter, registry_getter) -> str | None:
    """Return the active context's id, but only if it's still registered.

    Reads the *thread-local* context directly rather than going through
    ``get_active_*_context()``. The high-level resolver also consults
    Flask's per-request ``g._*_context``, which only returns a sensible
    value while a request handler is actively running (the resolver
    gates on ``g._vts_in_request_handler``). Between requests in a
    ``with app.test_client() as c:`` block, ``g`` exists but is no
    longer "active". Either path would work here, but going straight
    to the thread-local avoids paying the cost of a noop g lookup on
    every test request.

    Skips:

    - ``None`` contexts and ones whose id is an empty string or a
      ``__sentinel__`` marker.
    - Ids whose entry has been removed from the registry (e.g. a
      previous request unloaded the detector but the test thread's
      thread-local still points to the now-orphaned object. Injecting
      a stale id would cause ``before_request`` to stash
      ``_unloaded_detector_id`` and turn every proxy access into a
      409, which is exactly the silent-mistarget surface H34 is
      trying to avoid in *production*; in the test wrapper we
      want a clean "no header sent" so the route hits the proper
      no-active-context path instead.
    """
    try:
        ctx = thread_local_getter()
    except Exception:
        return None
    if ctx is None:
        return None
    cid = getattr(ctx, "dataset_id", None) or getattr(ctx, "detector_id", None)
    if not cid or cid.startswith("__"):
        return None
    if registry_getter(cid) is None:
        return None
    return cid


def _install_active_context_headers(c):
    """Make the test client behave like Angular's ``activeContextInterceptor``.

    Production routes that mutate vote / dataset state require ``X-Dataset-Id``
    and ``X-Detector-Id`` headers (logical-bug-audit H34). In production those
    headers are attached transparently by ``activeContextInterceptor`` in the
    Angular frontend. The Flask test client has no such interceptor, so this
    wrapper inspects the thread-local active context on each ``client.open()``
    call and fills in the headers when they're not already provided. Tests
    that need to exercise the header-absent code path can pass
    ``headers={"X-Dataset-Id": ""}`` / ``"X-Detector-Id": ""`` to suppress
    auto-injection for that key while still preserving the empty-string value
    (which fails the ``bool(...)`` check the decorators apply).
    """
    from werkzeug.datastructures import Headers

    original_open = c.open

    def _open(*args, **kwargs):
        path = args[0] if args else kwargs.get("path", "")
        if isinstance(path, str) and path.startswith("/api/"):
            headers = kwargs.get("headers")
            hdrs = Headers(headers) if headers is not None else Headers()
            if "X-Dataset-Id" not in hdrs:
                ds_id = _active_id_for_tests(_core.get_thread_dataset_context, _core.get_context)
                if ds_id:
                    hdrs.add("X-Dataset-Id", ds_id)
            if "X-Detector-Id" not in hdrs:
                det_id = _active_id_for_tests(_core.get_thread_detector_context, _core.get_detector_context)
                if det_id:
                    hdrs.add("X-Detector-Id", det_id)
            kwargs["headers"] = hdrs
        return original_open(*args, **kwargs)

    c.open = _open


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        _install_active_context_headers(c)
        yield c


def _wait_for_job(job_manager, *, timeout: float = 30.0) -> None:
    """Block until the running job (and any coalesced pending follow-up) finish.

    With the coalescing job manager, a ``start()`` issued while a job is
    running ends up in the pending slot and gets promoted automatically
    when the runner finishes.  Tests that issue several requests need to
    wait for the whole chain to drain, not just the first job.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        job = job_manager.current()
        if job is None:
            return
        if job.status in ("running", "pending"):
            job.done_event.wait(timeout=0.05)
            continue
        # Current is finished.  Yield briefly so any pending promotion
        # spawned from _run can replace current before we re-check.
        _time.sleep(0.01)
        follow = job_manager.current()
        if follow is None or follow.job_id == job.job_id:
            return
    raise TimeoutError(f"Job manager did not drain within {timeout}s")


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Print the run summary and force-exit (see ``tests_shared``)."""
    print_summary_and_exit(config, getattr(config, "_vtsearch_exitstatus", 0))


def pytest_sessionfinish(session, exitstatus):
    """Stash the exit status so pytest_unconfigure can use it."""
    session.config._vtsearch_exitstatus = exitstatus
