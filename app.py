import logging
import os
import warnings

# Limit threads to reduce memory overhead in constrained environments.  Native
# math libraries read these env vars during *their* import, which happens the
# moment torch / numpy / scipy are imported, so they have to be set before
# anything triggers that.  Mirrors ``vtscore.config.TORCH_THREADS`` but
# resolved inline to avoid importing ``vtscore.config`` (and therefore
# everything it transitively imports) this early.
_torch_threads = str(max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1"))))
os.environ["OMP_NUM_THREADS"] = _torch_threads
os.environ["MKL_NUM_THREADS"] = _torch_threads

# Configure structured logging: JSON lines by default with per-record
# request_id / dataset_id / detector_id / user fields. Override via:
#   VTSEARCH_LOG_LEVEL=INFO   (default WARNING)
#   VTSEARCH_LOG_FORMAT=text  (default json; switch to text for local dev)
from vtsearch.logging_config import new_request_id, setup_logging  # noqa: E402

setup_logging()

# All HF models we use are public, so no token is *required*.  Each
# from_pretrained() call passes token=hf_token(): the HF_TOKEN env var when
# set (authenticated requests sidestep the Hub's per-IP anonymous rate limits,
# which matter behind shared egress IPs like cluster NATs), else False to
# signal "anonymous on purpose" and suppress missing-token warnings.  The env
# var below disables *implicit* token pickup so the explicit pass in
# hf_token() stays the single path; the warnings filters are
# belt-and-suspenders in case any transitive HF code still warns.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Visual feedback for startup
print(f"⏳ Initializing VTSearch... (PID {os.getpid()})", flush=True)

from flask import Flask, g
from werkzeug.exceptions import MethodNotAllowed, NotFound

# Import refactored modules
from vtscore.state.core import DatasetNotLoadedError, DetectorNotLoadedError  # noqa: E402
from vtsearch.auth import get_login_provider  # noqa: E402
from vtscore.embedding import initialize_models, preload_predicted_embedders  # noqa: E402

# Install Flask-aware request-context resolvers on the (library-candidate)
# ``vtsearch.state`` core so its ``get_active_*_context()`` helpers can read
# the per-request dataset/detector context from ``flask.g`` without
# ``vtscore.state.core`` itself having to import Flask.  Also wire the
# library's "persist this" hooks (currently just last-embedder-per-media-
# type) to ``vtsearch.settings`` and install the app-side builder for
# ``CoreConfig.from_settings()``.  See ``vtscore/docs/architecture.md`` for
# the seam.
#
# This block runs BEFORE blueprint modules are imported so that any
# module-level code in a route that calls ``CoreConfig.from_settings()``
# (or any other shim-backed hook) finds the builder already installed.
# See logical-bug-audit M24.
from vtsearch.shim import (  # noqa: E402
    register_app_config_builder,
    register_app_persistence_hooks,
    register_app_plugin_families,
    register_flask_context_resolvers,
)

register_flask_context_resolvers()
register_app_persistence_hooks()
register_app_config_builder()
register_app_plugin_families()

from vtsearch.routes import (  # noqa: E402
    achievements_bp,
    auth_bp,
    detector_find_bp,
    detector_scoring_bp,
    detectors_crud_bp,
    detectors_labels_bp,
    detectors_registry_bp,
    embed_bp,
    eval_bp,
    events_bp,
    file_browser_bp,
    hf_auth_bp,
    health_bp,
    jobs_bp,
    labels_bp,
    media_server_bp,
    medias_bp,
    datasets_listings_bp,
    datasets_load_bp,
    datasets_registry_bp,
    datasets_staging_bp,
    datasets_status_bp,
    datasets_ui_bp,
    exporters_bp,
    label_importers_bp,
    main_bp,
    processors_crud_bp,
    processors_scoring_bp,
    projection_bp,
    sessions_bp,
    settings_bp,
    settings_io_bp,
    sorting_bp,
    sync_sources_bp,
)
from vtscore.media import set_progress_callback  # noqa: E402
from vtscore.concurrency.progress import update_progress

# Wire media types into the Flask app's progress reporting system.
# Without this call, media types use a silent no-op callback and can run
# standalone (e.g. in a CLI tool or notebook) without Flask.
set_progress_callback(update_progress)

app = Flask(__name__)
# Secret key for session cookies.  Read from VTSEARCH_SECRET_KEY env var
# if set; otherwise fall back to a dev-only default.  Production
# deployments should always set the env var.
app.secret_key = os.environ.get("VTSEARCH_SECRET_KEY", "vtsearch-dev-key-change-in-production")

# Optional cap on request body size (uploads).  ``MAX_UPLOAD_MB == 0`` leaves
# Flask's default of no limit in place; a positive value rejects oversized
# requests with HTTP 413 before they consume disk.
from vtscore.config import MAX_UPLOAD_MB as _MAX_UPLOAD_MB  # noqa: E402

if _MAX_UPLOAD_MB > 0:
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# OpenAPI / Swagger UI (flask-smorest)
# ---------------------------------------------------------------------------
# The API spec is served at /api/openapi.json and a browsable Swagger UI
# at /api/docs. See docs/plans/openapi-schema.md for the migration plan.
# Blueprints registered via ``api.register_blueprint`` contribute to the
# spec when they use ``flask_smorest.Blueprint``; plain Flask Blueprints
# still register but are absent from the spec until migrated.

from vtsearch import __version__ as _vtsearch_version  # noqa: E402

app.config["API_TITLE"] = "VTSearch API"
app.config["API_VERSION"] = _vtsearch_version
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/api"
app.config["OPENAPI_JSON_PATH"] = "openapi.json"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/docs"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

from flask_smorest import Api  # noqa: E402

from vtsearch.openapi_postprocess import (  # noqa: E402
    assign_operation_ids,
    normalize_unprocessable_response,
)

api = Api(app)


# flask-smorest / apispec doesn't populate ``operationId`` on its own, so
# ng-openapi-gen ends up synthesising client method names from path+method
# (``apiDetectorsRegistryDatasetIdRenamePut`` etc.). Wrap ``spec.to_dict``
# so both the live ``/api/openapi.json`` endpoint and ``dump_openapi.py``
# see operations tagged with their Flask view function name. The patch is
# safe to apply now because ``to_dict`` is only called lazily; blueprints
# registered later in this module are picked up automatically.
_apispec_to_dict = api.spec.to_dict


def _to_dict_with_operation_ids() -> dict:
    spec = _apispec_to_dict()
    assign_operation_ids(app, spec)
    normalize_unprocessable_response(spec)
    return spec


api.spec.to_dict = _to_dict_with_operation_ids


# ---------------------------------------------------------------------------
# Test-time attributes
# ---------------------------------------------------------------------------
# ``tests/conftest.py`` attaches a handful of helpers and proxies to this
# module so tests can use ``import app as app_module; app_module.medias`` etc.
# Declaring them here in a ``TYPE_CHECKING`` block lets pyright resolve the
# attribute accesses without changing runtime behaviour; the values are still
# only set by conftest and are absent in production. Same pattern as
# ``vtsearch/settings.py``'s dynamically generated accessors.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    NUM_MEDIAS: int
    generate_wav: Callable[..., bytes]
    train_and_score: Callable[..., Any]
    medias: dict[int, dict[str, Any]]
    good_votes: dict[int, None]
    bad_votes: dict[int, None]
    init_medias: Callable[[], None]


# ---------------------------------------------------------------------------
# Per-request user context
# ---------------------------------------------------------------------------


@app.before_request
def _set_request_id():
    """Assign a unique request id to every request.

    The id is exposed on ``g.request_id`` so log records produced during
    the request automatically carry it (via the structured-logging
    ContextFilter), and echoed back in the ``X-Request-Id`` response
    header so clients can quote it in bug reports. If the caller supplied
    their own ``X-Request-Id`` header we trust it (truncated to 64 chars
    to bound log line size), which lets gateways/load balancers propagate
    end-to-end trace ids.
    """
    from flask import request

    inbound = request.headers.get("X-Request-Id")
    g.request_id = inbound[:64] if inbound else new_request_id()


@app.before_request
def _set_user_context():
    """Populate ``g.user`` from the active LoginProvider on every request.

    A misbehaving provider must not be able to take down the server by
    raising from ``get_user``; swallow exceptions and fall back to
    ``"default"``.
    """
    from flask import request

    try:
        provider = get_login_provider()
        g.user = provider.get_user(request)
    except Exception:
        logging.getLogger(__name__).exception("Login provider get_user failed")
        g.user = "default"


# Endpoints whose handlers never read the dataset/detector proxies do not
# need the lock-taking state-sync in `_set_request_context` below. Keeping
# them off `_state_lock` stops a high-frequency poll (e.g. the jobs/active
# spinner feed) from piling up behind a long lock-holder and exhausting
# the worker's threads (2026-06-19: one stuck job parked 23 threads on
# the futex and froze the whole UI).
#
# `/api/events` is the SSE progress stream: a long-lived, read-only request
# that only subscribes to the *global* progress trackers and yields their
# snapshots. It needs no per-request vote rehydration, and gating it on
# `_state_lock` is self-defeating — while a long Find/load holds the worker
# busy, the EventSource's reconnect would block in this hook on the very
# lock the long job contends, so progress events never reach the client and
# the bar sits indeterminate. Exempting it keeps progress flowing during
# exactly the long operations the bar exists to report.
_STATE_SYNC_EXEMPT_PREFIXES = ("/api/jobs/active", "/api/events")


@app.before_request
def _set_request_context():
    """Resolve per-request dataset/detector context from HTTP headers.

    If the frontend sends ``X-Dataset-Id`` or ``X-Detector-Id``, the
    corresponding context is stashed on ``g`` so that proxy objects
    (``medias``, ``good_votes``, etc.) resolve to it for the duration of
    this request, without mutating global "active" state.

    When a header is absent the proxies fall back to the thread-local /
    empty context. When a header is **present but names an unloaded id**,
    the unloaded id is stashed on ``g`` so the resolver raises
    ``DatasetNotLoadedError`` / ``DetectorNotLoadedError`` on proxy
    access (mapped to 409). Routes that never touch the proxies still
    respond normally (see logical-bug-audit H16).

    Any failure here must not 500 every subsequent request; fall back to
    the default (empty) context.
    """
    from flask import request
    from vtscore.state.core import (
        get_context,
        get_detector_context,
    )

    # Pin a marker so the request-missing-context predicate can distinguish
    # "actively inside a route handler" from "Flask test client is still
    # preserving the popped request context for inspection". Cleared in
    # teardown_request below.
    g._vts_in_request_handler = True

    try:
        # Headers (Angular HttpClient interceptor) take priority, with query
        # params as fallback for browser-native requests (<img src>,
        # <audio src>, <video src>) that bypass Angular's interceptor.
        ds_id = request.headers.get("X-Dataset-Id") or request.args.get("dataset_id")
        if ds_id:
            ctx = get_context(ds_id)
            if ctx is not None:
                g._dataset_context = ctx
            else:
                # Header refers to a dataset that isn't loaded.  Stash the id
                # so the resolver raises DatasetNotLoadedError at proxy
                # access; silent fallback to the empty context hid stale
                # results from the client (logical-bug-audit H16).  Routes
                # that never touch the dataset proxies (registry listings,
                # auth, file browser, etc.) still respond normally.
                g._unloaded_dataset_id = ds_id

        detector_id = request.headers.get("X-Detector-Id") or request.args.get("detector_id")
        if detector_id:
            det_ctx = get_detector_context(detector_id)
            if det_ctx is not None:
                g._detector_context = det_ctx
            else:
                g._unloaded_detector_id = detector_id
    except Exception:
        logging.getLogger(__name__).exception("Request context resolution failed")

    # If the active (dataset, detector) pair has changed since the detector's
    # cid-keyed vote dicts were last derived, rehydrate them from the on-disk
    # labelset against the active dataset's medias.  Media ids are dataset-
    # specific, so without this the left-pane shows stale cids from the
    # previous dataset as if they were votes in the current one.
    #
    # Also drop the detector's cached MLP / per-label embedding cache when
    # the active dataset's embedder differs from the one the MLP was
    # trained on; scoring with a cross-space MLP either crashes (different
    # dim) or silently produces garbage labels (same dim).  See H5 in
    # docs/plans/logical-bug-audit.md.
    # Skip the lock-taking state-sync for endpoints whose handlers never
    # read the proxies (e.g. the jobs/active spinner poll), so a frequent
    # poll cannot queue on `_state_lock` behind a long-running job.
    if not request.path.startswith(_STATE_SYNC_EXEMPT_PREFIXES):
        try:
            from vtscore.detectors.dataset_sync import (
                ensure_detector_model_matches_active_embedder,
                ensure_votes_match_active_dataset,
            )

            ensure_votes_match_active_dataset()
            ensure_detector_model_matches_active_embedder()
        except (DatasetNotLoadedError, DetectorNotLoadedError):
            # The route handler will hit the same error when it touches
            # the proxies; the global error handler turns it into a 409.
            pass
        except Exception:
            logging.getLogger(__name__).exception("Vote rehydrate failed")


# ---------------------------------------------------------------------------
# Prevent browser caching of API responses
# ---------------------------------------------------------------------------


@app.teardown_request
def _clear_in_request_handler_marker(exc):  # noqa: ARG001
    """Clear the in-handler marker so the request-missing predicate
    doesn't fire while Flask's test client is preserving a popped
    request context after the response."""
    from flask import g, has_request_context

    if has_request_context():
        g._vts_in_request_handler = False


@app.after_request
def _no_cache_api(response):
    """Prevent browsers from caching mutable API responses.

    Without this, concurrent or rapid-fire fetches to the same endpoint
    (e.g. ``/api/datasets/registry``) can receive stale cached data,
    causing the frontend to miss newly loaded datasets.

    Endpoints serving genuinely immutable data (e.g. frozen projection
    tiles) opt out by setting their own ``Cache-Control`` in the view; we
    defer to that rather than clobbering it with ``no-store``.
    """
    from flask import request

    if request.path.startswith("/api/") and "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.after_request
def _echo_request_id(response):
    """Surface the per-request id on every response so clients can quote it."""
    rid = getattr(g, "request_id", None)
    if rid:
        response.headers["X-Request-Id"] = rid
    return response


# ---------------------------------------------------------------------------
# Global JSON error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(NotFound)
def _handle_404(exc):
    """JSON 404 for unknown ``/api/`` paths.

    Werkzeug's default 404 renders as HTML, which is awful for an SPA. The
    frontend ``ErrorService`` needs JSON to show a useful banner with the
    request_id. Non-API paths fall through to the SPA's catch-all route
    (which serves index.html for client-side routing).

    Scoped to ``NotFound`` specifically so flask-smorest's own
    ``HTTPException`` handler (which renders ``{message, errors}`` for
    marshmallow validation failures and custom ``abort()`` calls) keeps
    handling 400/422/etc.; Flask resolves the most specific exception
    class first.
    """
    from flask import request as _req
    from vtsearch.routes._shared import error_response

    if not _req.path.startswith("/api/"):
        return exc
    return error_response(exc.name, 404)


@app.errorhandler(MethodNotAllowed)
def _handle_405(exc):
    """JSON 405 for wrong-method requests on ``/api/`` paths. See _handle_404."""
    from flask import request as _req
    from vtsearch.routes._shared import error_response

    if not _req.path.startswith("/api/"):
        return exc
    return error_response(exc.name, 405)


@app.errorhandler(DatasetNotLoadedError)
def _handle_dataset_not_loaded(exc):
    """Return a JSON 409 when ``X-Dataset-Id`` names an unloaded dataset.

    Raised by the Flask resolver when a route handler touches the
    dataset proxies and the header doesn't resolve to a loaded context.
    Replaces the silent fallback to the empty context that returned 200
    with stale data (logical-bug-audit H16). The frontend can
    distinguish 409 + ``code="dataset_not_loaded"`` from a generic 500
    and offer a load action.
    """
    from vtsearch.routes._shared import error_response

    return error_response(
        "Dataset is not loaded",
        409,
        dataset_id=exc.dataset_id,
        code="dataset_not_loaded",
    )


@app.errorhandler(DetectorNotLoadedError)
def _handle_detector_not_loaded(exc):
    """Detector counterpart of :func:`_handle_dataset_not_loaded` (H16/H34)."""
    from vtsearch.routes._shared import error_response

    return error_response(
        "Detector is not loaded",
        409,
        detector_id=exc.detector_id,
        code="detector_not_loaded",
    )


from vtscore.state.core import RequestMissingContextError as _RequestMissingContextError


@app.errorhandler(_RequestMissingContextError)
def _handle_request_missing_context(exc):
    """Convert ``RequestMissingContextError`` into a clean 400.

    Raised by the frozen ``_RequestMissingDatasetContext`` /
    ``_RequestMissingDetectorContext`` sentinels when a mutation endpoint
    was hit without an ``X-Dataset-Id`` / ``X-Detector-Id`` header and no
    thread-local pinned context exists. The unloaded-id case is handled
    separately by :func:`_handle_dataset_not_loaded` /
    :func:`_handle_detector_not_loaded` (409 with a specific code). See
    ``docs/plans/logical-bug-audit.md`` H13.
    """
    from flask import request as _req
    from vtsearch.routes._shared import error_response

    if not _req.path.startswith("/api/"):
        raise exc
    return error_response(str(exc), 400)


@app.errorhandler(Exception)
def _handle_uncaught_exception(exc):
    """Return a standardized JSON 500 for any uncaught exception on /api/.

    Without this, an unhandled exception in a route renders Flask's HTML
    debug page (in dev) or a plain 500 (in prod), neither of which carries the
    request_id the user needs to file a bug. ``HTTPException`` is
    excluded so flask-smorest's own handler (and the 404/405 handlers
    above) keep their semantics.
    """
    from werkzeug.exceptions import HTTPException as _HTTPException

    if isinstance(exc, _HTTPException):
        return exc
    from flask import request as _req
    from vtsearch.routes._shared import error_response, format_exception_detail

    logging.getLogger(__name__).exception("Unhandled exception on %s %s", _req.method, _req.path)
    if not _req.path.startswith("/api/"):
        raise exc
    return error_response("Internal server error", 500, detail=format_exception_detail(exc))


# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

# achievements_bp, auth_bp, and main_bp are flask-smorest Blueprints
# (OpenAPI migration); register them via the Api so their decorated
# routes appear in /api/openapi.json. Their undecorated routes (e.g.
# main_bp's SPA-serving paths, achievements_bp's raw-markdown stream)
# attach to Flask normally and are simply absent from the spec.
api.register_blueprint(achievements_bp)
api.register_blueprint(auth_bp)
api.register_blueprint(eval_bp)
api.register_blueprint(file_browser_bp)
api.register_blueprint(health_bp)
api.register_blueprint(jobs_bp)
api.register_blueprint(labels_bp)
api.register_blueprint(media_server_bp)
api.register_blueprint(main_bp)
api.register_blueprint(medias_bp)
api.register_blueprint(sorting_bp)
api.register_blueprint(sessions_bp)
api.register_blueprint(processors_crud_bp)
api.register_blueprint(processors_scoring_bp)
api.register_blueprint(datasets_listings_bp)
api.register_blueprint(datasets_status_bp)
api.register_blueprint(datasets_staging_bp)
api.register_blueprint(datasets_load_bp)
api.register_blueprint(datasets_ui_bp)
api.register_blueprint(datasets_registry_bp)
api.register_blueprint(exporters_bp)
api.register_blueprint(label_importers_bp)
# settings_bp is a flask-smorest Blueprint (OpenAPI pilot); register
# it via the Api so its routes appear in /api/openapi.json. Other
# blueprints stay on the plain Flask path until migrated.
api.register_blueprint(settings_bp)
api.register_blueprint(settings_io_bp)
api.register_blueprint(sync_sources_bp)
api.register_blueprint(detectors_crud_bp)
api.register_blueprint(detectors_labels_bp)
api.register_blueprint(detectors_registry_bp)
api.register_blueprint(detector_scoring_bp)
api.register_blueprint(detector_find_bp)
api.register_blueprint(embed_bp)
api.register_blueprint(projection_bp)
app.register_blueprint(events_bp)
# Plain Flask blueprint (browser-redirect callback + JSON status); kept off the
# OpenAPI surface like events_bp.
app.register_blueprint(hf_auth_bp)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


def initialize_server(mode_label: str = "PRODUCTION") -> None:
    """Load models and preload media types.

    Called from ``__main__`` (when running ``python app.py``) and from
    gunicorn via ``VTSEARCH_SERVER_INIT=1`` at import time. Idempotent: safe
    to call multiple times; individual steps handle their own caching.

    Per-user ``settings_source`` sync runs lazily on each user's first
    settings access (see
    ``vtsearch.settings_store.UserSettingsStore._run_sync_from_source``)
    and re-fires whenever the source's ``peek_version`` token changes,
    not at server boot; there is no server-wide user to sync for.
    """
    print(f"\U0001f680 Running in {mode_label} mode", flush=True)

    # Deployment-level dataset retention override from the environment, for
    # the gunicorn-launched images that never parse ``--dataset-max-age-days``
    # (the argparse path below only runs under ``python app.py``). An explicit
    # CLI flag, if one was passed, always wins. Like the flag, this applies to
    # every user for the lifetime of the process and is not editable via the
    # settings API.
    from vtsearch.settings import get_cli_dataset_max_age_days, set_cli_dataset_max_age_days

    if get_cli_dataset_max_age_days() is None:
        _env_max_age = os.environ.get("VTSEARCH_DATASET_MAX_AGE_DAYS")
        if _env_max_age:
            try:
                _days = int(_env_max_age)
            except ValueError:
                _days = 0
            if _days >= 1:
                set_cli_dataset_max_age_days(_days)
                print(f"\U0001f5d3️  Dataset max age: {_days}d (from VTSEARCH_DATASET_MAX_AGE_DAYS)", flush=True)
            else:
                print(
                    f"⚠️  Ignoring VTSEARCH_DATASET_MAX_AGE_DAYS={_env_max_age!r} "
                    "(want a positive integer number of days)",
                    flush=True,
                )

    print("  Loading ML libraries...", flush=True)
    initialize_models(on_progress=lambda *a, **k: None)
    # ``--solo-media-type`` (process-level CLI fallback) tells us which
    # mediaType's default embedder to warm even if no datasets or detectors
    # are registered yet. Per-user explicit values are not consulted here
    # because there is no current user at startup.
    from vtsearch.settings import get_cli_solo_embedders, get_cli_solo_media_type

    cli_solo = get_cli_solo_media_type()
    extra_types = [cli_solo] if cli_solo else None
    if cli_solo:
        print(f"\U0001f3af Solo mediaType: {cli_solo} (from --solo-media-type)", flush=True)
    cli_solo_embedders = get_cli_solo_embedders()
    extra_embedders = list(cli_solo_embedders.values()) if cli_solo_embedders else None
    if cli_solo_embedders:
        pretty = ", ".join(f"{mt}={emb}" for mt, emb in cli_solo_embedders.items())
        print(f"\U0001f3af Solo mediaEmbedders: {pretty} (from --solo-embedder)", flush=True)
    preloaded = preload_predicted_embedders(
        extra_media_types=extra_types,
        extra_embedders=extra_embedders,
    )
    if preloaded:
        print(f"✅ Preloaded embedders: {', '.join(preloaded)}", flush=True)

    print("✅ VTSearch is ready!", flush=True)


# When running under gunicorn (or any other WSGI server), ``app.py`` is
# imported rather than executed, so the ``__main__`` block below never
# runs. The Dockerfile sets ``VTSEARCH_SERVER_INIT=1`` to trigger the
# same startup sequence at import time.
if os.environ.get("VTSEARCH_SERVER_INIT") == "1":
    initialize_server()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vtsearch.cli_main import main

    main(app, initialize_server)
