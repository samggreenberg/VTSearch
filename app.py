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
from vtsearch.logging_config import setup_logging  # noqa: E402

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

from flask import Flask

# Import refactored modules
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
    detectors_export_bp,
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

# Cap on request body size (uploads).  Defaults to 2 GiB (see
# ``vtscore.config.MAX_UPLOAD_MB``); a positive value rejects oversized
# requests with HTTP 413 before they consume disk.  ``MAX_UPLOAD_MB == 0``
# (set via ``VTSEARCH_MAX_UPLOAD_MB=0``) leaves Flask's default of no limit.
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
# Request-lifecycle hooks and global JSON error handlers
# ---------------------------------------------------------------------------
# Both live in their own modules (``vtsearch.hooks`` / ``vtsearch.errors``);
# registration is decorator-based on this module-level ``app``, so each
# module exposes a ``register_*`` function that wires its handlers on in the
# original order.

from vtsearch.errors import register_error_handlers  # noqa: E402
from vtsearch.hooks import register_hooks  # noqa: E402

register_hooks(app)
register_error_handlers(app)


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
api.register_blueprint(detectors_export_bp)
api.register_blueprint(embed_bp)
api.register_blueprint(projection_bp)
app.register_blueprint(events_bp)
# Plain Flask blueprint (browser-redirect callback + JSON status); kept off the
# OpenAPI surface like events_bp.
app.register_blueprint(hf_auth_bp)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


def _apply_env_dataset_max_age() -> None:
    """Honour ``VTSEARCH_DATASET_MAX_AGE_DAYS`` when no CLI flag was passed.

    The gunicorn-launched images never parse ``--dataset-max-age-days`` (the
    argparse path only runs under ``python app.py``), so the same override is
    reachable from the environment. An explicit CLI flag always wins. Like the
    flag, this applies to every user for the lifetime of the process and is not
    editable via the settings API.
    """
    from vtsearch.settings import get_cli_dataset_max_age_days, set_cli_dataset_max_age_days

    if get_cli_dataset_max_age_days() is not None:
        return
    raw = os.environ.get("VTSEARCH_DATASET_MAX_AGE_DAYS")
    if not raw:
        return
    try:
        days = int(raw)
    except ValueError:
        days = 0
    if days >= 1:
        set_cli_dataset_max_age_days(days)
        print(f"\U0001f5d3️  Dataset max age: {days}d (from VTSEARCH_DATASET_MAX_AGE_DAYS)", flush=True)
    else:
        print(
            f"⚠️  Ignoring VTSEARCH_DATASET_MAX_AGE_DAYS={raw!r} (want a positive integer number of days)",
            flush=True,
        )


def _apply_env_support_email() -> None:
    """Honour ``VTSEARCH_SUPPORT_EMAIL`` when no CLI flag was passed.

    Same rationale as :func:`_apply_env_dataset_max_age`: the gunicorn images
    never parse ``--support-email``. An explicit CLI flag always wins.
    """
    from vtsearch.settings import get_cli_support_email, set_cli_support_email

    if get_cli_support_email() is not None:
        return
    email = (os.environ.get("VTSEARCH_SUPPORT_EMAIL") or "").strip()
    if email:
        set_cli_support_email(email)
        print(f"\U0001f4e7 Support email: {email} (from VTSEARCH_SUPPORT_EMAIL)", flush=True)


def _apply_env_semantic_only() -> None:
    """Honour ``VTSEARCH_SEMANTIC_ONLY`` when no CLI flag was passed.

    Same rationale as the two helpers above (gunicorn never parses
    ``--semantic-only``); any of ``1``/``true``/``yes``/``on`` enables the
    lock. Reports the lock however it arrived - flag, env var, or the persisted
    ``semantic_only`` setting - so an operator can confirm from the startup log
    that the prototype embedder types are off.
    """
    from vtsearch.settings import (
        get_cli_semantic_only,
        get_effective_semantic_only,
        set_cli_semantic_only,
    )

    source = "--semantic-only"
    if get_cli_semantic_only() is None:
        if (os.environ.get("VTSEARCH_SEMANTIC_ONLY") or "").strip().lower() in ("1", "true", "yes", "on"):
            set_cli_semantic_only(True)
            source = "VTSEARCH_SEMANTIC_ONLY"
        else:
            source = "the semantic_only server setting"
    if get_effective_semantic_only():
        print(f"\U0001f512 Semantic embedders only (from {source})", flush=True)


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

    # Deployment-level overrides that the gunicorn-launched images can only
    # reach through the environment (they never parse argv).
    _apply_env_dataset_max_age()
    _apply_env_support_email()
    _apply_env_semantic_only()

    print("\U0001f4da Loading ML libraries...", flush=True)
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
