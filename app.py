import logging
import os
import sys
import warnings

# Limit threads to reduce memory overhead in constrained environments.  Native
# math libraries read these env vars during *their* import, which happens the
# moment torch / numpy / scipy are imported — so they have to be set before
# anything triggers that.  Mirrors ``vtsearch.config.TORCH_THREADS`` but
# resolved inline to avoid importing ``vtsearch.config`` (and therefore
# everything it transitively imports) this early.
_torch_threads = str(max(1, int(os.environ.get("VTSEARCH_TORCH_THREADS", "1"))))
os.environ["OMP_NUM_THREADS"] = _torch_threads
os.environ["MKL_NUM_THREADS"] = _torch_threads

# Configure structured logging — JSON lines by default with per-record
# request_id / dataset_id / detector_id / user fields. Override via:
#   VTSEARCH_LOG_LEVEL=INFO   (default WARNING)
#   VTSEARCH_LOG_FORMAT=text  (default json — switch to text for local dev)
from vtsearch.logging_config import new_request_id, setup_logging  # noqa: E402

setup_logging()

# All HF models we use are public — no token needed.  Each from_pretrained()
# call passes token=False to signal this explicitly.  The env var + warnings
# filter below are belt-and-suspenders in case any transitive HF code still
# warns about missing tokens.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Visual feedback for startup
print(f"⏳ Initializing VTSearch... (PID {os.getpid()})", flush=True)

from flask import Flask, g

# Import refactored modules
from vtsearch.auth import get_login_provider  # noqa: E402
from vtsearch.embedding import initialize_models, preload_predicted_embedders  # noqa: E402
from vtsearch.routes import (  # noqa: E402
    achievements_bp,
    auth_bp,
    detector_find_bp,
    detector_scoring_bp,
    detectors_bp,
    detectors_registry_bp,
    embed_bp,
    eval_bp,
    events_bp,
    file_browser_bp,
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
    processors_bp,
    settings_bp,
    settings_io_bp,
    sorting_bp,
    sync_sources_bp,
)
from vtsearch.media import set_progress_callback  # noqa: E402
from vtsearch.concurrency.progress import update_progress

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
from vtsearch.config import MAX_UPLOAD_MB as _MAX_UPLOAD_MB  # noqa: E402

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

api = Api(app)


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


@app.before_request
def _set_request_context():
    """Resolve per-request dataset/detector context from HTTP headers.

    If the frontend sends ``X-Dataset-Id`` or ``X-Detector-Id``, the
    corresponding context is stashed on ``g`` so that proxy objects
    (``medias``, ``good_votes``, etc.) resolve to it for the duration of
    this request — without mutating global "active" state.

    When the headers are absent the proxies fall back to the global active
    pointers, preserving backward compatibility.

    Any failure here must not 500 every subsequent request — fall back to
    the default (empty) context.
    """
    from flask import request
    from vtsearch.state.core import (
        get_context,
        get_detector_context,
    )

    try:
        # Headers (Angular HttpClient interceptor) take priority, with query
        # params as fallback for browser-native requests (<img src>,
        # <audio src>, <video src>) that bypass Angular's interceptor.
        ds_id = request.headers.get("X-Dataset-Id") or request.args.get("dataset_id")
        if ds_id:
            ctx = get_context(ds_id)
            if ctx is not None:
                g._dataset_context = ctx

        detector_id = request.headers.get("X-Detector-Id") or request.args.get("detector_id")
        if detector_id:
            det_ctx = get_detector_context(detector_id)
            if det_ctx is not None:
                g._detector_context = det_ctx
    except Exception:
        logging.getLogger(__name__).exception("Request context resolution failed")

    # If the active (dataset, detector) pair has changed since the detector's
    # cid-keyed vote dicts were last derived, rehydrate them from the on-disk
    # labelset against the active dataset's medias.  Media ids are dataset-
    # specific, so without this the left-pane shows stale cids from the
    # previous dataset as if they were votes in the current one.
    try:
        from vtsearch.detectors.dataset_sync import ensure_votes_match_active_dataset

        ensure_votes_match_active_dataset()
    except Exception:
        logging.getLogger(__name__).exception("Vote rehydrate failed")


# ---------------------------------------------------------------------------
# Prevent browser caching of API responses
# ---------------------------------------------------------------------------


@app.after_request
def _no_cache_api(response):
    """Prevent browsers from caching mutable API responses.

    Without this, concurrent or rapid-fire fetches to the same endpoint
    (e.g. ``/api/datasets/registry``) can receive stale cached data,
    causing the frontend to miss newly loaded datasets.
    """
    from flask import request

    if request.path.startswith("/api/"):
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
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(achievements_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(eval_bp)
app.register_blueprint(file_browser_bp)
app.register_blueprint(labels_bp)
app.register_blueprint(media_server_bp)
app.register_blueprint(main_bp)
app.register_blueprint(medias_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(processors_bp)
app.register_blueprint(datasets_listings_bp)
app.register_blueprint(datasets_status_bp)
app.register_blueprint(datasets_staging_bp)
app.register_blueprint(datasets_load_bp)
app.register_blueprint(datasets_ui_bp)
app.register_blueprint(datasets_registry_bp)
app.register_blueprint(exporters_bp)
app.register_blueprint(label_importers_bp)
# settings_bp is a flask-smorest Blueprint (OpenAPI pilot); register
# it via the Api so its routes appear in /api/openapi.json. Other
# blueprints stay on the plain Flask path until migrated.
api.register_blueprint(settings_bp)
app.register_blueprint(settings_io_bp)
app.register_blueprint(sync_sources_bp)
app.register_blueprint(detectors_bp)
app.register_blueprint(detectors_registry_bp)
app.register_blueprint(detector_scoring_bp)
app.register_blueprint(detector_find_bp)
app.register_blueprint(embed_bp)
app.register_blueprint(events_bp)


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
    :func:`vtsearch.settings._maybe_sync_from_source_locked`), not at
    server boot — there is no server-wide user to sync for.
    """
    print(f"\U0001f680 Running in {mode_label} mode", flush=True)
    initialize_models()
    preloaded = preload_predicted_embedders()
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
    import argparse

    parser = argparse.ArgumentParser(description="VTSearch \u2014 media explorer web app")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--login",
        type=str,
        choices=["trivial"],
        default=None,
        help="Login provider: 'trivial' shows a username prompt (no password, cookie-based)",
    )
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        dest="list_plugins",
        help=(
            "List every auto-discovered plugin (importers, exporters, embedders, "
            "converters, clippers, …) and exit. Useful for shell completion."
        ),
    )
    parser.add_argument(
        "--plugin-family",
        type=str,
        default=None,
        dest="plugin_family",
        help=(
            "When given with --list-plugins, restrict output to this family "
            "(e.g. importers, exporters). Combine with --format=names for "
            "completion-friendly output."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="plain",
        choices=["plain", "json", "names"],
        dest="output_format",
        help=(
            "Output format for --list-plugins / --openapi-schema. 'plain' is "
            "human-readable, 'json' is machine-readable, 'names' emits bare "
            "plugin names one per line (shell-completion friendly). "
            "--openapi-schema only honours 'json' (default for that flag)."
        ),
    )
    parser.add_argument(
        "--openapi-schema",
        action="store_true",
        dest="openapi_schema",
        help="Print the auto-generated OpenAPI 3.0 spec for the HTTP API and exit.",
    )
    parser.add_argument("--dataset", type=str, help="Path to a dataset pickle file (used with --autodetect)")
    parser.add_argument(
        "--settings",
        type=str,
        help=(
            "Path to a settings JSON file containing autorun processors. "
            "Used with --autodetect. Defaults to data/settings.json."
        ),
    )
    parser.add_argument(
        "--importer",
        type=str,
        help="Name of the data importer to use (e.g. folder, pickle, http_archive). Used with --autodetect.",
    )
    parser.add_argument(
        "--exporter",
        type=str,
        help="Name of the results exporter to use (e.g. file, email_smtp, gui). Used with --autodetect.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        dest="chunk_size",
        help=(
            "Process the dataset in chunks of N medias at a time to limit "
            "memory usage. Used with --autodetect. When omitted the entire "
            "dataset is loaded at once (original behaviour)."
        ),
    )
    parser.add_argument(
        "--import-labels-into",
        type=str,
        default=None,
        dest="import_labels_into",
        help=("Detector name to merge labels into before scoring. Used with --autodetect plus --label-importer-file."),
    )
    parser.add_argument(
        "--label-importer",
        type=str,
        default="server_json_file",
        dest="label_importer",
        help=("Label importer name to use with --import-labels-into (default: server_json_file)."),
    )
    parser.add_argument(
        "--label-importer-file",
        type=str,
        default=None,
        dest="label_importer_file",
        help=("Path passed to the label importer's ``filepath`` field. Used with --import-labels-into."),
    )

    # Two-pass parsing: first pass gets --importer and --exporter names;
    # second pass adds their plugin-specific arguments and re-parses.
    args, remaining = parser.parse_known_args()

    # ---- Early-exit informational flags --------------------------------
    # These run before the autodetect / server paths so they don't trigger
    # model loading or the full Flask app boot.
    if args.list_plugins:
        from vtsearch.plugins.inventory import format_json, format_names, format_plain, gather_plugins

        inventory = gather_plugins()
        if args.plugin_family:
            if args.plugin_family not in inventory:
                parser.error(f"Unknown plugin family: {args.plugin_family}. Available: {', '.join(inventory)}")
            inventory = {args.plugin_family: inventory[args.plugin_family]}
        if args.output_format == "json":
            sys.stdout.write(format_json(inventory))
            sys.stdout.write("\n")
        elif args.output_format == "names":
            sys.stdout.write(format_names(inventory, family=args.plugin_family))
        else:
            sys.stdout.write(format_plain(inventory))
        sys.exit(0)

    if args.openapi_schema:
        import json as _json

        from vtsearch.openapi import generate_openapi_spec

        spec = generate_openapi_spec(app)
        sys.stdout.write(_json.dumps(spec, indent=2))
        sys.stdout.write("\n")
        sys.exit(0)

    importer = None
    exporter = None

    if args.autodetect and args.importer:
        from vtsearch.datasets.importers import get_importer, list_importers

        importer = get_importer(args.importer)
        if importer is None:
            available = ", ".join(imp.name for imp in list_importers())
            parser.error(f"Unknown importer: {args.importer}. Available: {available}")

        importer.add_cli_arguments(parser)

    if args.autodetect and args.exporter:
        from vtsearch.exporters import get_exporter, list_exporters

        exporter = get_exporter(args.exporter)
        if exporter is None:
            available = ", ".join(exp.name for exp in list_exporters())
            parser.error(f"Unknown exporter: {args.exporter}. Available: {available}")

        exporter.add_cli_arguments(parser)

    if importer or exporter:
        args = parser.parse_args()
    elif remaining:
        # No importer/exporter specified but there are unknown args; let
        # argparse report the error.
        parser.parse_args()

    if args.autodetect:
        # Collect exporter field values if an exporter was specified
        exporter_field_values = None
        if exporter:
            exporter_field_values = {f.key: getattr(args, f.key, f.default) for f in exporter.fields}

        settings_path = getattr(args, "settings", None)

        chunk_size = getattr(args, "chunk_size", None)

        # Optional one-shot label import into a detector before scoring.
        # The merged labelset is picked up by the autodetect pipeline below.
        if args.import_labels_into:
            if not args.label_importer_file:
                parser.error("--import-labels-into requires --label-importer-file <path>")
            # Settings file controls detectors_dir, so apply it first.
            if settings_path:
                from vtsearch.settings import set_settings_path

                set_settings_path(settings_path)
            from vtsearch.cli import import_labels_into_detector_from_file

            try:
                applied, skipped = import_labels_into_detector_from_file(
                    args.import_labels_into,
                    args.label_importer,
                    args.label_importer_file,
                )
                print(
                    f"Imported {applied} label(s) into detector "
                    f"'{args.import_labels_into}' (skipped {skipped} duplicate/invalid).",
                    flush=True,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error importing labels: {exc}", file=sys.stderr)
                sys.exit(1)

        if args.importer:
            # Importer-based path
            field_values = {f.key: getattr(args, f.key, f.default) for f in importer.fields}

            if chunk_size:
                from vtsearch.cli import autodetect_importer_main_chunked

                autodetect_importer_main_chunked(
                    args.importer, field_values, chunk_size, settings_path, args.exporter, exporter_field_values
                )
            else:
                from vtsearch.cli import autodetect_importer_main

                autodetect_importer_main(
                    args.importer, field_values, settings_path, args.exporter, exporter_field_values
                )

        elif args.dataset:
            # Pickle-file path
            if chunk_size:
                from vtsearch.cli import autodetect_main_chunked

                autodetect_main_chunked(args.dataset, chunk_size, settings_path, args.exporter, exporter_field_values)
            else:
                from vtsearch.cli import autodetect_main

                autodetect_main(args.dataset, settings_path, args.exporter, exporter_field_values)

        else:
            parser.error("--autodetect requires either --dataset <file.pkl> or --importer <name>")

    elif args.local or not args.autodetect:
        # Activate the chosen login provider before starting the server.
        if getattr(args, "login", None) == "trivial":
            from vtsearch.auth import TrivialLoginProvider, set_login_provider

            set_login_provider(TrivialLoginProvider())
            print("\U0001f511 Trivial login enabled \u2014 users will be prompted for a username", flush=True)

        initialize_server(mode_label="LOCAL" if args.local else "PRODUCTION")
        print("\U0001f310 Open http://localhost:5000 in your browser", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
