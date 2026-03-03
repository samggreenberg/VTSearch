import logging
import os
import warnings

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Suppress Werkzeug request logging (GET/POST lines) — only show errors
logging.getLogger("werkzeug").setLevel(logging.ERROR)
# Suppress huggingface_hub "unauthenticated requests" console warning —
# all models we use are public, so no token is needed.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# All HF models we use are public — no token needed.  Each from_pretrained()
# call passes token=False to signal this explicitly.  The env var + warnings
# filter below are belt-and-suspenders in case any transitive HF code still
# warns about missing tokens.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*")

# Visual feedback for startup
print("⏳ Initializing VTSearch...", flush=True)

from flask import Flask

# Import refactored modules
from vtsearch.medias import init_medias  # noqa: E402, F401 — used by tests via app_module.init_medias()
from vtsearch.models import initialize_models, preload_autoload_media_types  # noqa: E402
from vtsearch.routes import (  # noqa: E402
    medias_bp,
    datasets_bp,
    detectors_bp,
    exporters_bp,
    label_importers_bp,
    main_bp,
    processor_importers_bp,
    settings_bp,
    sorting_bp,
    trainable_models_bp,
)
from vtsearch.media import set_progress_callback  # noqa: E402
from vtsearch.utils import update_progress  # noqa: E402

# Wire media types into the Flask app's progress reporting system.
# Without this call, media types use a silent no-op callback and can run
# standalone (e.g. in a CLI tool or notebook) without Flask.
set_progress_callback(update_progress)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(main_bp)
app.register_blueprint(medias_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(detectors_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(exporters_bp)
app.register_blueprint(label_importers_bp)
app.register_blueprint(processor_importers_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(trainable_models_bp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VTSearch \u2014 media explorer web app")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
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

    # Two-pass parsing: first pass gets --importer and --exporter names;
    # second pass adds their plugin-specific arguments and re-parses.
    args, remaining = parser.parse_known_args()

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

                autodetect_main_chunked(
                    args.dataset, chunk_size, settings_path, args.exporter, exporter_field_values
                )
            else:
                from vtsearch.cli import autodetect_main

                autodetect_main(args.dataset, settings_path, args.exporter, exporter_field_values)

        else:
            parser.error("--autodetect requires either --dataset <file.pkl> or --importer <name>")

    elif args.local:
        # Local development mode
        print("\U0001f680 Running in LOCAL mode (accessible from other devices)", flush=True)
        initialize_models()
        preloaded = preload_autoload_media_types()
        if preloaded:
            print(f"\u2705 Preloaded autoload media types: {', '.join(preloaded)}", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        # Production mode \u2014 models load lazily when the first dataset is loaded
        print("\U0001f680 Running in PRODUCTION mode", flush=True)
        initialize_models()
        preloaded = preload_autoload_media_types()
        if preloaded:
            print(f"\u2705 Preloaded autoload media types: {', '.join(preloaded)}", flush=True)

        print("\u2705 VTSearch is ready!", flush=True)
        print("\U0001f310 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
