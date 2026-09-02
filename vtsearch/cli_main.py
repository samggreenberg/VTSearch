"""Command-line entry point for ``python app.py``.

Holds the argparse definition, the early-exit informational flags
(``--list-plugins`` and family shortcuts), the ``--pipeline`` and
``--autodetect`` dispatch, and the dev-server launch path. Extracted from
``app.py`` so the WSGI ``app`` object gunicorn imports is not weighed down by
~650 lines of CLI plumbing that only runs under ``python app.py``.

The Flask ``app`` object and ``initialize_server`` callable are passed in by
``app.py``'s ``__main__`` block rather than imported here: importing ``app``
from this module while ``app.py`` is running as ``__main__`` would re-execute
the whole module under a second name, double-registering blueprints and
re-running server init.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from vtscore.media import set_progress_callback

from vtsearch import admin_overrides
from vtsearch.logging_config import setup_logging
from vtsearch.port_preflight import _acquire_single_instance_lock, _preflight_port


def _build_parser() -> argparse.ArgumentParser:
    """Construct the full ``python app.py`` argument parser."""
    parser = argparse.ArgumentParser(description="VTSearch \u2014 media explorer web app", add_help=False)
    # Help is a plain flag, not argparse's built-in ``action="help"``. The
    # built-in action fires and exits during the *first* parse pass, which
    # happens before ``--importer``/``--exporter`` have had their plugin-specific
    # flags registered -- so ``--importer server_folder --help`` would print help
    # that omits the very flags the user asked about. ``main()`` registers the
    # selected plugin's arguments first and prints help afterwards.
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="show this help message and exit (lists the selected --importer/--exporter's flags too)",
    )
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Port for the dev server to bind (default 5000, or VTSEARCH_PORT "
            "if set). Useful when several instances share a host, e.g. "
            "co-located SLURM jobs on a multi-GPU node."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help=(
            "Increase log verbosity. -v turns on INFO logging, which includes "
            "the dev-server access log (one line per HTTP request); -vv turns "
            "on DEBUG. Only raises the level set by VTSEARCH_LOG_LEVEL, never "
            "lowers it. Applies to both the web server and --autodetect."
        ),
    )
    parser.add_argument(
        "--login",
        type=str,
        choices=["trivial", "api_key"],
        default=None,
        help=(
            "Login provider: 'trivial' shows a username prompt (no password, cookie-based); "
            "'api_key' authenticates via Authorization: Bearer <key> against data/api_keys.json"
        ),
    )
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help=(
            "Run --autodetect as this user, so their per-user Auto-Find list "
            "(autofind_detectors) and results exporter apply. Requires --api-key "
            "to authenticate against data/api_keys.json (same credentials as the "
            "server's api_key login). Without --user the run uses the built-in "
            "'default' user, which reads the --settings file."
        ),
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default=None,
        help="Bearer key authenticating --user against data/api_keys.json.",
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        dest="list_plugins",
        help=(
            "List every auto-discovered plugin (importers, exporters, embedders, "
            "converters, clippers, …) and exit. Useful for shell completion. "
            "Per-family shortcuts are also available; see --list-importers, "
            "--list-exporters, etc."
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

    # Per-family shortcuts: ``--list-importers`` ≡ ``--list-plugins
    # --plugin-family importers``, and so on for every family in
    # vtscore.plugins.inventory.FAMILIES.
    from vtscore.plugins.inventory import register_family_shortcuts

    register_family_shortcuts(parser)
    parser.add_argument(
        "--format",
        type=str,
        default="plain",
        choices=["plain", "json", "names"],
        dest="output_format",
        help=(
            "Output format for --list-plugins. 'plain' is human-readable, "
            "'json' is machine-readable, 'names' emits bare plugin names "
            "one per line (shell-completion friendly)."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        dest="pipeline",
        help=(
            "Run the importer/detector/exporter sequence declared in a YAML "
            "pipeline file. Replaces the --autodetect flag set with one "
            "config file. See docs/CLI.md for the schema."
        ),
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
        "--stream-results",
        action="store_true",
        dest="stream_results",
        help=(
            "Stream each chunk's hits straight to the exporter instead of "
            "accumulating them all in memory. Requires --chunk-size and a "
            "streaming-capable exporter (server_json_file → NDJSON, "
            "server_csv_file, gui). Output is ordered by chunk, not globally "
            "sorted by score. Lets --autodetect run against a media source "
            "with more items (and more hits) than fit in RAM."
        ),
    )
    parser.add_argument(
        "--keep-negatives",
        action="store_true",
        dest="keep_negatives",
        help=(
            "With --stream-results, also stream below-threshold (negative) "
            "hits, tagged label=bad. Off by default: a find over a massive "
            "source only emits the predicted-good items."
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Print what --autodetect would embed, score, and export without "
            "doing it. Validates importer/exporter names, settings file, and "
            "any --import-labels-into request, but loads no media and trains "
            "no models."
        ),
    )
    parser.add_argument(
        "--progress-format",
        type=str,
        default="text",
        choices=["text", "json"],
        dest="progress_format",
        help=(
            "Format for CLI status output. 'text' (default) prints "
            "human-readable prose; 'json' emits NDJSON on stdout, one event "
            "per line, for scripted callers and CI. See vtscore.cli_progress "
            "for the event schema. Applies to --autodetect."
        ),
    )
    # The process-level admin overrides (--solo-media-type, --solo-embedder,
    # --hide-plugin, --dataset-max-age-days, --support-email, --semantic-only)
    # are declared once in vtsearch.admin_overrides, which owns their flag
    # spellings, help text, env-var equivalents and validators together.
    admin_overrides.register_override_flags(parser)
    return parser


def _maybe_print_help(args, parser) -> None:
    """Handle ``-h``/``--help``, including plugin-specific flags, then exit.

    Registers the flags of any ``--importer``/``--exporter`` named on the
    command line before printing, so ``python app.py --autodetect --importer
    server_folder --help`` lists ``--path``, ``--media-type``, and the rest of
    that importer's fields rather than only the base flags. (argparse's default
    help action can't do this: it fires during the first parse pass, before the
    plugin flags exist.)
    """
    if not getattr(args, "show_help", False):
        return
    _register_plugin_cli_args(args, parser)
    parser.print_help()
    sys.exit(0)


def _maybe_list_plugins(args, parser) -> None:
    """Handle ``--list-plugins`` (and family shortcuts), then exit."""
    # ---- Early-exit informational flags --------------------------------
    # These run before the autodetect / server paths so they don't trigger
    # model loading or the full Flask app boot.
    if args.list_plugins:
        from vtscore.plugins.inventory import format_json, format_names, format_plain, gather_plugins

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


def _maybe_run_pipeline(args, parser, remaining) -> None:
    """Handle ``--pipeline FILE`` (mutually exclusive with autodetect flags), then exit."""
    # ---- Pipeline file ---------------------------------------------------
    # `--pipeline pipeline.yaml` declares an autodetect run in YAML instead
    # of flags. It is mutually exclusive with the rest of the autodetect
    # CLI: any extra autodetect flag (importer/dataset/exporter/settings/
    # chunk-size/import-labels-into) belongs in the YAML, not on the command
    # line.
    if args.pipeline:
        for conflicting in (
            "autodetect",
            "dataset",
            "importer",
            "exporter",
            "settings",
            "chunk_size",
            "import_labels_into",
            "label_importer_file",
            "dry_run",
        ):
            if getattr(args, conflicting, None):
                cli_flag = f"--{conflicting.replace('_', '-')}"
                parser.error(f"--pipeline cannot be combined with {cli_flag}; declare it in the YAML file instead.")
        if remaining:
            parser.error(
                f"--pipeline does not accept extra flags ({' '.join(remaining)}); "
                "declare plugin field values in the YAML file instead."
            )
        from vtscore.cli_pipeline import run_pipeline_file

        run_pipeline_file(args.pipeline)
        sys.exit(0)


def _register_plugin_cli_args(args, parser: argparse.ArgumentParser):
    """Look up ``--importer``/``--exporter`` and register their plugin flags.

    Returns the ``(importer, exporter)`` plugin instances (``None`` when the
    corresponding flag wasn't given). Called both by :func:`_resolve_plugins`
    for the second parse pass and by the ``--help`` path in :func:`main`, so
    ``--importer <name> --help`` prints that importer's flags.
    """
    importer = None
    exporter = None

    if args.importer:
        from vtscore.datasets.importers import get_importer, list_importers

        importer = get_importer(args.importer)
        if importer is None:
            available = ", ".join(imp.name for imp in list_importers())
            parser.error(f"Unknown importer: {args.importer}. Available: {available}")

        importer.add_cli_arguments(parser)

    if args.exporter:
        from vtscore.exporters import get_exporter, list_exporters

        exporter = get_exporter(args.exporter)
        if exporter is None:
            available = ", ".join(exp.name for exp in list_exporters())
            parser.error(f"Unknown exporter: {args.exporter}. Available: {available}")

        exporter.add_cli_arguments(parser)

    return importer, exporter


def _resolve_plugins(args, parser, remaining):
    """Two-pass resolution of ``--importer``/``--exporter`` and their plugin args.

    Returns ``(args, importer, exporter)``; ``args`` is re-parsed once the
    plugin-specific arguments have been registered.
    """
    importer = None
    exporter = None

    if args.autodetect:
        importer, exporter = _register_plugin_cli_args(args, parser)

    if importer or exporter:
        args = parser.parse_args()
    elif remaining:
        # No importer/exporter specified but there are unknown args; let
        # argparse report the error.
        parser.parse_args()
    return args, importer, exporter


def _apply_verbosity(args) -> None:
    """Apply ``-v``/``-vv`` by re-running logging setup at the higher level."""
    # -v/--verbose bumps the log level for this process. setup_logging() already
    # ran at import time with the env-driven default (WARNING); re-run it at the
    # higher level so the dev-server access log (werkzeug INFO) and our own
    # INFO/DEBUG records start showing. Only raise verbosity, never lower it
    # below an explicit VTSEARCH_LOG_LEVEL=debug, so -v on top of a debug env
    # doesn't quiet things back down. Applies before both the autodetect CLI
    # and server branches below.
    verbose = getattr(args, "verbose", 0) or 0
    if verbose:
        target = logging.DEBUG if verbose >= 2 else logging.INFO
        # Lower numeric level == more verbose; keep whichever is more verbose.
        effective = min(target, logging.getLogger().level)
        setup_logging(level=logging.getLevelName(effective))


def _apply_admin_overrides(args, parser) -> None:
    """Validate and stash every process-level admin override the flags carried.

    One loop over :data:`vtsearch.admin_overrides.OVERRIDES` replaces what used
    to be six near-identical ``_apply_X`` helpers. Each descriptor owns its own
    validator, so ``--solo-media-type`` still checks the media-type registry
    and ``--solo-embedder`` still checks that the embedder is registered *for
    that type* -- but the same validator now runs for the env-var form, so the
    two entry paths can no longer disagree.

    Runs for both the autodetect CLI path and the server path, before any
    media is loaded or any listing endpoint is served: a typo here would
    otherwise silently no-op the restriction and the user would only notice
    when the picker reappeared.
    """
    try:
        admin_overrides.apply_flag_values(args)
    except admin_overrides.OverrideValueError as exc:
        parser.error(str(exc))


def _authenticate_cli_user(args, parser) -> None:
    """Establish and authenticate the ``--user`` an Auto-Find run executes as."""
    from vtscore import cli_progress

    # Establish (and authenticate) the user this Auto-Find runs as, mirroring
    # the server's api_key login. With --user the run reads that user's
    # per-user Auto-Find list + results exporter; without it, the built-in
    # "default" user applies (which reads the --settings flat file).
    if args.user:
        from vtscore.config import DATA_DIR

        from vtsearch.auth import ApiKeyLoginProvider, set_login_provider, set_thread_user

        if not args.api_key:
            parser.error("--user requires --api-key <key> to authenticate against data/api_keys.json")

        class _CliRequest:
            headers = {"Authorization": f"Bearer {args.api_key}"}

        _req = _CliRequest()
        _provider = ApiKeyLoginProvider(keys_file=DATA_DIR / "api_keys.json")
        if not _provider.is_authenticated(_req):
            parser.error("Invalid --api-key: no matching key in data/api_keys.json")
        _authed = _provider.get_user(_req)
        if _authed != args.user:
            parser.error(f"--api-key authenticates as {_authed!r}, not --user {args.user!r}")
        # Activate the provider process-wide, not just for the auth check: it is
        # what get_user_data_dir() consults to resolve per-user paths. Without
        # this the default provider stays installed and ignores the username, so
        # data/<user>/user_settings.json would silently read (and write) through
        # to the default user's flat file -- i.e. the wrong Auto-Find list and
        # exporter config, under the right username. Mirrors _run_server's
        # --login api_key setup so a CLI run resolves paths identically.
        set_login_provider(_provider)
        set_thread_user(args.user)
        cli_progress.emit(
            "authenticated_user",
            text=f"Running Auto-Find as user {args.user!r}.",
            user=args.user,
        )
    elif args.api_key:
        parser.error("--api-key requires --user <name>")


def _maybe_import_labels(args, parser, settings_path, dry_run) -> None:
    """Optionally merge labels into a detector before scoring (``--import-labels-into``)."""
    from vtscore import cli_progress

    # Optional one-shot label import into a detector before scoring.
    # The merged labelset is picked up by the autodetect pipeline below.
    if args.import_labels_into:
        if not args.label_importer_file:
            parser.error("--import-labels-into requires --label-importer-file <path>")
        # Settings file controls detectors_dir, so apply it first.
        if settings_path:
            from vtsearch.settings import set_settings_path

            set_settings_path(settings_path)
        if dry_run:
            cli_progress.emit(
                "labels_import_dry_run",
                text=(
                    f"DRY RUN: would import labels from {args.label_importer_file!r} "
                    f"via importer {args.label_importer!r} into detector "
                    f"{args.import_labels_into!r}."
                ),
                detector=args.import_labels_into,
                importer=args.label_importer,
                filepath=args.label_importer_file,
            )
            if cli_progress.get_format() == "text":
                print("", flush=True)
        else:
            from vtscore.cli import import_labels_into_detector_from_file

            try:
                applied, skipped = import_labels_into_detector_from_file(
                    args.import_labels_into,
                    args.label_importer,
                    args.label_importer_file,
                )
                cli_progress.emit(
                    "labels_imported",
                    text=(
                        f"Imported {applied} label(s) into detector "
                        f"'{args.import_labels_into}' (skipped {skipped} duplicate/invalid)."
                    ),
                    detector=args.import_labels_into,
                    applied=applied,
                    skipped=skipped,
                )
            except (FileNotFoundError, ValueError) as exc:
                cli_progress.emit_error(f"importing labels: {exc}")
                sys.exit(1)


def _dispatch_autodetect(
    args,
    parser,
    importer,
    exporter,
    exporter_field_values,
    settings_path,
    chunk_size,
    dry_run,
    stream_results,
    keep_negatives,
) -> None:
    """Run the autodetect workflow via the importer- or pickle-file code path."""
    from vtscore.cli import (
        autodetect_importer_main,
        autodetect_importer_main_chunked,
        autodetect_main,
        autodetect_main_chunked,
    )

    # Pick the entry point and its source-specific leading arguments; every
    # other argument is identical across the four, so the call is written once.
    entry_point: Callable[..., None]
    source_args: tuple[Any, ...]
    if args.importer:
        field_values = {f.key: getattr(args, f.key, f.default) for f in importer.fields}
        if chunk_size:
            entry_point, source_args = autodetect_importer_main_chunked, (args.importer, field_values, chunk_size)
        else:
            entry_point, source_args = autodetect_importer_main, (args.importer, field_values)
    elif args.dataset:
        if chunk_size:
            entry_point, source_args = autodetect_main_chunked, (args.dataset, chunk_size)
        else:
            entry_point, source_args = autodetect_main, (args.dataset,)
    else:
        parser.error("--autodetect requires either --dataset <file.pkl> or --importer <name>")

    entry_point(
        *source_args,
        settings_path,
        args.exporter,
        exporter_field_values,
        dry_run=dry_run,
        stream_results=stream_results,
        keep_negatives=keep_negatives,
    )


def _run_autodetect(args, parser, importer, exporter) -> None:
    """Drive the ``--autodetect`` flow: progress wiring, auth, optional label
    import, then dispatch to the importer/pickle code path."""
    # Wire the CLI progress format (text/json) before any pipeline call
    # produces output. In JSON mode we also re-route the process-wide media
    # progress callback from update_progress (which resolves per-thread, so
    # headless it reaches nothing) to an NDJSON emitter on stdout.
    from vtscore import cli_progress
    from vtscore.concurrency.notifications import notifications

    cli_progress.set_format(args.progress_format)
    if args.progress_format == "json":
        set_progress_callback(cli_progress.progress_callback)
    # Plugin notifications become toasts in the GUI; headless there is no
    # browser to toast at, so print them instead. Without this the CLI is the
    # one place where "keep going but tell the user" silently drops the second
    # half of its promise.
    notifications.subscribe(cli_progress.notification_subscriber)

    _authenticate_cli_user(args, parser)

    # Collect exporter field values if an exporter was specified
    exporter_field_values = None
    if exporter:
        exporter_field_values = {f.key: getattr(args, f.key, f.default) for f in exporter.fields}

    settings_path = getattr(args, "settings", None)

    chunk_size = getattr(args, "chunk_size", None)

    dry_run = bool(getattr(args, "dry_run", False))

    stream_results = bool(getattr(args, "stream_results", False))
    keep_negatives = bool(getattr(args, "keep_negatives", False))
    if stream_results and not chunk_size:
        parser.error("--stream-results requires --chunk-size N (it streams chunk by chunk)")
    if keep_negatives and not stream_results:
        parser.error("--keep-negatives only applies with --stream-results")

    _maybe_import_labels(args, parser, settings_path, dry_run)

    _dispatch_autodetect(
        args,
        parser,
        importer,
        exporter,
        exporter_field_values,
        settings_path,
        chunk_size,
        dry_run,
        stream_results,
        keep_negatives,
    )


def _run_server(args, app, initialize_server) -> None:
    """Activate the chosen login provider, run the port preflight, and serve."""
    # Activate the chosen login provider before starting the server.
    login_choice = getattr(args, "login", None)
    if login_choice == "trivial":
        from vtsearch.auth import TrivialLoginProvider, set_login_provider

        set_login_provider(TrivialLoginProvider())
        print("\U0001f511 Trivial login enabled \u2014 users will be prompted for a username", flush=True)
    elif login_choice == "api_key":
        from vtsearch.auth import ApiKeyLoginProvider, set_login_provider
        from vtscore.config import DATA_DIR

        provider = ApiKeyLoginProvider()
        set_login_provider(provider)
        print(
            f"\U0001f511 API-key login enabled \u2014 reading keys from {DATA_DIR / 'api_keys.json'}",
            flush=True,
        )

    # Single-instance lock FIRST -- before the model load -- so a
    # duplicate launch fails in milliseconds. _preflight_port only
    # catches an already-*listening* instance; this also covers the
    # model-loading window when the port is briefly still free. Held
    # for the process lifetime (released on exit).
    #
    # Port precedence: --port > VTSEARCH_PORT > 5000. Lets several instances
    # coexist on one host (e.g. co-located SLURM jobs on a multi-GPU node),
    # where a single shared :5000 would collide.
    port = args.port or int(os.environ.get("VTSEARCH_PORT", "5000"))
    _instance_lock = _acquire_single_instance_lock(port)  # noqa: F841
    # Catch a leftover instance before the expensive model load, so the
    # user is prompted up front instead of after a long startup.
    _preflight_port(port)
    initialize_server(mode_label="LOCAL" if args.local else "PRODUCTION")
    print(f"\U0001f310 Open http://localhost:{port} in your browser", flush=True)
    # The dev server spawns a thread per connection (threaded=True): there is
    # no bounded worker pool for long-lived SSE streams to starve, so the
    # gunicorn-oriented connection cap is pure downside here (#2816).
    from vtscore.concurrency.events import uncap_sse_connections

    uncap_sse_connections()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


def main(app, initialize_server) -> None:
    """Parse argv and dispatch to the list-plugins, pipeline, autodetect, or
    server path. Called by ``app.py``'s ``__main__`` block."""
    parser = _build_parser()

    # Two-pass parsing: first pass gets --importer and --exporter names;
    # second pass adds their plugin-specific arguments and re-parses.
    args, remaining = parser.parse_known_args()

    _maybe_print_help(args, parser)
    _maybe_list_plugins(args, parser)
    _maybe_run_pipeline(args, parser, remaining)
    args, importer, exporter = _resolve_plugins(args, parser, remaining)

    _apply_verbosity(args)
    _apply_admin_overrides(args, parser)

    if args.autodetect:
        _run_autodetect(args, parser, importer, exporter)
    else:
        _run_server(args, app, initialize_server)
