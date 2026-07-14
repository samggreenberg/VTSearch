"""In-process coverage for ``vtsearch.cli_main`` — the ``python app.py``
argparse build, two-pass parsing, ``_apply_*`` overrides, and the
list-plugins / pipeline / autodetect / server dispatch.

Unlike ``test_cli_main_subprocess.py`` (which spawns a real ``python
app.py`` and is ``slow``-marked), these call ``cli_main.main`` and its
helpers directly with the heavy stages mocked: the ``vtscore.cli``
``autodetect_*`` functions, the pipeline runner, and ``_run_server`` are
patched to record their calls instead of loading media or serving. This
exercises the flag-validation, wiring, and error-exit paths that the
subprocess tests can't cheaply reach, at ~0ms each.
"""

from __future__ import annotations

import logging
import sys

import pytest

from vtsearch import cli_main
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def _restore_cli_overrides():
    """Snapshot and restore the process-level CLI override globals.

    ``settings.reset()`` (run by ``isolated_settings`` around each test)
    does *not* clear these, so a test that applies a valid
    ``--solo-media-type`` / ``--hide-plugin`` / etc. would otherwise leak
    the value into later tests.
    """
    saved = (
        settings_mod._cli_solo_media_type,
        dict(settings_mod._cli_hidden_plugins),
        dict(settings_mod._cli_solo_embedders),
        settings_mod._cli_dataset_max_age_days,
        settings_mod._cli_support_email,
    )
    yield
    (
        settings_mod._cli_solo_media_type,
        hidden,
        solo_emb,
        settings_mod._cli_dataset_max_age_days,
        settings_mod._cli_support_email,
    ) = saved
    settings_mod._cli_hidden_plugins.clear()
    settings_mod._cli_hidden_plugins.update(hidden)
    settings_mod._cli_solo_embedders.clear()
    settings_mod._cli_solo_embedders.update(solo_emb)


def _run_main(monkeypatch, argv, app=None, initialize_server=None):
    """Invoke ``cli_main.main`` with ``argv`` as the process arguments."""
    monkeypatch.setattr(sys, "argv", ["app.py", *argv])
    cli_main.main(app, initialize_server)


class _AutodetectRecorder:
    """Patches the four ``vtscore.cli`` autodetect entry points to record calls."""

    def __init__(self, monkeypatch):
        self.calls = {}
        import vtscore.cli as vtcli

        for fn in (
            "autodetect_main",
            "autodetect_main_chunked",
            "autodetect_importer_main",
            "autodetect_importer_main_chunked",
        ):
            monkeypatch.setattr(vtcli, fn, self._make(fn))

    def _make(self, name):
        def _rec(*args, **kwargs):
            self.calls[name] = (args, kwargs)

        return _rec


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        parser = cli_main._build_parser()
        args = parser.parse_args([])
        assert args.local is False
        assert args.port is None
        assert args.verbose == 0
        assert args.autodetect is False
        assert args.progress_format == "text"
        assert args.output_format == "plain"
        assert args.label_importer == "server_json_file"
        assert args.hide_plugin == []
        assert args.solo_embedders is None
        assert args.dry_run is False
        assert args.list_plugins is False

    def test_autodetect_flags(self):
        parser = cli_main._build_parser()
        args = parser.parse_args(["--autodetect", "--dataset", "x.pkl", "--chunk-size", "50"])
        assert args.autodetect is True
        assert args.dataset == "x.pkl"
        assert args.chunk_size == 50

    def test_verbose_counts(self):
        parser = cli_main._build_parser()
        assert parser.parse_args(["-v"]).verbose == 1
        assert parser.parse_args(["-vv"]).verbose == 2

    def test_port_and_login(self):
        parser = cli_main._build_parser()
        args = parser.parse_args(["--port", "8080", "--login", "api_key"])
        assert args.port == 8080
        assert args.login == "api_key"

    def test_invalid_login_choice_exits(self):
        parser = cli_main._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--login", "nope"])

    def test_family_shortcut_sets_list_plugins(self):
        parser = cli_main._build_parser()
        args = parser.parse_args(["--list-importers"])
        assert args.list_plugins is True
        assert args.plugin_family == "importers"

    def test_repeatable_hide_and_solo_embedder(self):
        parser = cli_main._build_parser()
        args = parser.parse_args(
            ["--hide-plugin", "importers:a", "--hide-plugin", "exporters:b", "--solo-embedder", "image=x"]
        )
        assert args.hide_plugin == ["importers:a", "exporters:b"]
        assert args.solo_embedders == ["image=x"]


# ---------------------------------------------------------------------------
# --list-plugins early exit
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_list_plugins_names_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--list-plugins", "--format", "names"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip()

    def test_list_family_shortcut_json(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--list-importers", "--format", "json"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "importers" in out

    def test_unknown_plugin_family_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--list-plugins", "--plugin-family", "no_such_family"])
        assert exc.value.code == 2
        assert "Unknown plugin family" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --pipeline dispatch
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pipeline_runs_file_and_exits(self, monkeypatch):
        import vtscore.cli_pipeline as pipe

        seen = {}
        monkeypatch.setattr(pipe, "run_pipeline_file", lambda p: seen.setdefault("path", p))
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--pipeline", "flow.yaml"])
        assert exc.value.code == 0
        assert seen["path"] == "flow.yaml"

    def test_pipeline_conflicts_with_autodetect_flag(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--pipeline", "flow.yaml", "--dataset", "x.pkl"])
        assert exc.value.code == 2
        assert "--pipeline cannot be combined with --dataset" in capsys.readouterr().err

    def test_pipeline_rejects_extra_flags(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--pipeline", "flow.yaml", "--totally-unknown", "1"])
        assert exc.value.code == 2
        assert "does not accept extra flags" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Autodetect wiring (heavy stages mocked)
# ---------------------------------------------------------------------------


class TestAutodetectWiring:
    def test_pickle_path(self, monkeypatch):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--settings", "s.json"])
        assert "autodetect_main" in rec.calls
        args, _ = rec.calls["autodetect_main"]
        assert args[0] == "x.pkl"
        assert args[1] == "s.json"

    def test_pickle_chunked_path(self, monkeypatch):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--chunk-size", "100"])
        assert "autodetect_main_chunked" in rec.calls
        args, _ = rec.calls["autodetect_main_chunked"]
        assert args[0] == "x.pkl"
        assert args[1] == 100

    def test_importer_path_passes_field_values(self, monkeypatch):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(
            monkeypatch,
            ["--autodetect", "--importer", "server_folder", "--path", "/data/sounds", "--media-type", "audio"],
        )
        assert "autodetect_importer_main" in rec.calls
        args, _ = rec.calls["autodetect_importer_main"]
        assert args[0] == "server_folder"
        field_values = args[1]
        assert field_values["path"] == "/data/sounds"
        assert field_values["media_type"] == "audio"

    def test_importer_chunked_path(self, monkeypatch):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(
            monkeypatch,
            [
                "--autodetect",
                "--importer",
                "server_folder",
                "--path",
                "/data/sounds",
                "--media-type",
                "audio",
                "--chunk-size",
                "25",
            ],
        )
        assert "autodetect_importer_main_chunked" in rec.calls
        args, _ = rec.calls["autodetect_importer_main_chunked"]
        assert args[2] == 25

    def test_exporter_field_values_flow_through(self, monkeypatch):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(
            monkeypatch,
            ["--autodetect", "--dataset", "x.pkl", "--exporter", "server_json_file", "--filepath", "/out.json"],
        )
        args, _ = rec.calls["autodetect_main"]
        # autodetect_main(dataset, settings, exporter_name, exporter_field_values, dry_run=...)
        assert args[2] == "server_json_file"
        assert args[3]["filepath"] == "/out.json"


# ---------------------------------------------------------------------------
# Autodetect flag validation / error exits
# ---------------------------------------------------------------------------


class TestAutodetectValidation:
    def test_missing_target_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect"])
        assert exc.value.code == 2
        assert "requires either --dataset" in capsys.readouterr().err

    def test_stream_results_requires_chunk_size(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--stream-results"])
        assert exc.value.code == 2
        assert "--stream-results requires --chunk-size" in capsys.readouterr().err

    def test_keep_negatives_requires_stream_results(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--chunk-size", "10", "--keep-negatives"])
        assert exc.value.code == 2
        assert "--keep-negatives only applies with --stream-results" in capsys.readouterr().err

    def test_unknown_importer_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--importer", "no_such_importer"])
        assert exc.value.code == 2
        assert "Unknown importer: no_such_importer" in capsys.readouterr().err

    def test_unknown_exporter_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--exporter", "no_such_exporter"])
        assert exc.value.code == 2
        assert "Unknown exporter: no_such_exporter" in capsys.readouterr().err

    def test_import_labels_requires_file(self, monkeypatch, capsys):
        rec = _AutodetectRecorder(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--import-labels-into", "det"])
        assert exc.value.code == 2
        assert "--import-labels-into requires --label-importer-file" in capsys.readouterr().err
        # Dispatch never reached.
        assert rec.calls == {}


# ---------------------------------------------------------------------------
# --user / --api-key authentication
# ---------------------------------------------------------------------------


class TestCliUserAuth:
    def test_user_without_api_key_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--user", "alice"])
        assert exc.value.code == 2
        assert "--user requires --api-key" in capsys.readouterr().err

    def test_api_key_without_user_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--api-key", "secret"])
        assert exc.value.code == 2
        assert "--api-key requires --user" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _apply_* process-level overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_solo_media_type_invalid_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-media-type", "not_a_type"])
        assert exc.value.code == 2
        assert "Unknown --solo-media-type" in capsys.readouterr().err

    def test_solo_media_type_valid_is_stashed(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: recorded.setdefault("ran", True))
        _run_main(monkeypatch, ["--solo-media-type", "audio"], app="APP", initialize_server="INIT")
        assert recorded["ran"] is True
        assert settings_mod.get_cli_solo_media_type() == "audio"

    def test_hide_plugin_missing_colon_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--hide-plugin", "importers"])
        assert exc.value.code == 2
        assert "expects FAMILY:NAME" in capsys.readouterr().err

    def test_hide_plugin_unknown_family_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--hide-plugin", "no_such_family:foo"])
        assert exc.value.code == 2
        assert "Unknown --hide-plugin family" in capsys.readouterr().err

    def test_hide_plugin_valid_is_stashed(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, ["--hide-plugin", "importers:server_folder"])
        assert "server_folder" in settings_mod._cli_hidden_plugins.get("importers", set())

    def test_solo_embedder_missing_equals_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-embedder", "image"])
        assert exc.value.code == 2
        assert "Expected TYPE=EMBEDDER" in capsys.readouterr().err

    def test_solo_embedder_unknown_type_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-embedder", "not_a_type=whatever"])
        assert exc.value.code == 2
        assert "Unknown mediaType in --solo-embedder" in capsys.readouterr().err

    def test_solo_embedder_unknown_embedder_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-embedder", "image=definitely_not_an_embedder"])
        assert exc.value.code == 2
        assert "Unknown embedder in --solo-embedder" in capsys.readouterr().err

    def test_solo_embedder_valid_is_stashed(self, monkeypatch):
        from vtscore.media import all_type_ids, embedders_for_type

        # Pick a real (media_type, embedder) pair from the live registry.
        pair = None
        for mt in sorted(all_type_ids()):
            embs = embedders_for_type(mt)
            if embs:
                pair = (mt, embs[0].name)
                break
        assert pair is not None, "no embedders registered"
        mt, emb = pair

        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, ["--solo-embedder", f"{mt}={emb}"])
        assert settings_mod.get_cli_solo_embedders().get(mt) == emb

    def test_dataset_max_age_non_positive_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--dataset-max-age-days", "0"])
        assert exc.value.code == 2
        assert "must be a positive integer" in capsys.readouterr().err

    def test_dataset_max_age_valid_is_stashed(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, ["--dataset-max-age-days", "7"])
        assert settings_mod.get_cli_dataset_max_age_days() == 7

    def test_support_email_blank_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--support-email", "   "])
        assert exc.value.code == 2
        assert "must be a non-empty address" in capsys.readouterr().err

    def test_support_email_valid_is_stashed(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, ["--support-email", "help@example.com"])
        assert settings_mod.get_cli_support_email() == "help@example.com"


class TestApplyVerbosity:
    def test_verbose_raises_log_level(self, monkeypatch):
        import argparse

        root = logging.getLogger()
        saved_level = root.level
        try:
            root.setLevel(logging.WARNING)
            cli_main._apply_verbosity(argparse.Namespace(verbose=1))
            assert root.level <= logging.INFO
        finally:
            root.setLevel(saved_level)

    def test_no_verbose_is_noop(self):
        import argparse

        root = logging.getLogger()
        saved_level = root.level
        try:
            root.setLevel(logging.ERROR)
            cli_main._apply_verbosity(argparse.Namespace(verbose=0))
            assert root.level == logging.ERROR
        finally:
            root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Server dispatch
# ---------------------------------------------------------------------------


class TestServerDispatch:
    def test_no_autodetect_runs_server(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli_main, "_run_server", lambda args, app, init: seen.update(app=app, init=init))
        _run_main(monkeypatch, [], app="APP", initialize_server="INIT")
        assert seen == {"app": "APP", "init": "INIT"}
