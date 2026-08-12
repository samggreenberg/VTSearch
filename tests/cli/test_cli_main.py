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
        settings_mod._cli_semantic_only,
    )
    yield
    (
        settings_mod._cli_solo_media_type,
        hidden,
        solo_emb,
        settings_mod._cli_dataset_max_age_days,
        settings_mod._cli_support_email,
        settings_mod._cli_semantic_only,
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
# -h / --help (plugin-aware)
# ---------------------------------------------------------------------------


class TestHelp:
    """``--help`` must list the selected plugin's flags.

    argparse's built-in help action would fire during the *first* parse pass,
    before ``_resolve_plugins`` registers the importer/exporter flags, so the
    doc-advertised ``--importer <name> --help`` discovery path printed only the
    base flags. ``cli_main`` registers help as a plain flag and prints it after
    plugin resolution instead.
    """

    def test_bare_help_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--autodetect" in out
        # No plugin selected: the importer's own flags stay out of the way.
        assert "--dig-archives" not in out

    def test_short_h_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["-h"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_help_lists_importer_flags(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--importer", "server_folder", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--path", "--media-type", "--recursive", "--dig-archives"):
            assert flag in out, f"{flag} missing from --importer server_folder --help"

    def test_help_lists_exporter_flags(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--exporter", "webhook", "--help"])
        assert exc.value.code == 0
        assert "--auth-header" in capsys.readouterr().out

    def test_help_with_unknown_importer_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--importer", "no_such_importer", "--help"])
        assert exc.value.code == 2
        assert "Unknown importer" in capsys.readouterr().err

    def test_help_does_not_start_server(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_run_server", lambda *a, **k: pytest.fail("server started"))
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--help"])
        assert exc.value.code == 0


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

    def test_list_plugins_plain_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--list-plugins"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip()

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

    def test_json_progress_format_reroutes_callback(self, monkeypatch):
        _AutodetectRecorder(monkeypatch)
        seen = {}
        import vtscore.media as media_mod

        monkeypatch.setattr(media_mod, "set_progress_callback", lambda cb: seen.setdefault("cb", cb))
        # cli_main imports set_progress_callback at module load, so patch the bound name too.
        monkeypatch.setattr(cli_main, "set_progress_callback", lambda cb: seen.setdefault("cb", cb))
        _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--progress-format", "json"])
        assert "cb" in seen


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
# --import-labels-into wiring
# ---------------------------------------------------------------------------


class TestImportLabels:
    def test_dry_run_label_import_emits_plan_and_dispatches(self, monkeypatch, capsys):
        rec = _AutodetectRecorder(monkeypatch)
        _run_main(
            monkeypatch,
            [
                "--autodetect",
                "--dataset",
                "x.pkl",
                "--import-labels-into",
                "det",
                "--label-importer-file",
                "labels.json",
                "--dry-run",
            ],
        )
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "det" in out
        # Dry-run still dispatches to the (mocked) autodetect run.
        assert "autodetect_main" in rec.calls

    def test_dry_run_label_import_json_format(self, monkeypatch, capsys):
        self_rec = _AutodetectRecorder(monkeypatch)
        _run_main(
            monkeypatch,
            [
                "--autodetect",
                "--dataset",
                "x.pkl",
                "--progress-format",
                "json",
                "--import-labels-into",
                "det",
                "--label-importer-file",
                "labels.json",
                "--dry-run",
            ],
        )
        # NDJSON event carries the dry-run marker; no trailing text print.
        out = capsys.readouterr().out
        assert "labels_import_dry_run" in out
        assert "autodetect_main" in self_rec.calls

    def test_real_label_import_calls_importer(self, monkeypatch, tmp_path):
        rec = _AutodetectRecorder(monkeypatch)
        import vtscore.cli as vtcli

        seen = {}

        def _fake_import(detector, importer, filepath):
            seen.update(detector=detector, importer=importer, filepath=filepath)
            return (3, 1)

        monkeypatch.setattr(vtcli, "import_labels_into_detector_from_file", _fake_import)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        _run_main(
            monkeypatch,
            [
                "--autodetect",
                "--dataset",
                "x.pkl",
                "--settings",
                str(settings_path),
                "--import-labels-into",
                "det",
                "--label-importer",
                "server_json_file",
                "--label-importer-file",
                "labels.json",
            ],
        )
        assert seen == {"detector": "det", "importer": "server_json_file", "filepath": "labels.json"}
        assert "autodetect_main" in rec.calls

    def test_label_import_failure_exits_one(self, monkeypatch, capsys):
        _AutodetectRecorder(monkeypatch)
        import vtscore.cli as vtcli

        def _boom(*a, **k):
            raise ValueError("bad labels file")

        monkeypatch.setattr(vtcli, "import_labels_into_detector_from_file", _boom)
        with pytest.raises(SystemExit) as exc:
            _run_main(
                monkeypatch,
                [
                    "--autodetect",
                    "--dataset",
                    "x.pkl",
                    "--import-labels-into",
                    "det",
                    "--label-importer-file",
                    "labels.json",
                ],
            )
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# --user / --api-key authentication
# ---------------------------------------------------------------------------


class TestCliUserAuth:
    def test_authenticate_user_happy_path(self, monkeypatch):
        import argparse

        import vtsearch.auth as auth_mod

        class _FakeProvider:
            name = "api_key"

            def __init__(self, keys_file=None):
                pass

            def is_authenticated(self, req):
                return True

            def get_user(self, req):
                return "alice"

        set_users = []
        provs = []
        monkeypatch.setattr(auth_mod, "ApiKeyLoginProvider", _FakeProvider)
        monkeypatch.setattr(auth_mod, "set_thread_user", lambda u: set_users.append(u))
        monkeypatch.setattr(auth_mod, "set_login_provider", lambda p: provs.append(p))

        parser = cli_main._build_parser()
        cli_main._authenticate_cli_user(argparse.Namespace(user="alice", api_key="key"), parser)
        assert set_users == ["alice"]
        # The authenticated provider must be installed process-wide, not merely
        # used for the key check: get_user_data_dir() consults it to resolve
        # data/<user>/user_settings.json.
        assert [type(p) for p in provs] == [_FakeProvider]

    def test_authenticate_installs_provider_so_user_dir_resolves(self, monkeypatch):
        """After --user auth, get_user_data_dir() resolves per-user paths."""
        import argparse

        import vtsearch.auth as auth_mod

        class _FakeProvider(auth_mod.LoginProvider):
            name = "api_key"

            def __init__(self, keys_file=None):
                pass

            def is_authenticated(self, req):
                return True

            def get_user(self, req):
                return "alice"

            def get_user_data_dir(self, username, base_dir):
                return base_dir / username

        monkeypatch.setattr(auth_mod, "ApiKeyLoginProvider", _FakeProvider)
        from vtscore.config import DATA_DIR

        original = auth_mod.get_login_provider()
        try:
            parser = cli_main._build_parser()
            cli_main._authenticate_cli_user(argparse.Namespace(user="alice", api_key="key"), parser)
            assert auth_mod.get_user_data_dir("alice") == DATA_DIR / "alice"
            # Resolution also works off the thread-local user set by the CLI.
            assert auth_mod.get_user_data_dir() == DATA_DIR / "alice"
        finally:
            auth_mod.set_login_provider(original)
            auth_mod.set_thread_user(None)

    def test_authenticate_wrong_user_errors(self, monkeypatch, capsys):
        import argparse

        import vtsearch.auth as auth_mod

        class _FakeProvider:
            def __init__(self, keys_file=None):
                pass

            def is_authenticated(self, req):
                return True

            def get_user(self, req):
                return "bob"

        monkeypatch.setattr(auth_mod, "ApiKeyLoginProvider", _FakeProvider)
        parser = cli_main._build_parser()
        with pytest.raises(SystemExit) as exc:
            cli_main._authenticate_cli_user(argparse.Namespace(user="alice", api_key="key"), parser)
        assert exc.value.code == 2
        assert "authenticates as 'bob'" in capsys.readouterr().err

    def test_authenticate_invalid_key_errors(self, monkeypatch, capsys):
        import argparse

        import vtsearch.auth as auth_mod

        class _FakeProvider:
            def __init__(self, keys_file=None):
                pass

            def is_authenticated(self, req):
                return False

        monkeypatch.setattr(auth_mod, "ApiKeyLoginProvider", _FakeProvider)
        parser = cli_main._build_parser()
        with pytest.raises(SystemExit) as exc:
            cli_main._authenticate_cli_user(argparse.Namespace(user="alice", api_key="bad"), parser)
        assert exc.value.code == 2
        assert "Invalid --api-key" in capsys.readouterr().err

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

    def test_hide_plugin_empty_half_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--hide-plugin", "importers:"])
        assert exc.value.code == 2
        assert "empty family or name" in capsys.readouterr().err

    def test_solo_embedder_empty_half_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-embedder", "image="])
        assert exc.value.code == 2
        assert "must be non-empty" in capsys.readouterr().err

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

    def test_semantic_only_flag_is_stashed(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, ["--semantic-only"])
        assert settings_mod.get_cli_semantic_only() is True
        assert settings_mod.get_effective_semantic_only() is True

    def test_semantic_only_omitted_leaves_the_setting_in_charge(self, monkeypatch):
        """No flag means no override: the persisted server setting decides, so
        a deployment can turn the lock on from settings.json alone."""
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        _run_main(monkeypatch, [])
        assert settings_mod.get_cli_semantic_only() is None


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

    def test_unknown_flag_without_plugins_errors(self, monkeypatch, capsys):
        # No importer/exporter to consume it, no pipeline: the leftover flag
        # is surfaced by a full re-parse in _resolve_plugins.
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--totally-unknown-flag"])
        assert exc.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


class _FakeApp:
    def __init__(self):
        self.ran: dict = {}

    def run(self, **kwargs):
        self.ran = kwargs


class TestRunServer:
    def _patch_preflight(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_acquire_single_instance_lock", lambda port: object())
        monkeypatch.setattr(cli_main, "_preflight_port", lambda port: None)

    def test_run_server_trivial_login(self, monkeypatch):
        import argparse

        import vtsearch.auth as auth_mod

        provs = []
        monkeypatch.setattr(auth_mod, "set_login_provider", lambda p: provs.append(p))
        self._patch_preflight(monkeypatch)

        init_calls = {}
        app = _FakeApp()
        cli_main._run_server(
            argparse.Namespace(login="trivial", port=5055, local=True),
            app,
            lambda **kw: init_calls.update(kw),
        )
        assert app.ran["port"] == 5055
        assert init_calls["mode_label"] == "LOCAL"
        assert len(provs) == 1

    def test_run_server_api_key_login_and_default_port(self, monkeypatch):
        import argparse

        import vtsearch.auth as auth_mod

        provs = []
        monkeypatch.setattr(auth_mod, "set_login_provider", lambda p: provs.append(p))
        self._patch_preflight(monkeypatch)
        monkeypatch.delenv("VTSEARCH_PORT", raising=False)

        app = _FakeApp()
        cli_main._run_server(
            argparse.Namespace(login="api_key", port=None, local=False),
            app,
            lambda **kw: None,
        )
        # Falls back to the default 5000 when neither --port nor env is set.
        assert app.ran["port"] == 5000
        assert len(provs) == 1

    def test_run_server_no_login_provider(self, monkeypatch):
        import argparse

        import vtsearch.auth as auth_mod

        provs = []
        monkeypatch.setattr(auth_mod, "set_login_provider", lambda p: provs.append(p))
        self._patch_preflight(monkeypatch)
        monkeypatch.setenv("VTSEARCH_PORT", "6001")

        app = _FakeApp()
        cli_main._run_server(argparse.Namespace(login=None, port=None, local=True), app, lambda **kw: None)
        # No --login: no provider activated; port comes from VTSEARCH_PORT.
        assert provs == []
        assert app.ran["port"] == 6001


class TestSoloEmbedderCrossType:
    def test_embedder_valid_but_wrong_type_errors(self, monkeypatch, capsys):
        from vtscore.media import all_embedders, all_type_ids, embedders_for_type

        all_names = {e.name for e in all_embedders()}
        # Find (type, embedder) where the embedder exists globally but is not
        # registered for that type.
        pair = None
        for mt in sorted(all_type_ids()):
            for_type = {e.name for e in embedders_for_type(mt)}
            other = all_names - for_type
            if other:
                pair = (mt, sorted(other)[0])
                break
        if pair is None:
            pytest.skip("no cross-type embedder pair available")
        mt, emb = pair
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--solo-embedder", f"{mt}={emb}"])
        assert exc.value.code == 2
        assert "not registered for media type" in capsys.readouterr().err
