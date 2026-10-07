"""The admin's ``--on-dataset-imported`` hooks (#4616).

:mod:`vtsearch.import_hooks` turns ``module:function`` specs from the flag or
``VTSEARCH_ON_DATASET_IMPORTED`` into the functions the load pipeline calls when
a user's import ends.  What it owes an admin: a bad spec fails at startup
(loudly on the CLI, as a warning under gunicorn), the flag wins over the env
var, hooks run in the order named, and one failing hook neither stops the next
nor fails the import.  The pipeline side (when ``on_finished`` fires, and with
what) is covered in ``tests/datasets/test_load_post_load.py``.
"""

from __future__ import annotations

import pytest

from vtscore.datasets.import_event import FAILED, SUCCEEDED, DatasetImported, ImportOutcome
from vtsearch import cli_main, import_hooks

#: What the module-level hooks below were called with, in call order.
_calls: list[tuple[str, DatasetImported]] = []

_SELF = __name__


def first_hook(event: DatasetImported) -> None:
    _calls.append(("first", event))


def second_hook(event: DatasetImported) -> None:
    _calls.append(("second", event))


def raising_hook(event: DatasetImported) -> None:
    _calls.append(("raising", event))
    raise RuntimeError("mail server unreachable")


NOT_A_FUNCTION = "just a string"


@pytest.fixture(autouse=True)
def _clear_calls():
    _calls.clear()
    yield
    _calls.clear()


def _event(outcome: ImportOutcome = SUCCEEDED) -> DatasetImported:
    return DatasetImported(
        outcome=outcome,
        dataset_id="ds-1" if outcome == SUCCEEDED else "",
        name="Birds",
        user="alice",
        media_type="audio",
        n_media=3 if outcome == SUCCEEDED else 0,
        origin={"importer": "server_folder", "params": {}},
        error="" if outcome == SUCCEEDED else "disk full",
    )


class TestResolveHook:
    def test_resolves_module_colon_function(self):
        assert import_hooks.resolve_hook(f"{_SELF}:first_hook") is first_hook

    def test_a_missing_module_names_the_flag_and_the_spec(self):
        with pytest.raises(import_hooks.HookSpecError, match=r"--on-dataset-imported 'no_such_mod:notify'"):
            import_hooks.resolve_hook("no_such_mod:notify")

    def test_a_missing_function_is_an_error(self):
        with pytest.raises(import_hooks.HookSpecError, match="could not be loaded"):
            import_hooks.resolve_hook(f"{_SELF}:no_such_function")

    def test_a_non_callable_is_an_error(self):
        with pytest.raises(import_hooks.HookSpecError, match="names a str, not a function"):
            import_hooks.resolve_hook(f"{_SELF}:NOT_A_FUNCTION")

    def test_an_empty_spec_is_an_error(self):
        with pytest.raises(import_hooks.HookSpecError, match="expects module:function"):
            import_hooks.resolve_hook("  ")


class TestFire:
    def test_nothing_configured_is_a_no_op(self):
        import_hooks.fire_dataset_imported(_event())
        assert _calls == []
        assert import_hooks.describe() is None

    def test_hooks_run_in_the_order_named(self):
        import_hooks.configure([f"{_SELF}:second_hook", f"{_SELF}:first_hook"], source=import_hooks.FLAG)
        event = _event(FAILED)
        import_hooks.fire_dataset_imported(event)
        assert _calls == [("second", event), ("first", event)]

    def test_a_raising_hook_is_logged_and_the_next_still_runs(self, caplog):
        import_hooks.configure([f"{_SELF}:raising_hook", f"{_SELF}:first_hook"], source=import_hooks.FLAG)
        import_hooks.fire_dataset_imported(_event())
        assert [name for name, _ in _calls] == ["raising", "first"]
        assert f"{_SELF}:raising_hook failed" in caplog.text

    def test_a_bad_spec_leaves_the_hooks_in_force(self):
        import_hooks.configure([f"{_SELF}:first_hook"], source=import_hooks.FLAG)
        with pytest.raises(import_hooks.HookSpecError):
            import_hooks.configure([f"{_SELF}:second_hook", "no_such_mod:notify"], source=import_hooks.FLAG)
        assert import_hooks.describe() == f"{_SELF}:first_hook (from --on-dataset-imported)"


class TestEnvironment:
    def test_comma_separated_env_var(self, monkeypatch):
        monkeypatch.setenv(import_hooks.ENV, f"{_SELF}:first_hook, {_SELF}:second_hook")
        import_hooks.configure_from_env()
        import_hooks.fire_dataset_imported(_event())
        assert [name for name, _ in _calls] == ["first", "second"]
        assert import_hooks.describe() == f"{_SELF}:first_hook, {_SELF}:second_hook (from {import_hooks.ENV})"

    def test_a_bad_env_value_warns_names_the_variable_and_installs_nothing(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setenv(import_hooks.ENV, f"{_SELF}:first_hook,no_such_mod:notify")
        import_hooks.configure_from_env(warn=warnings.append)
        assert import_hooks.describe() is None
        assert len(warnings) == 1
        assert import_hooks.ENV in warnings[0]
        assert import_hooks.FLAG not in warnings[0]

    def test_a_blank_env_var_sets_nothing(self, monkeypatch):
        monkeypatch.setenv(import_hooks.ENV, " , ")
        import_hooks.configure_from_env(warn=pytest.fail)
        assert import_hooks.describe() is None

    def test_an_explicit_flag_wins_over_the_env(self, monkeypatch):
        import_hooks.configure([f"{_SELF}:first_hook"], source=import_hooks.FLAG)
        monkeypatch.setenv(import_hooks.ENV, f"{_SELF}:second_hook")
        import_hooks.configure_from_env()
        assert import_hooks.describe() == f"{_SELF}:first_hook (from --on-dataset-imported)"


class TestCommandLine:
    @staticmethod
    def _run_main(monkeypatch, argv):
        import sys

        monkeypatch.setattr(sys, "argv", ["app.py", *argv])
        monkeypatch.setattr(cli_main, "_run_server", lambda *a: None)
        cli_main.main(None, None)

    def test_repeated_flag_is_installed_in_order(self, monkeypatch):
        self._run_main(
            monkeypatch,
            ["--on-dataset-imported", f"{_SELF}:second_hook", "--on-dataset-imported", f"{_SELF}:first_hook"],
        )
        assert import_hooks.describe() == f"{_SELF}:second_hook, {_SELF}:first_hook (from --on-dataset-imported)"

    def test_a_bad_spec_stops_startup(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            self._run_main(monkeypatch, ["--on-dataset-imported", "no_such_mod:notify"])
        assert exc.value.code == 2
        assert "could not be loaded" in capsys.readouterr().err


class TestImportRoutesPassTheHook:
    """User-started imports hand the pipeline the hook dispatcher."""

    def test_generic_importer_route(self, client, monkeypatch):
        import vtsearch.routes.datasets.staging as staging_mod

        captured: dict = {}

        def fake_run(importer, field_values, *, post_load=None, on_finished=None):
            captured["on_finished"] = on_finished
            return "task"

        monkeypatch.setattr(staging_mod, "_run_importer_in_background", fake_run)
        resp = client.post("/api/dataset/import/synthetic", json={"autofind": "false"})
        assert resp.status_code == 200, resp.get_json()
        assert captured["on_finished"] is import_hooks.fire_dataset_imported

    def test_demo_route(self, client, monkeypatch):
        import vtsearch.routes.datasets.load as load_mod
        from vtscore.datasets import DEMO_DATASETS

        captured: dict = {}

        def fake_run(importer, field_values, *, post_load=None, on_finished=None):
            captured["on_finished"] = on_finished
            return "task"

        monkeypatch.setattr(load_mod, "_run_importer_in_background", fake_run)
        resp = client.post("/api/dataset/load-demo", json={"name": next(iter(DEMO_DATASETS))})
        assert resp.status_code == 200, resp.get_json()
        assert captured["on_finished"] is import_hooks.fire_dataset_imported
