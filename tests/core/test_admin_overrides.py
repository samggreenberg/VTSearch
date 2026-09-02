"""Tests for the declarative admin-override registry.

The registry (:mod:`vtsearch.admin_overrides`) is what stops the six
process-level admin knobs from drifting apart the way they had: before it,
each was hand-plumbed through argparse, the environment, the settings
resolvers and the ``/api/settings`` overlay separately, and only three of the
six ever got an env var -- so which restrictions a Docker deployment could set
was arbitrary.

The first class below is the guard against that recurring: it asserts
*structurally* that every registered override is reachable from a flag, from
the environment, and from the settings payload. A seventh knob that forgets one
of the three fails here rather than being discovered by an operator.
"""

from __future__ import annotations

import pytest

from vtsearch import admin_overrides
from vtsearch import cli_main
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def _isolated_overrides():
    """Snapshot and restore the process-level override store around each test."""
    saved = admin_overrides.snapshot()
    admin_overrides.reset_overrides()
    yield
    admin_overrides.restore(saved)


class TestRegistryCoverage:
    """Every override must be reachable from all four consumers."""

    def test_every_override_declares_a_flag_and_an_env_var(self):
        for name, override in admin_overrides.OVERRIDES.items():
            assert override.flag.startswith("--"), name
            assert override.env.startswith("VTSEARCH_"), name
            assert override.effective_key, name
            assert override.help, name

    def test_flags_and_env_names_are_unique(self):
        flags = [ov.flag for ov in admin_overrides.OVERRIDES.values()]
        envs = [ov.env for ov in admin_overrides.OVERRIDES.values()]
        assert len(set(flags)) == len(flags)
        assert len(set(envs)) == len(envs)

    def test_persisted_getter_exists_on_settings(self):
        for name, override in admin_overrides.OVERRIDES.items():
            getter = getattr(settings_mod, override.persisted_getter, None)
            assert callable(getter), f"{name}: no settings.{override.persisted_getter}"

    def test_every_override_is_registered_on_the_parser(self):
        parser = cli_main._build_parser()
        registered = {opt for action in parser._actions for opt in action.option_strings}
        for name, override in admin_overrides.OVERRIDES.items():
            assert override.flag in registered, name

    def test_value_and_append_overrides_carry_a_parser(self):
        """Only ``switch`` knobs may omit ``parse``; ``append`` needs ``fold`` too."""
        for name, override in admin_overrides.OVERRIDES.items():
            if override.kind == "switch":
                continue
            assert override.parse is not None, name
            if override.kind == "append":
                assert override.fold is not None, name

    def test_settings_resolves_every_override(self):
        """``get_effective_override`` works for every name, with nothing set."""
        for name in admin_overrides.OVERRIDES:
            settings_mod.get_effective_override(name)


class TestEnvOverrides:
    """The env form of each knob -- the half that used to be missing."""

    def test_solo_media_type_from_env(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SOLO_MEDIA_TYPE", "image")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_effective_solo_media_type() == "image"
        assert admin_overrides.override_source("solo_media_type") == "VTSEARCH_SOLO_MEDIA_TYPE"

    def test_hide_plugins_from_env_is_comma_separated(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_HIDE_PLUGINS", "embedders:clap, importers:synthetic")
        admin_overrides.apply_env_overrides()
        hidden = settings_mod.get_effective_hidden_plugins()
        assert hidden["embedders"] == {"clap"}
        assert hidden["importers"] == {"synthetic"}

    def test_solo_embedders_from_env_is_comma_separated(self, monkeypatch):
        from vtscore.media import all_type_ids, embedders_for_type

        pair = next(
            ((mt, embs[0].name) for mt in all_type_ids() if (embs := embedders_for_type(mt))),
            None,
        )
        assert pair is not None, "no embedders registered"
        mt, emb = pair
        monkeypatch.setenv("VTSEARCH_SOLO_EMBEDDERS", f"{mt}={emb}")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_cli_solo_embedders()[mt] == emb

    def test_dataset_max_age_from_env(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_DATASET_MAX_AGE_DAYS", "30")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_effective_dataset_max_age_days() == 30

    def test_support_email_from_env(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SUPPORT_EMAIL", "ops@example.org")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_effective_support_email() == "ops@example.org"

    def test_semantic_only_from_env_accepts_truthy_words(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SEMANTIC_ONLY", "yes")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_effective_semantic_only() is True

    def test_semantic_only_env_zero_does_not_loosen(self, monkeypatch):
        """The switch can only enable; ``0`` leaves the persisted setting in charge."""
        monkeypatch.setenv("VTSEARCH_SEMANTIC_ONLY", "0")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_cli_semantic_only() is None

    def test_an_explicit_flag_wins_over_the_env(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SUPPORT_EMAIL", "env@example.org")
        admin_overrides.set_override("support_email", "flag@example.org", source="--support-email")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_effective_support_email() == "flag@example.org"

    def test_a_blank_env_var_sets_nothing(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SUPPORT_EMAIL", "   ")
        admin_overrides.apply_env_overrides()
        assert settings_mod.get_cli_support_email() is None


class TestSharedValidation:
    """A value the flag rejects is rejected the same way from the environment."""

    def test_env_and_flag_run_the_same_validator(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setenv("VTSEARCH_DATASET_MAX_AGE_DAYS", "0")
        admin_overrides.apply_env_overrides(warn=warnings.append)
        assert settings_mod.get_cli_dataset_max_age_days() is None
        assert warnings and "must be a positive integer" in warnings[0]

    def test_a_bad_env_value_names_the_variable_not_the_flag(self, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setenv("VTSEARCH_SOLO_MEDIA_TYPE", "not_a_type")
        admin_overrides.apply_env_overrides(warn=warnings.append)
        assert warnings
        assert "VTSEARCH_SOLO_MEDIA_TYPE" in warnings[0]
        assert "--solo-media-type" not in warnings[0]

    def test_a_bad_env_value_leaves_the_server_bootable(self, monkeypatch):
        """Unlike the CLI, the env path warns rather than exiting."""
        monkeypatch.setenv("VTSEARCH_HIDE_PLUGINS", "no-colon-here")
        admin_overrides.apply_env_overrides(warn=lambda _msg: None)
        assert settings_mod.get_cli_hidden_plugins() == {}

    def test_non_numeric_dataset_max_age_flag_is_our_error(self):
        """``type=str`` on the flag means the descriptor writes the message."""
        with pytest.raises(admin_overrides.OverrideValueError, match="positive integer"):
            admin_overrides.OVERRIDES["dataset_max_age_days"].parse_token("abc", source="--dataset-max-age-days")


class TestStore:
    """The process-level store itself."""

    def test_get_override_returns_a_defensive_copy(self):
        settings_mod.add_cli_hidden_plugin("embedders", "clap")
        got = admin_overrides.get_override("hidden_plugins")
        got.setdefault("embedders", set()).add("mutated")
        assert admin_overrides.get_override("hidden_plugins")["embedders"] == {"clap"}

    def test_clearing_an_override_clears_its_source(self):
        admin_overrides.set_override("support_email", "a@b.c", source="--support-email")
        settings_mod.set_cli_support_email(None)
        assert admin_overrides.override_source("support_email") is None

    def test_reset_clears_everything(self):
        settings_mod.set_cli_support_email("a@b.c")
        settings_mod.add_cli_hidden_plugin("embedders", "clap")
        admin_overrides.reset_overrides()
        assert settings_mod.get_cli_support_email() is None
        assert settings_mod.get_cli_hidden_plugins() == {}


class TestSettingsPayload:
    """``/api/settings`` publishes the effective value of every override."""

    def test_every_effective_key_is_present(self, client):
        body = client.get("/api/settings").get_json()
        for name, override in admin_overrides.OVERRIDES.items():
            assert override.effective_key in body, f"{name} missing from /api/settings"
