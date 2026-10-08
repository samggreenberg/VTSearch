"""The CLI resolves the AutoFind list + exporter from the running user.

``--autodetect --user X`` authenticates X and sets the thread-local user; the
pipeline then builds its config via ``CoreConfig.from_settings()``, which must
reflect that user's per-user AutoFind settings (detector list + exporter).
Without ``--user`` the built-in "default" user applies, reading through to the
server ``--settings`` file.
"""

from __future__ import annotations

from vtsearch import settings
from vtsearch.auth import thread_user
from vtscore.config import CoreConfig


class TestCliPerUserAutofind:
    def test_named_user_autofind_isolated(self, isolated_settings):
        with thread_user("alice"):
            settings.set_autofind_detectors(["alice-det"])
            settings.set_autofind_exporter("server_json_file")
        with thread_user("bob"):
            cfg = CoreConfig.from_settings()
            assert list(cfg.autofind_detectors) == []
            assert cfg.autofind_exporter == ""
        with thread_user("alice"):
            cfg = CoreConfig.from_settings()
            assert list(cfg.autofind_detectors) == ["alice-det"]
            assert cfg.autofind_exporter == "server_json_file"

    def test_default_user_reads_through_to_settings_file(self, isolated_settings):
        """The default user (CLI without --user) sees the --settings file's list."""
        import json

        # Simulate a CLI ``--settings`` file that carries autofind_detectors at
        # the (server) file the default user reads through to.
        isolated_settings._server.write_text(json.dumps({"autofind_detectors": ["from-settings-file"]}))
        settings.reset()
        # No thread user -> "default".
        cfg = CoreConfig.from_settings()
        assert list(cfg.autofind_detectors) == ["from-settings-file"]

    def test_named_user_cli_delete_setting_isolated(self, isolated_settings):
        """``autofind_cli_delete_dataset`` (#4674) is each user's own choice."""
        with thread_user("alice"):
            settings.set_autofind_cli_delete_dataset(True)
        with thread_user("bob"):
            assert CoreConfig.from_settings().autofind_cli_delete_dataset is False
        with thread_user("alice"):
            assert CoreConfig.from_settings().autofind_cli_delete_dataset is True
        # Off for everyone who never turned it on, the default user included.
        assert CoreConfig.from_settings().autofind_cli_delete_dataset is False

    def test_default_user_reads_cli_delete_setting_from_settings_file(self, isolated_settings):
        """A ``--settings`` flat file can turn it on for a CLI run without ``--user``."""
        import json

        isolated_settings._server.write_text(json.dumps({"autofind_cli_delete_dataset": True}))
        settings.reset()
        assert CoreConfig.from_settings().autofind_cli_delete_dataset is True
        assert settings.get_all()["autofind_cli_delete_dataset"] is True
        # The read-through is the default user's alone: a named user keeps the default.
        with thread_user("alice"):
            assert CoreConfig.from_settings().autofind_cli_delete_dataset is False
            assert settings.get_all()["autofind_cli_delete_dataset"] is False
