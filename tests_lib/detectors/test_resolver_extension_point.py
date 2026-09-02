"""The public resolver extension point must actually take effect.

``register_source_resolver`` / ``register_importer_resolver`` are documented
in ``vtscore/docs/packages/detectors.md`` for out-of-tree consumers, so this
repo has no registrants of its own.  That is what a working extension point
looks like — and it is also how the hooks could rot unnoticed.  These tests
stand in for the absent in-repo caller: they register a fake resolver through
the public function and assert it displaces the default that
``vtscore.detectors.resolver`` installs at import time.
"""

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from vtscore.detectors import resolver as resolver_mod


@pytest.fixture
def restore_resolvers(monkeypatch):
    """Undo whatever the test registers, without going through a public API.

    There is deliberately no ``unregister_*`` function, so the defaults are
    put back by re-binding the module globals to the values they currently
    hold; ``monkeypatch`` restores them at teardown.
    """
    monkeypatch.setattr(resolver_mod, "_source_resolver", resolver_mod._source_resolver)
    monkeypatch.setattr(resolver_mod, "_importer_resolver", resolver_mod._importer_resolver)


class TestResolverRegistration:
    def test_defaults_are_installed_at_import(self):
        """No first-use auto-wiring: importing the module is enough."""
        assert resolver_mod._source_resolver is resolver_mod._default_source_resolver
        assert resolver_mod._importer_resolver is resolver_mod._default_importer_resolver

    def test_registered_source_resolver_wins_over_default(self, tmp_path, restore_resolvers):
        """A registered source resolver is consulted instead of the default.

        The origin names ``server_folder`` with a path the default source
        resolver would happily resolve, so a passing assertion means the fake
        really displaced it rather than merely running first.
        """
        folder = tmp_path / "media"
        folder.mkdir()
        real = folder / "thing.txt"
        real.write_bytes(b"real")
        fake_file = tmp_path / "fake.txt"
        fake_file.write_bytes(b"fake")

        calls: list[tuple[str, str]] = []

        def _fake_source_resolver(
            stack: ExitStack,
            origin: dict[str, Any],
            origin_name: str,
            filename: str,
        ) -> Path | None:
            calls.append((origin_name, filename))
            return fake_file

        resolver_mod.register_source_resolver(_fake_source_resolver)

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        with resolver_mod.resolve_file_context(origin, origin_name="thing.txt") as path:
            assert path is not None
            assert path == fake_file, "registered resolver must win over the default"
            assert path.read_bytes() == b"fake"

        assert calls == [("thing.txt", "")]

    def test_registered_source_resolver_may_defer_to_importer_dispatch(self, tmp_path, restore_resolvers):
        """Returning ``None`` falls through to the importer resolver."""
        folder = tmp_path / "media"
        folder.mkdir()
        (folder / "thing.txt").write_bytes(b"real")

        importer_calls: list[str] = []

        def _decline(
            stack: ExitStack,
            origin: dict[str, Any],
            origin_name: str,
            filename: str,
        ) -> Path | None:
            return None

        def _fake_importer_resolver(
            origin: dict[str, Any],
            origin_name: str,
            filename: str,
        ) -> Path | None:
            importer_calls.append(origin_name)
            return folder / origin_name

        resolver_mod.register_source_resolver(_decline)
        resolver_mod.register_importer_resolver(_fake_importer_resolver)

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        with resolver_mod.resolve_file_context(origin, origin_name="thing.txt") as path:
            assert path == folder / "thing.txt"

        assert importer_calls == ["thing.txt"], "importer resolver must be reached"

    def test_registered_importer_resolver_wins_over_default(self, tmp_path, restore_resolvers):
        """An importer-only origin routes through the registered hook."""
        target = tmp_path / "from_importer.txt"
        target.write_bytes(b"payload")

        calls: list[str] = []

        def _fake_importer_resolver(
            origin: dict[str, Any],
            origin_name: str,
            filename: str,
        ) -> Path | None:
            calls.append(origin.get("importer", ""))
            return target

        resolver_mod.register_importer_resolver(_fake_importer_resolver)

        # No media source backs this importer name, so source dispatch
        # declines and the importer resolver is what answers.
        origin = {"importer": "not_a_real_importer", "params": {}}
        with resolver_mod.resolve_file_context(origin, origin_name="x.txt") as path:
            assert path == target

        assert calls == ["not_a_real_importer"]
