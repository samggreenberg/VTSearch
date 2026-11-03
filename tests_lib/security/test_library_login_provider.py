"""The login-provider abstraction is library-tier (issue #3042).

:mod:`vtscore.security.path_validation` decides file-access confinement by
asking :func:`vtscore.security.login.get_login_provider` whether a per-user
boundary exists.  That used to reach into ``vtsearch.auth``, so the whole
containment check raised ``ImportError`` in a process without the app
package - and a library-only embedder had no way to *opt into* confinement
at all.

These tests pin both halves of the fix: the default stays "single user, no
confinement" (what the app already did), and a plain
:class:`~vtscore.security.login.LoginProvider` subclass registered from
library code turns confinement on for every path check in the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vtscore.security.login import (
    DefaultLoginProvider,
    LoginProvider,
    get_login_provider,
    get_user_data_dir,
    is_safe_username,
    set_login_provider,
)
from vtscore.security.path_validation import (
    get_file_access_base_dir,
    media_file_read_roots,
    validate_server_filepath,
)
from vtscore.state.current_user import get_current_user, thread_user


class _EmbedderProvider(LoginProvider):
    """What a Flask-free multi-user embedding would register."""

    name = "embedder"

    def get_user(self, request: Any) -> str:  # noqa: ARG002
        return get_current_user()

    def is_authenticated(self, request: Any) -> bool:  # noqa: ARG002
        return True

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username


@pytest.fixture
def restore_provider():
    """Leave the process-wide provider as we found it."""
    original = get_login_provider()
    yield
    set_login_provider(original)


@pytest.fixture
def as_alice(restore_provider):
    """Register a confining provider and run as user ``alice``.

    Both halves are needed, and they are separate seams: the provider says
    *where* a user's data lives, while :mod:`vtscore.state.current_user`
    says *which* user this work is for.  ``get_user_data_dir()`` with no
    argument reads the latter, so a library embedder scopes work to a user
    with ``thread_user`` (or its own registered request-user resolver) —
    not by baking the name into the provider.
    """
    set_login_provider(_EmbedderProvider())
    with thread_user("alice"):
        yield


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """Point ``DATA_DIR`` at *tmp_path* everywhere it is read.

    ``get_user_data_dir`` imports it lazily (so patching the config module
    is enough), while ``path_validation`` bound it at import time.
    """
    monkeypatch.setattr("vtscore.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("vtscore.security.path_validation.DATA_DIR", tmp_path)
    return tmp_path


class TestDefaultIsUnconfinedSingleUser:
    def test_default_provider_is_active_out_of_the_box(self):
        assert isinstance(get_login_provider(), DefaultLoginProvider)

    def test_no_confinement_by_default(self):
        assert get_file_access_base_dir() is None
        assert media_file_read_roots() is None

    def test_user_data_dir_is_the_shared_data_dir(self):
        from vtscore.config import DATA_DIR

        assert get_user_data_dir("anyone") == DATA_DIR


class TestLibraryOnlyEmbedderCanOptIn:
    """The registered provider must drive confinement without any app tier."""

    def test_base_dir_follows_the_registered_provider(self, as_alice, data_dir):
        assert get_file_access_base_dir() == data_dir / "alice"
        assert media_file_read_roots() == [data_dir / "alice", data_dir]

    def test_escape_from_the_registered_base_dir_raises(self, as_alice, data_dir):
        base = get_file_access_base_dir()
        assert base is not None
        base.mkdir(parents=True)

        with pytest.raises(ValueError, match="outside the allowed directory"):
            validate_server_filepath("../bob/secret.txt", base_dir=base)

        inside = validate_server_filepath("notes.txt", base_dir=base)
        assert inside == (base / "notes.txt").resolve()

    def test_current_user_resolves_through_the_library_seam(self, as_alice, data_dir):
        """The two seams compose: a nested scope re-points the base dir."""
        assert get_user_data_dir() == data_dir / "alice"
        with thread_user("bob"):
            assert get_user_data_dir() == data_dir / "bob"
            assert get_file_access_base_dir() == data_dir / "bob"

    def test_enforce_auth_defaults_to_fail_closed(self, as_alice):
        """A provider that forgets to override must gate, not serve anonymously."""
        assert get_login_provider().enforce_auth() is True
        assert get_login_provider().www_authenticate() is None


class TestSafeUsername:
    @pytest.mark.parametrize("name", ["alice", "ci-bot", "a.b_c-1"])
    def test_accepts_path_safe_names(self, name):
        assert is_safe_username(name)

    @pytest.mark.parametrize("name", ["", ".", "..", "...", "a/b", "a\\b", "a b", None, 7])
    def test_rejects_traversal_and_non_strings(self, name):
        assert not is_safe_username(name)
