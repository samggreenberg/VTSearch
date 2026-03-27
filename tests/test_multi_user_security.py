"""Tests for multi-user security: LoginProvider abstraction and user isolation.

Verifies that:
- The LoginProvider ABC and DefaultLoginProvider work correctly
- ``g.user`` is set on every request via the before_request middleware
- Dataset, model, and detector entries include ``created_by``
- The ``/api/auth/status`` endpoint returns correct information
- get_current_user() falls back to "default" outside Flask context
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtsearch.auth import (
    DefaultLoginProvider,
    LoginProvider,
    TrivialLoginProvider,
    get_current_user,
    get_login_provider,
    get_user_data_dir,
    set_login_provider,
)


# ---------------------------------------------------------------------------
# LoginProvider abstraction
# ---------------------------------------------------------------------------


class TestLoginProviderABC:
    """Test the LoginProvider base class and DefaultLoginProvider."""

    def test_default_provider_returns_default_user(self):
        provider = DefaultLoginProvider()
        assert provider.get_user(None) == "default"

    def test_default_provider_always_authenticated(self):
        provider = DefaultLoginProvider()
        assert provider.is_authenticated(None) is True

    def test_default_provider_no_login_required(self):
        provider = DefaultLoginProvider()
        assert provider.login_required() is False

    def test_default_provider_name(self):
        provider = DefaultLoginProvider()
        assert provider.name == "default"

    def test_default_provider_data_dir_unchanged(self):
        """DefaultLoginProvider uses DATA_DIR directly (no subdirectory)."""
        provider = DefaultLoginProvider()
        base = Path("/some/data")
        assert provider.get_user_data_dir("default", base) == base
        # Even with a different username, default provider returns base unchanged
        assert provider.get_user_data_dir("alice", base) == base

    def test_default_provider_status_dict(self):
        provider = DefaultLoginProvider()
        status = provider.status_dict(None)
        assert status["provider"] == "default"
        assert status["user"] == "default"
        assert status["authenticated"] is True
        assert status["login_required"] is False

    def test_cannot_instantiate_abc_directly(self):
        """LoginProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LoginProvider()  # type: ignore[abstract]


class TestCustomLoginProvider:
    """Test that a custom LoginProvider implementation works correctly."""

    def test_custom_provider(self):
        class TestProvider(LoginProvider):
            name = "test"

            def get_user(self, request):
                return "alice"

            def is_authenticated(self, request):
                return True

            def get_user_data_dir(self, username, base_data_dir):
                return base_data_dir / username

        provider = TestProvider()
        assert provider.get_user(None) == "alice"
        assert provider.is_authenticated(None) is True
        assert provider.login_required() is False  # default
        data_dir = provider.get_user_data_dir("alice", Path("/data"))
        assert data_dir == Path("/data/alice")

    def test_custom_provider_with_login_required(self):
        class SecureProvider(LoginProvider):
            name = "secure"

            def get_user(self, request):
                return "bob"

            def is_authenticated(self, request):
                return False

            def login_required(self):
                return True

        provider = SecureProvider()
        assert provider.login_required() is True
        assert provider.is_authenticated(None) is False
        status = provider.status_dict(None)
        assert status["login_required"] is True
        assert status["authenticated"] is False


# ---------------------------------------------------------------------------
# Module-level provider management
# ---------------------------------------------------------------------------


class TestProviderManagement:
    """Test set_login_provider / get_login_provider / get_current_user."""

    def test_default_provider_is_set_initially(self):
        provider = get_login_provider()
        assert isinstance(provider, DefaultLoginProvider)

    def test_set_and_get_provider(self):
        class Custom(LoginProvider):
            name = "custom"

            def get_user(self, request):
                return "custom_user"

            def is_authenticated(self, request):
                return True

        original = get_login_provider()
        try:
            set_login_provider(Custom())
            assert get_login_provider().name == "custom"
        finally:
            set_login_provider(original)

    def test_get_current_user_outside_flask_context(self):
        """Outside a Flask request context, get_current_user returns 'default'."""
        # The reset_state fixture already resets to DefaultLoginProvider
        user = get_current_user()
        assert user == "default"

    def test_get_user_data_dir_default(self):
        """get_user_data_dir returns DATA_DIR for the default provider."""
        from vtsearch.config import DATA_DIR

        data_dir = get_user_data_dir("default")
        assert data_dir == DATA_DIR


# ---------------------------------------------------------------------------
# Flask integration
# ---------------------------------------------------------------------------


class TestFlaskAuthMiddleware:
    """Test that g.user is set on Flask requests."""

    def test_auth_status_endpoint(self, client):
        """GET /api/auth/status returns provider info."""
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["provider"] == "default"
        assert data["user"] == "default"
        assert data["authenticated"] is True
        assert data["login_required"] is False

    def test_g_user_set_on_api_requests(self, client):
        """Verify g.user is set via the before_request middleware."""
        # Any API call should work without errors — the middleware sets g.user
        resp = client.get("/api/medias")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Ownership metadata
# ---------------------------------------------------------------------------


class TestOwnershipMetadata:
    """Test that created_by is recorded on detectors, datasets, and models."""

    def test_autorun_detector_has_created_by(self):
        from vtsearch.utils import add_autorun_detector, get_autorun_detectors

        add_autorun_detector("test_det", "audio", created_by="alice")
        dets = get_autorun_detectors()
        assert dets["test_det"]["created_by"] == "alice"

    def test_autorun_detector_default_created_by(self):
        from vtsearch.utils import add_autorun_detector, get_autorun_detectors

        add_autorun_detector("test_det2", "audio")
        dets = get_autorun_detectors()
        assert dets["test_det2"]["created_by"] == "default"

    def test_dataset_registry_has_created_by(self):
        from vtsearch.datasets.registry import register_dataset

        entry = register_dataset(
            name="test_ds",
            media_type="audio",
            num_items=10,
            pkl_path="/tmp/test.pkl",
            created_by="bob",
        )
        assert entry["created_by"] == "bob"

    def test_dataset_registry_default_created_by(self):
        from vtsearch.datasets.registry import register_dataset

        entry = register_dataset(
            name="test_ds2",
            media_type="audio",
            num_items=5,
            pkl_path="/tmp/test2.pkl",
        )
        assert entry["created_by"] == "default"

    def test_model_registry_has_created_by(self):
        from vtsearch.models.registry import register_model

        entry = register_model(
            name="test_model",
            media_type="audio",
            trainable=True,
            created_by="carol",
        )
        assert entry["created_by"] == "carol"

    def test_model_registry_default_created_by(self):
        from vtsearch.models.registry import register_model

        entry = register_model(
            name="test_model2",
            media_type="audio",
            trainable=False,
        )
        assert entry["created_by"] == "default"


# ---------------------------------------------------------------------------
# User data directory isolation
# ---------------------------------------------------------------------------


class TestUserDataDirIsolation:
    """Test per-user data directory logic for different providers."""

    def test_default_provider_no_subdirectory(self):
        """DefaultLoginProvider maps all users to DATA_DIR directly."""
        provider = DefaultLoginProvider()
        base = Path("/app/data")
        assert provider.get_user_data_dir("default", base) == base
        assert provider.get_user_data_dir("alice", base) == base

    def test_multi_user_provider_uses_subdirectory(self):
        """A multi-user provider scopes each user to their own subdirectory."""

        class MultiUserProvider(LoginProvider):
            name = "multiuser"

            def get_user(self, request):
                return "alice"

            def is_authenticated(self, request):
                return True

            def get_user_data_dir(self, username, base_data_dir):
                return base_data_dir / username

        provider = MultiUserProvider()
        base = Path("/app/data")
        assert provider.get_user_data_dir("alice", base) == Path("/app/data/alice")
        assert provider.get_user_data_dir("bob", base) == Path("/app/data/bob")

    def test_get_user_data_dir_uses_active_provider(self):
        """get_user_data_dir() delegates to the active provider."""
        from vtsearch.config import DATA_DIR

        # Default provider should return DATA_DIR unchanged
        result = get_user_data_dir("anything")
        assert result == DATA_DIR


# ---------------------------------------------------------------------------
# Provider swap safety
# ---------------------------------------------------------------------------


class TestProviderSwapSafety:
    """Test that swapping providers doesn't break existing functionality."""

    def test_swap_provider_and_back(self, client):
        """Swapping to a custom provider and back doesn't break the app."""

        class TempProvider(LoginProvider):
            name = "temp"

            def get_user(self, request):
                return "temp_user"

            def is_authenticated(self, request):
                return True

        set_login_provider(TempProvider())
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["user"] == "temp_user"
        assert data["provider"] == "temp"

        # Swap back
        set_login_provider(DefaultLoginProvider())
        resp = client.get("/api/auth/status")
        data = resp.get_json()
        assert data["user"] == "default"

    def test_api_works_after_provider_swap(self, client):
        """Core API endpoints still work after a provider swap."""

        class AnotherProvider(LoginProvider):
            name = "another"

            def get_user(self, request):
                return "test_user"

            def is_authenticated(self, request):
                return True

        set_login_provider(AnotherProvider())

        # These should all return 200 regardless of provider
        resp = client.get("/api/medias")
        assert resp.status_code == 200

        resp = client.get("/api/auth/status")
        assert resp.status_code == 200

        # Restore default
        set_login_provider(DefaultLoginProvider())


# ---------------------------------------------------------------------------
# TrivialLoginProvider
# ---------------------------------------------------------------------------


class TestTrivialLoginProvider:
    """Test the cookie-based trivial login provider."""

    def test_provider_properties(self):
        provider = TrivialLoginProvider()
        assert provider.name == "trivial"
        assert provider.login_required() is True

    def test_unauthenticated_outside_flask(self):
        provider = TrivialLoginProvider()
        assert provider.get_user(None) == "anonymous"
        assert provider.is_authenticated(None) is False

    def test_data_dir_uses_subdirectory(self):
        provider = TrivialLoginProvider()
        base = Path("/data")
        assert provider.get_user_data_dir("alice", base) == Path("/data/alice")

    def test_status_dict_unauthenticated(self):
        provider = TrivialLoginProvider()
        status = provider.status_dict(None)
        assert status["provider"] == "trivial"
        assert status["user"] == "anonymous"
        assert status["authenticated"] is False
        assert status["login_required"] is True


class TestTrivialLoginEndpoints:
    """Test the /api/auth/login and /api/auth/logout endpoints."""

    def test_login_rejected_with_default_provider(self, client):
        """Login endpoint returns 400 when the trivial provider is not active."""
        resp = client.post("/api/auth/login", json={"username": "alice"})
        assert resp.status_code == 400
        assert "not supported" in resp.get_json()["error"]

    def test_logout_rejected_with_default_provider(self, client):
        """Logout endpoint returns 400 when the trivial provider is not active."""
        resp = client.post("/api/auth/logout", json={})
        assert resp.status_code == 400

    def test_login_and_logout_flow(self, client):
        """Full login → status → logout → status cycle."""
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            # Initially unauthenticated
            resp = client.get("/api/auth/status")
            data = resp.get_json()
            assert data["authenticated"] is False
            assert data["user"] == "anonymous"
            assert data["login_required"] is True

            # Log in
            resp = client.post("/api/auth/login", json={"username": "alice"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["user"] == "alice"
            assert data["authenticated"] is True

            # Status reflects login
            resp = client.get("/api/auth/status")
            data = resp.get_json()
            assert data["user"] == "alice"
            assert data["authenticated"] is True

            # Log out
            resp = client.post("/api/auth/logout", json={})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["authenticated"] is False
            assert data["user"] == "anonymous"

        finally:
            set_login_provider(original)

    def test_login_empty_username_rejected(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            resp = client.post("/api/auth/login", json={"username": ""})
            assert resp.status_code == 400
        finally:
            set_login_provider(original)

    def test_login_invalid_username_rejected(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            resp = client.post("/api/auth/login", json={"username": "alice bob"})
            assert resp.status_code == 400
        finally:
            set_login_provider(original)

    def test_login_special_chars_rejected(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            resp = client.post("/api/auth/login", json={"username": "../etc"})
            assert resp.status_code == 400
        finally:
            set_login_provider(original)

    def test_login_valid_usernames_accepted(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            for name in ["alice", "Bob_123", "user-1", "A"]:
                resp = client.post("/api/auth/login", json={"username": name})
                assert resp.status_code == 200, f"Failed for {name!r}"
                assert resp.get_json()["user"] == name
        finally:
            set_login_provider(original)

    def test_g_user_reflects_trivial_login(self, client):
        """After trivial login, g.user (and thus get_current_user) returns the logged-in name."""
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            client.post("/api/auth/login", json={"username": "carol"})

            # Any subsequent request should see g.user = "carol"
            resp = client.get("/api/auth/status")
            assert resp.get_json()["user"] == "carol"
        finally:
            set_login_provider(original)
