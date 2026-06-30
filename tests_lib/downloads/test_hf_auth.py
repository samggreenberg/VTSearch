"""HuggingFace credential store + its effect on the download path.

Covers the in-memory token store (`vtscore.security.hf_auth`), the per-host
Bearer-header injection in `download_file_with_progress`, and the short
`GatedResourceError` raised on a 401/403.
"""

import pytest

from vtscore.security.hf_auth import (
    GatedResourceError,
    auth_header_for_url,
    clear_credential,
    get_status,
    get_token,
    is_authenticated,
    set_credential,
)


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response (see
    ``test_download_and_extract.py`` for the sibling used by the resume tests)."""

    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, chunks=(), status_code=200, headers=None):
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        yield from self._chunks

    def close(self):
        self.closed = True


def _install(monkeypatch, responses):
    """Patch ``requests.Session.get`` to hand out *responses* and record the
    ``(url, headers)`` of each call."""
    import requests

    from vtscore.datasets.downloader import core as dl_core

    calls = []

    def fake_get(self, url, *args, headers=None, **kwargs):
        calls.append({"url": url, "headers": headers})
        return responses.pop(0)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(dl_core.time, "sleep", lambda *a, **k: None)
    return calls


class TestCredentialStore:
    def test_unset_is_anonymous(self):
        assert get_token() is None
        assert is_authenticated() is False
        assert get_status() == {"authenticated": False, "username": "", "scopes": ""}

    def test_set_get_clear(self):
        set_credential("tok-abc", username="alice", scopes="read-repos")
        assert get_token() == "tok-abc"
        assert is_authenticated() is True
        assert get_status() == {"authenticated": True, "username": "alice", "scopes": "read-repos"}
        clear_credential()
        assert get_token() is None
        assert is_authenticated() is False

    def test_empty_token_rejected(self):
        with pytest.raises(ValueError):
            set_credential("   ")

    def test_expired_token_is_inactive(self):
        # expires_at in the past -> treated as signed out.
        set_credential("tok-old", username="bob", expires_at=1.0)
        assert get_token() is None
        assert is_authenticated() is False
        assert get_status()["authenticated"] is False


class TestAuthHeaderForUrl:
    def test_no_header_without_token(self):
        assert auth_header_for_url("https://huggingface.co/x") == {}

    def test_header_only_for_hf_hosts(self):
        set_credential("tok-xyz")
        assert auth_header_for_url("https://huggingface.co/datasets/x/resolve/main/a") == {
            "Authorization": "Bearer tok-xyz"
        }
        # CDN subdomain of the Hub is still the Hub.
        assert auth_header_for_url("https://cdn-lfs.huggingface.co/x") == {"Authorization": "Bearer tok-xyz"}
        # Third-party hosts (e.g. a presigned redirect target) must not see it.
        assert auth_header_for_url("https://example.com/x") == {}
        assert auth_header_for_url("https://nothuggingface.co.evil.com/x") == {}


class TestDownloadUsesToken:
    def test_sends_bearer_to_hf_host(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        set_credential("tok-dl")
        resp = _FakeResponse([b"data"], status_code=200, headers={"content-length": "4"})
        calls = _install(monkeypatch, [resp])

        dest = tmp_path / "f.bin"
        dl_core.download_file_with_progress(
            "https://huggingface.co/datasets/x/resolve/main/f.bin", dest, on_progress=lambda *a: None
        )

        assert dest.read_bytes() == b"data"
        assert calls[0]["headers"]["Authorization"] == "Bearer tok-dl"

    def test_no_bearer_to_third_party_host(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        set_credential("tok-dl")
        resp = _FakeResponse([b"data"], status_code=200, headers={"content-length": "4"})
        calls = _install(monkeypatch, [resp])

        dest = tmp_path / "f.bin"
        dl_core.download_file_with_progress("https://example.com/f.bin", dest, on_progress=lambda *a: None)

        assert "Authorization" not in (calls[0]["headers"] or {})


class TestGatedError:
    @pytest.mark.parametrize("status", [401, 403])
    def test_gated_status_raises_gated_error(self, tmp_path, monkeypatch, status):
        from vtscore.datasets.downloader import core as dl_core

        # A single response is enough: a gated error must NOT retry.
        resp = _FakeResponse([], status_code=status)
        calls = _install(monkeypatch, [resp])

        dest = tmp_path / "f.bin"
        with pytest.raises(GatedResourceError) as exc:
            dl_core.download_file_with_progress(
                "https://huggingface.co/datasets/x/resolve/main/f.bin", dest, on_progress=lambda *a: None
            )
        assert exc.value.status == status
        assert len(calls) == 1  # no retry
        assert "huggingface" in str(exc.value).lower()

    def test_signed_in_message_mentions_access(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        set_credential("tok-dl", username="alice")
        resp = _FakeResponse([], status_code=403)
        _install(monkeypatch, [resp])

        dest = tmp_path / "f.bin"
        with pytest.raises(GatedResourceError) as exc:
            dl_core.download_file_with_progress(
                "https://huggingface.co/datasets/x/resolve/main/f.bin", dest, on_progress=lambda *a: None
            )
        assert "access" in str(exc.value).lower()


class TestHfTokenResolution:
    def test_store_token_preferred_over_env(self, monkeypatch):
        from vtscore.media.embedder import hf_token

        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert hf_token() == "from-env"
        set_credential("from-oauth")
        assert hf_token() == "from-oauth"

    def test_env_used_when_not_signed_in(self, monkeypatch):
        from vtscore.media.embedder import hf_token

        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert hf_token() == "from-env"

    def test_false_when_neither(self, monkeypatch):
        from vtscore.media.embedder import hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert hf_token() is False
