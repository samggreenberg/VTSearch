"""Tests for externally-supplied origin confinement validation.

:func:`vtscore.security.origin_validation.confine_origin_params`
guards flows that accept a full origin dict from outside the server (the
example-sort-origin route, a detector's saved media examples) before the
origin's path-like params are used for filesystem access (issue #2774).
"""

from __future__ import annotations

import pytest

from vtscore.security.origin_validation import confine_origin_params


@pytest.fixture
def confined_base(monkeypatch, tmp_path):
    """Simulate multi-user mode: confine file access to a per-user dir."""
    import vtscore.security.path_validation as paths_mod

    base = tmp_path / "userdata"
    base.mkdir()
    monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: base)
    return base


class TestSingleUserMode:
    def test_any_path_allowed_when_unconfined(self, monkeypatch):
        import vtscore.security.path_validation as paths_mod

        monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: None)
        confine_origin_params({"importer": "server_file", "params": {"path": "/etc/passwd"}})


class TestMultiUserConfinement:
    def test_confined_path_passes(self, confined_base):
        inside = confined_base / "clip.wav"
        confine_origin_params({"importer": "server_file", "params": {"path": str(inside)}})

    def test_escaping_path_raises(self, confined_base):
        with pytest.raises(ValueError, match="outside the allowed directory"):
            confine_origin_params({"importer": "server_file", "params": {"path": "/etc/passwd"}})

    def test_url_params_are_exempt(self, confined_base):
        # URLs are re-validated by validate_url at fetch time; the path
        # check must not spuriously reject them in multi-user mode.
        confine_origin_params({"importer": "url_download", "params": {"url": "https://media.example.test/a.wav"}})

    def test_pathless_params_ignored(self, confined_base):
        confine_origin_params({"importer": "example_media", "params": {"filename": "abc.wav"}})

    def test_dupe_set_members_are_recursed(self, confined_base):
        origin = {
            "importer": "dupe_set",
            "members": [
                {"origin": {"importer": "server_file", "params": {"path": "/etc/passwd"}}},
            ],
        }
        with pytest.raises(ValueError, match="outside the allowed directory"):
            confine_origin_params(origin)

    def test_non_dict_origin_is_a_noop(self, confined_base):
        assert confine_origin_params(None) is None
        assert confine_origin_params("not-a-dict") == "not-a-dict"


class TestConfinedCopy:
    """Issue #2917: the check anchors a relative param at the user's data dir
    while the media source would anchor it at the CWD, so the caller must
    resolve the *returned* origin, not the one it passed in."""

    def test_relative_param_comes_back_as_the_approved_path(self, confined_base):
        out = confine_origin_params({"importer": "server_folder", "params": {"path": "data/alice"}})
        assert out["params"]["path"] == str((confined_base / "data" / "alice").resolve())

    def test_input_is_not_mutated(self, confined_base):
        origin = {"importer": "server_folder", "params": {"path": "data/alice"}}
        confine_origin_params(origin)
        assert origin["params"]["path"] == "data/alice"

    def test_members_are_confined_too(self, confined_base):
        origin = {
            "importer": "dupe_set",
            "members": [{"origin": {"importer": "server_file", "params": {"path": "sounds/clip.wav"}}}],
        }
        out = confine_origin_params(origin)
        assert out["members"][0]["origin"]["params"]["path"] == str((confined_base / "sounds" / "clip.wav").resolve())

    def test_url_params_pass_through_unchanged(self, confined_base):
        url = "https://media.example.test/a.wav"
        out = confine_origin_params({"importer": "url_download", "params": {"url": url}})
        assert out["params"]["url"] == url

    def test_unconfined_returns_params_verbatim(self, monkeypatch):
        import vtscore.security.path_validation as paths_mod

        monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: None)
        out = confine_origin_params({"importer": "server_folder", "params": {"path": "data/sounds"}})
        assert out["params"]["path"] == "data/sounds"
