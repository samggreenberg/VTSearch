"""Tests for externally-supplied origin confinement validation.

:func:`vtscore.security.origin_validation.check_origin_param_confinement`
guards flows that accept a full origin dict from outside the server (the
example-sort-origin route, a detector's saved media examples) before the
origin's path-like params are used for filesystem access (issue #2774).
"""

from __future__ import annotations

import pytest

from vtscore.security.origin_validation import check_origin_param_confinement


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
        check_origin_param_confinement({"importer": "server_file", "params": {"path": "/etc/passwd"}})


class TestMultiUserConfinement:
    def test_confined_path_passes(self, confined_base):
        inside = confined_base / "clip.wav"
        check_origin_param_confinement({"importer": "server_file", "params": {"path": str(inside)}})

    def test_escaping_path_raises(self, confined_base):
        with pytest.raises(ValueError, match="outside the allowed directory"):
            check_origin_param_confinement({"importer": "server_file", "params": {"path": "/etc/passwd"}})

    def test_url_params_are_exempt(self, confined_base):
        # URLs are re-validated by validate_url at fetch time; the path
        # check must not spuriously reject them in multi-user mode.
        check_origin_param_confinement(
            {"importer": "url_download", "params": {"url": "https://media.example.test/a.wav"}}
        )

    def test_pathless_params_ignored(self, confined_base):
        check_origin_param_confinement({"importer": "example_media", "params": {"filename": "abc.wav"}})

    def test_dupe_set_members_are_recursed(self, confined_base):
        origin = {
            "importer": "dupe_set",
            "members": [
                {"origin": {"importer": "server_file", "params": {"path": "/etc/passwd"}}},
            ],
        }
        with pytest.raises(ValueError, match="outside the allowed directory"):
            check_origin_param_confinement(origin)

    def test_non_dict_origin_is_a_noop(self, confined_base):
        check_origin_param_confinement(None)
        check_origin_param_confinement("not-a-dict")
