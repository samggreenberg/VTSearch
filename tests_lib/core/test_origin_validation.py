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
        check_origin_param_confinement({"importer": "server_folder", "params": {"path": ".."}})


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

    def test_plain_relative_params_pass(self, confined_base):
        # Opaque params (a cache filename, a media type, an archive member
        # key) resolve inside the user's own dir, so validating them is
        # harmless — the check errs towards validating too much.
        check_origin_param_confinement({"importer": "example_media", "params": {"filename": "abc.wav"}})
        check_origin_param_confinement(
            {
                "importer": "local_archive_member",
                "params": {"member": "sounds/clip.wav", "media_type": "audio"},
            }
        )

    @pytest.mark.parametrize("token", ["..", ".", "~", "~/secrets", "../..", "..\\..", "./data", "a/../.."])
    def test_separator_free_and_dot_tokens_are_rejected(self, confined_base, token):
        """Issue #2918: ``..`` / ``.`` / ``~`` carry no path separator, so the
        old heuristic skipped them entirely and ``LocalFolderSource("..")``
        got built against the process CWD's parent."""
        with pytest.raises(ValueError, match="outside the allowed directory"):
            check_origin_param_confinement({"importer": "server_folder", "params": {"path": token}})

    def test_non_string_params_are_ignored(self, confined_base):
        check_origin_param_confinement(
            {"importer": "local_archive_member", "params": {"clip_start": 1.5, "thin": True, "n": None}}
        )

    def test_nested_container_params_are_recursed(self, confined_base):
        with pytest.raises(ValueError, match="outside the allowed directory"):
            check_origin_param_confinement({"importer": "server_files", "params": {"paths": ["ok.wav", ".."]}})
        with pytest.raises(ValueError, match="outside the allowed directory"):
            check_origin_param_confinement({"importer": "custom", "params": {"nested": {"path": "/etc/passwd"}}})

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
