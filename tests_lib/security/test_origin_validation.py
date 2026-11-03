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
        confine_origin_params({"importer": "server_folder", "params": {"path": ".."}})


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

    def test_plain_relative_params_pass(self, confined_base):
        # Opaque params (a cache filename, a media type, an archive member
        # key) resolve inside the user's own dir, so validating them is
        # harmless — the check errs towards validating too much.
        confine_origin_params({"importer": "example_media", "params": {"filename": "abc.wav"}})
        confine_origin_params(
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
            confine_origin_params({"importer": "server_folder", "params": {"path": token}})

    def test_non_string_params_are_ignored(self, confined_base):
        confine_origin_params(
            {"importer": "local_archive_member", "params": {"clip_start": 1.5, "thin": True, "n": None}}
        )

    def test_nested_container_params_are_recursed(self, confined_base):
        with pytest.raises(ValueError, match="outside the allowed directory"):
            confine_origin_params({"importer": "server_files", "params": {"paths": ["ok.wav", ".."]}})
        with pytest.raises(ValueError, match="outside the allowed directory"):
            confine_origin_params({"importer": "custom", "params": {"nested": {"path": "/etc/passwd"}}})

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
    while the consuming source would anchor it at the CWD, so the caller must
    resolve the *returned* origin, not the one it passed in."""

    def test_relative_path_param_comes_back_as_the_approved_path(self, confined_base):
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

    def test_manifest_and_paths_file_are_confined(self, confined_base):
        out = confine_origin_params({"importer": "local_archive_member", "params": {"manifest": "m.json"}})
        assert out["params"]["manifest"] == str((confined_base / "m.json").resolve())
        out = confine_origin_params({"importer": "server_files", "params": {"paths_file": "p.txt"}})
        assert out["params"]["paths_file"] == str((confined_base / "p.txt").resolve())

    def test_opaque_params_are_checked_but_never_rewritten(self, confined_base):
        """Only the keys the source factories open as files are rewritten; an
        archive member key or a media type would be destroyed by it."""
        out = confine_origin_params(
            {
                "importer": "local_archive_member",
                "params": {"member": "sounds/clip.wav", "media_type": "audio", "clip_start": 1.5},
            }
        )
        assert out["params"] == {"member": "sounds/clip.wav", "media_type": "audio", "clip_start": 1.5}

    def test_url_params_pass_through_unchanged(self, confined_base):
        url = "https://media.example.test/a.wav"
        out = confine_origin_params({"importer": "url_download", "params": {"url": url}})
        assert out["params"]["url"] == url

    def test_unconfined_returns_params_verbatim(self, monkeypatch):
        import vtscore.security.path_validation as paths_mod

        monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: None)
        out = confine_origin_params({"importer": "server_folder", "params": {"path": "data/sounds"}})
        assert out["params"]["path"] == "data/sounds"
