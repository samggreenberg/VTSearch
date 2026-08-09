"""Tests for per-user confinement of a media's own file references (issue #2926).

``media_path`` and the archive-member archive path arrive from a dataset
pickle, so in a multi-user deployment they are attacker-supplied: without a
check, a pickled ``media_path`` of ``/etc/shadow`` is read and served straight
back by the media routes.  :func:`vtscore.security.path_validation.
resolve_media_file_path` is the single guard every such read goes through.

The allowed roots are the user's data dir **and** ``DATA_DIR`` — demo datasets
extract into the latter as siblings of the per-user dirs, so a thin demo
dataset's ``media_path`` legitimately sits outside ``data/<username>/``.
"""

from __future__ import annotations

import pytest

from vtscore.security.path_validation import resolve_media_file_path


@pytest.fixture
def single_user(monkeypatch):
    """Simulate single-user / no-auth mode: no confinement at all."""
    import vtscore.security.path_validation as paths_mod

    monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: None)


@pytest.fixture
def multi_user(monkeypatch, tmp_path):
    """Simulate multi-user mode; returns ``(user_dir, shared_data_dir)``."""
    import vtscore.security.path_validation as paths_mod

    user_dir = tmp_path / "data" / "alice"
    shared = tmp_path / "data"
    user_dir.mkdir(parents=True)
    monkeypatch.setattr(paths_mod, "get_file_access_base_dir", lambda: user_dir)
    monkeypatch.setattr(paths_mod, "DATA_DIR", shared)
    return user_dir, shared


def _audio_type():
    from vtscore.media import get as get_media_type

    return get_media_type("audio")


def _text_type():
    from vtscore.media import get as get_media_type

    return get_media_type("text")


# ---------------------------------------------------------------------------
# resolve_media_file_path
# ---------------------------------------------------------------------------


class TestSingleUserMode:
    def test_any_path_is_allowed(self, single_user):
        assert resolve_media_file_path("/etc/shadow") is not None

    def test_path_is_returned_unchanged(self, single_user):
        from pathlib import Path

        assert resolve_media_file_path("/srv/media/clip.wav") == Path("/srv/media/clip.wav")


class TestMultiUserConfinement:
    def test_path_inside_user_dir_is_allowed(self, multi_user):
        user_dir, _ = multi_user
        inside = user_dir / "clip.wav"
        assert resolve_media_file_path(str(inside)) == inside

    def test_path_in_shared_data_dir_is_allowed(self, multi_user):
        """Demo datasets extract into DATA_DIR, not the per-user subtree."""
        _, shared = multi_user
        demo = shared / "ESC-50-master" / "audio" / "1-137-A-32.wav"
        assert resolve_media_file_path(str(demo)) == demo

    @pytest.mark.parametrize("escaping", ["/etc/shadow", "/etc/passwd", "/root/.ssh/id_rsa"])
    def test_path_outside_the_data_tree_is_refused(self, multi_user, escaping):
        assert resolve_media_file_path(escaping) is None

    def test_traversal_out_of_the_user_dir_is_refused(self, multi_user):
        user_dir, _ = multi_user
        assert resolve_media_file_path(str(user_dir / ".." / ".." / ".." / "etc" / "shadow")) is None

    def test_symlink_escape_is_refused(self, multi_user, tmp_path):
        user_dir, _ = multi_user
        secret = tmp_path / "outside" / "secret.txt"
        secret.parent.mkdir()
        secret.write_text("classified")
        (user_dir / "escape").symlink_to(secret.parent)
        assert resolve_media_file_path(str(user_dir / "escape" / "secret.txt")) is None


# ---------------------------------------------------------------------------
# The media resolution chain honours the guard
# ---------------------------------------------------------------------------


class TestResolveMediaBytesConfinement:
    def test_escaping_media_path_serves_nothing(self, multi_user, tmp_path):
        secret = tmp_path / "outside_secret"
        secret.write_bytes(b"root:x:0:0:")
        media = {"media_path": str(secret), "filename": "passwd"}
        assert _audio_type()._resolve_media_bytes(media) is None

    def test_confined_media_path_still_serves(self, multi_user):
        user_dir, _ = multi_user
        clip = user_dir / "clip.wav"
        clip.write_bytes(b"RIFFfake")
        assert _audio_type()._resolve_media_bytes({"media_path": str(clip)}) == b"RIFFfake"

    def test_shared_demo_media_still_serves(self, multi_user):
        _, shared = multi_user
        demo = shared / "ESC-50-master" / "audio"
        demo.mkdir(parents=True)
        clip = demo / "1-137-A-32.wav"
        clip.write_bytes(b"RIFFdemo")
        assert _audio_type()._resolve_media_bytes({"media_path": str(clip)}) == b"RIFFdemo"

    def test_single_user_mode_is_unaffected(self, single_user, tmp_path):
        outside = tmp_path / "anywhere.wav"
        outside.write_bytes(b"RIFFany")
        assert _audio_type()._resolve_media_bytes({"media_path": str(outside)}) == b"RIFFany"


class TestResolveMediaStringConfinement:
    def test_escaping_media_path_serves_nothing(self, multi_user, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        assert _text_type()._resolve_media_string({"media_path": str(secret)}) == ""

    def test_confined_media_path_still_serves(self, multi_user):
        user_dir, _ = multi_user
        doc = user_dir / "note.txt"
        doc.write_text("hello")
        assert _text_type()._resolve_media_string({"media_path": str(doc)}) == "hello"


class TestLazyClipSourceConfinement:
    """A clip recipe must not be a way around the ``media_path`` guard."""

    def test_escaping_source_path_reads_nothing(self, multi_user, tmp_path):
        from vtscore.media.lazy_clip import _read_source_bytes

        secret = tmp_path / "secret.bin"
        secret.write_bytes(b"classified")
        assert _read_source_bytes({"media_path": str(secret)}) is None

    def test_confined_source_path_is_read(self, multi_user):
        from vtscore.media.lazy_clip import _read_source_bytes

        user_dir, _ = multi_user
        source = user_dir / "source.wav"
        source.write_bytes(b"RIFFsource")
        assert _read_source_bytes({"media_path": str(source)}) == b"RIFFsource"


class TestArchiveMemberConfinement:
    def test_escaping_archive_path_reads_nothing(self, multi_user, tmp_path):
        from vtscore.media.base import _resolve_archive_member_bytes

        media = {"archive_member": {"path": str(tmp_path / "outside.zip"), "member": "a.wav"}}
        assert _resolve_archive_member_bytes(media) is None
