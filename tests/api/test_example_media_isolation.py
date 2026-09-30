"""Tests never write exemplars into the checkout's ``data/example_media/`` (#4271).

In single-user mode ``example_media_dir()`` resolves to ``DATA_DIR /
"example_media"``, the directory the app lists as server example media.
Before the shared autouse ``isolated_example_media_dir`` fixture, every
exemplar a test uploaded or wrote there survived the run and showed up in the
Load Sort dialog, and so in screenshots reshot after a test run.
"""

from __future__ import annotations

import io


class TestExampleMediaStaysOutOfTheCheckout:
    def test_resolver_points_under_tmp_path(self, tmp_path):
        from vtscore.config import DATA_DIR
        from vtscore.security.path_validation import example_media_dir

        assert example_media_dir() == tmp_path / "example_media"
        assert example_media_dir() != DATA_DIR / "example_media"

    def test_uploaded_exemplar_lands_in_tmp_path(self, client, tmp_path):
        """Drive a real writer and check where its bytes went.

        The upload route names the file with a fresh uuid, so its absence
        from the checkout's directory is exact even when that directory
        already holds files from the developer's own use of the app.
        """
        from vtscore.config import DATA_DIR

        resp = client.post(
            "/api/server-media-files/upload",
            data={"file": (io.BytesIO(b"RIFF" + b"\x00" * 80), "exemplar.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201, resp.get_json()
        filename = resp.get_json()["filename"]

        assert (tmp_path / "example_media" / filename).is_file()
        assert not (DATA_DIR / "example_media" / filename).exists()
