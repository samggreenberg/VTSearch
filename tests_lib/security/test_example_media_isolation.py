"""The library tier's readers never see the checkout's ``data/example_media/`` (#4271).

Media seeding, label building and the ``example_media`` sentinel resolver all
read through ``example_media_dir()``.  The shared autouse
``isolated_example_media_dir`` fixture points it at ``tmp_path`` here too, so a
file a developer left in the real directory cannot change what a test sees.
"""

from __future__ import annotations


def test_resolver_points_under_tmp_path(tmp_path):
    from vtscore.config import DATA_DIR
    from vtscore.security.path_validation import example_media_dir

    assert example_media_dir() == tmp_path / "example_media"
    assert example_media_dir() != DATA_DIR / "example_media"
