"""``DemoDataset.requires_download`` marks the demos that fetch nothing (#5546).

The Synthetic World Map demos generate their media at load time, so the app's
demo list must not call them "Needs Download" before their first load.  The
list decides that from this declared field, never from ``download_size_mb == 0``
(an advertised size that several hub-fetched demos also leave at 0).
"""

from __future__ import annotations

from vtscore.media import all_demo_datasets, all_types
from vtscore.media._toponymy_demo import SOURCE_ID


def _all_demos():
    return [ds for mt in all_types() for ds in mt.demo_datasets]


def test_defaults_to_requiring_a_download():
    from vtscore.media.base import DemoDataset

    assert DemoDataset(id="x", label="X", description="", categories=[]).requires_download is True


def test_both_synthetic_world_demos_fetch_nothing():
    synthetic = {ds.id: ds for ds in _all_demos() if ds.source == SOURCE_ID}
    assert set(synthetic) == {"synthetic_world_image", "synthetic_world_audio"}
    assert all(ds.requires_download is False for ds in synthetic.values())


def test_a_demo_that_fetches_nothing_declares_no_download_and_no_source_folder():
    """``requires_download=False`` must agree with the demo's other fields: nothing to fetch, nowhere to fetch it to."""
    for ds in _all_demos():
        if ds.requires_download:
            continue
        assert ds.download_size_mb == 0, f"{ds.id} fetches nothing but advertises a download"
        assert ds.required_folder is None, f"{ds.id} fetches nothing but declares a source folder"


def test_all_demo_datasets_carries_the_flag():
    catalogue = all_demo_datasets()
    assert catalogue["synthetic_world_image"]["requires_download"] is False
    assert catalogue["synthetic_world_audio"]["requires_download"] is False
    assert all(isinstance(info["requires_download"], bool) for info in catalogue.values())
