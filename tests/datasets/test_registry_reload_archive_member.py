"""End-to-end regression: reloading an archive-member dataset from the registry.

``POST /api/datasets/registry/<id>/load`` reads the saved ``.pkl`` in *full*
mode.  Items whose bytes live only as a byte range inside an unextracted tar
shard (``local_archive_member`` - audio tiles, video windows) carry no inline
bytes and no ``media_path``, so full mode used to drop every one of them: the
reload produced a context with 0 medias, the registry row's ``num_items`` was
overwritten with 0, and the next Browse attempt 409'd with "Dataset is empty -
nothing to project".
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


def _archive_member_media(media_id: int, tmp_path: Path) -> dict[str, Any]:
    """One archive-member audio media: bytes exist only inside a tar shard.

    Carries a top-level playback window too (each item is one window of its
    member): the whole member is served, so the player seeks to ``clip_start``
    and loops within ``[clip_start, clip_end]``, and losing the window on
    reload would collapse every window of a member into the same item.
    """
    return {
        "id": media_id,
        "media_type": "audio",
        "duration": 0,
        "file_size": 4096,
        "md5": f"md5-{media_id}",
        "embeddings": {"clap": np.zeros(512, dtype=np.float32)},
        "embedder": "clap",
        "filename": f"shard0/clip{media_id}.m4a",
        "category": "custom",
        "origin": {
            "importer": "local_archive_member",
            "params": {
                "archive_path": str(tmp_path / "shard0.tar"),
                "member": f"clip{media_id}.m4a",
                "media_type": "audio",
                "clip_start": float(media_id * 10),
                "clip_end": float(media_id * 10 + 10),
            },
        },
        "origin_name": f"shard0.tar::clip{media_id}.m4a",
        "clip_start": float(media_id * 10),
        "clip_end": float(media_id * 10 + 10),
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
    }


def _wait_for_task(task_id: str, timeout: float = 12.0) -> str | None:
    """Poll the loading-task tracker until it reports ``idle``."""
    from vtscore.concurrency.progress import loading_tasks

    status = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = next((t for t in loading_tasks.list_tasks() if t["task_id"] == task_id), None)
        if task is not None:
            status = task["status"]
            if status == "idle":
                return status
        time.sleep(0.1)
    return status


class TestRegistryReloadArchiveMemberDataset:
    def test_reload_keeps_every_item(self, client, tmp_path):
        from vtscore.datasets.container import write_container
        from vtscore.datasets.registry import get_dataset, register_dataset, unregister_dataset
        from vtscore.state.core import get_context
        from vtsearch.settings import get_saved_datasets_dir

        medias = {i: _archive_member_media(i, tmp_path) for i in (1, 2, 3)}
        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = ds_dir / "test_archive_member_reload.pkl"
        write_container(pkl_path, pickle.dumps({"medias": medias}), {"format_version": 1})

        entry = register_dataset(
            name="archive member reload",
            media_type="audio",
            num_items=len(medias),
            pkl_path=str(pkl_path),
            embedder="clap",
            created_by="default",
        )
        dataset_id = entry["id"]
        try:
            resp = client.post(f"/api/datasets/registry/{dataset_id}/load")
            assert resp.status_code == 200
            task_id = resp.get_json()["task_id"]
            assert _wait_for_task(task_id) == "idle"

            ctx = get_context(dataset_id)
            assert ctx is not None
            assert len(ctx.medias) == len(medias)
            # The registry stat must not be overwritten with 0.
            reloaded_entry = get_dataset(dataset_id)
            assert reloaded_entry is not None
            assert reloaded_entry["num_items"] == len(medias)
            # Kept lazily: the member is streamed from its shard on demand.
            assert all(m["media_bytes"] is None for m in ctx.medias.values())
            # Each item's playback window survives, so the windows of a shared
            # member still sound (and draw) like different stretches of audio.
            assert sorted((m["clip_start"], m["clip_end"]) for m in ctx.medias.values()) == [
                (10.0, 20.0),
                (20.0, 30.0),
                (30.0, 40.0),
            ]
        finally:
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            unregister_dataset(dataset_id)
            pkl_path.unlink(missing_ok=True)
